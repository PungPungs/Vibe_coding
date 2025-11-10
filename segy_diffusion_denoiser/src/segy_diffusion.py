"""
SEG-Y Diffusion Denoiser
Applies diffusion models to add and remove noise from seismic traces
"""

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet1DModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import struct
import mmap
import os
from typing import Optional, Tuple


class SegyReader:
    """Simplified SEG-Y reader for loading seismic data"""

    def __init__(self):
        self.data: Optional[np.ndarray] = None
        self.num_traces: int = 0
        self.num_samples: int = 0
        self.sample_rate: float = 0.0

    def load_file(self, filename: str) -> bool:
        """Load SEG-Y file"""
        try:
            with open(filename, 'rb') as f:
                mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

                # Read number of samples
                self.num_samples = struct.unpack('>H', mmap_obj[3220:3222])[0]

                # Read sample interval
                sample_interval_us = struct.unpack('>H', mmap_obj[3216:3218])[0]
                self.sample_rate = sample_interval_us / 1_000_000.0

                # Calculate number of traces
                file_size = os.path.getsize(filename)
                header_size = 3600
                trace_size = 240 + self.num_samples * 4
                self.num_traces = (file_size - header_size) // trace_size

                # Load data
                self.data = np.zeros((self.num_samples, self.num_traces), dtype=np.float32)

                for i in range(self.num_traces):
                    trace_offset = header_size + i * trace_size + 240
                    trace_bytes = mmap_obj[trace_offset:trace_offset + self.num_samples * 4]
                    trace_data = self._ibm_to_float(trace_bytes, self.num_samples)
                    self.data[:, i] = trace_data

                mmap_obj.close()

                # Normalize
                self._normalize_data()

                return True

        except Exception as e:
            print(f"Error loading SEG-Y: {e}")
            return False

    def _ibm_to_float(self, data: bytes, count: int) -> np.ndarray:
        """Convert IBM float to IEEE float"""
        result = np.zeros(count, dtype=np.float32)

        for i in range(count):
            ibm_bytes = data[i*4:(i+1)*4]
            if len(ibm_bytes) < 4:
                break

            ibm_int = struct.unpack('>I', ibm_bytes)[0]

            sign = (ibm_int >> 31) & 1
            exponent = (ibm_int >> 24) & 0x7f
            mantissa = ibm_int & 0x00ffffff

            if mantissa == 0:
                result[i] = 0.0
            else:
                try:
                    value = mantissa / 16777216.0
                    exp_val = exponent - 64

                    if -60 <= exp_val <= 60:
                        value *= (16.0 ** exp_val)
                        if sign:
                            value = -value
                        if np.isfinite(value):
                            result[i] = np.float32(value)
                except:
                    result[i] = 0.0

        return result

    def _normalize_data(self):
        """Normalize data per trace"""
        if self.data is not None:
            self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)

            for i in range(self.num_traces):
                trace = self.data[:, i]
                max_abs = np.max(np.abs(trace))
                if max_abs > 0 and np.isfinite(max_abs):
                    self.data[:, i] = trace / max_abs


class SimpleDenoisingModel(nn.Module):
    """Simple 1D U-Net style denoising model for traces"""

    def __init__(self, trace_length=1000):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder with skip connections
        u2 = self.up2(b)
        # Adjust size if needed
        if u2.shape[2] != e2.shape[2]:
            u2 = torch.nn.functional.interpolate(u2, size=e2.shape[2], mode='linear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[2] != e1.shape[2]:
            u1 = torch.nn.functional.interpolate(u1, size=e1.shape[2], mode='linear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return d1


class SegyDiffusionProcessor:
    """Process SEG-Y traces with diffusion models"""

    def __init__(self):
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        self.denoising_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_noise_to_trace(self, trace: np.ndarray, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add noise to a seismic trace

        Args:
            trace: Input trace (1D numpy array)
            timestep: Diffusion timestep (0-999)

        Returns:
            (noisy_trace, noise_added)
        """
        # Convert to tensor
        trace_tensor = torch.from_numpy(trace).float().unsqueeze(0).unsqueeze(0)

        # Generate noise
        noise = torch.randn_like(trace_tensor)

        # Add noise
        timestep_tensor = torch.tensor([timestep])
        noisy_trace = self.scheduler.add_noise(trace_tensor, noise, timestep_tensor)

        return noisy_trace.squeeze().numpy(), noise.squeeze().numpy()

    def denoise_trace_simple(self, noisy_trace: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Simple denoising using wavelet-like filtering

        Args:
            noisy_trace: Noisy trace
            noise_level: Amount of denoising (0-1)

        Returns:
            Denoised trace
        """
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter1d

        # Adaptive sigma based on noise level
        sigma = noise_level * 5.0
        denoised = gaussian_filter1d(noisy_trace, sigma=sigma)

        return denoised

    def denoise_trace_diffusion(self, noisy_trace: np.ndarray, num_steps: int = 50) -> np.ndarray:
        """
        Denoise using trained diffusion model (placeholder - would need training)

        Args:
            noisy_trace: Noisy trace
            num_steps: Number of denoising steps

        Returns:
            Denoised trace
        """
        # For now, use simple denoising as placeholder
        # In a real implementation, this would use a trained model
        return self.denoise_trace_simple(noisy_trace, noise_level=0.2)


class SegyDiffusionVisualizer:
    """Main visualizer class"""

    def __init__(self):
        self.segy_reader = None
        self.processor = SegyDiffusionProcessor()
        self.current_data = None

    def load_segy(self, file_path: str):
        """Load SEG-Y file"""
        if file_path is None:
            return None, "Please upload a SEG-Y file"

        self.segy_reader = SegyReader()
        success = self.segy_reader.load_file(file_path)

        if success:
            self.current_data = self.segy_reader.data
            info = f"Loaded: {self.segy_reader.num_traces} traces, {self.segy_reader.num_samples} samples"

            # Create preview plot
            fig = self.plot_section(self.current_data, "Original Data")

            return fig, info
        else:
            return None, "Failed to load SEG-Y file"

    def add_noise_visualization(self, trace_index: int, timestep: int):
        """Add noise to a specific trace"""
        if self.current_data is None:
            return None, "Please load SEG-Y file first"

        trace_index = int(trace_index)
        if trace_index >= self.segy_reader.num_traces:
            trace_index = self.segy_reader.num_traces - 1

        # Get original trace
        original_trace = self.current_data[:, trace_index]

        # Add noise
        noisy_trace, noise = self.processor.add_noise_to_trace(original_trace, timestep)

        # Create comparison plot
        fig = self.plot_trace_comparison(original_trace, noisy_trace, noise, timestep)

        noise_level = self.processor.scheduler.alphas_cumprod[timestep].item()
        info = f"Trace {trace_index}, Timestep {timestep}\nNoise level: {(1-noise_level)*100:.1f}%"

        return fig, info

    def denoise_trace_visualization(self, trace_index: int, noise_amount: float, denoise_strength: float):
        """Denoise a trace"""
        if self.current_data is None:
            return None, "Please load SEG-Y file first"

        trace_index = int(trace_index)
        if trace_index >= self.segy_reader.num_traces:
            trace_index = self.segy_reader.num_traces - 1

        # Get original trace
        original_trace = self.current_data[:, trace_index]

        # Add noise first
        timestep = int(noise_amount * 999)
        noisy_trace, _ = self.processor.add_noise_to_trace(original_trace, timestep)

        # Denoise
        denoised_trace = self.processor.denoise_trace_simple(noisy_trace, denoise_strength)

        # Create comparison plot
        fig = self.plot_denoise_comparison(original_trace, noisy_trace, denoised_trace)

        info = f"Trace {trace_index}\nNoise: {noise_amount*100:.1f}%, Denoise: {denoise_strength*100:.1f}%"

        return fig, info

    def plot_section(self, data: np.ndarray, title: str) -> Figure:
        """Plot seismic section"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot only subset if too large
        max_traces = 200
        if data.shape[1] > max_traces:
            step = data.shape[1] // max_traces
            plot_data = data[:, ::step]
        else:
            plot_data = data

        im = ax.imshow(plot_data, aspect='auto', cmap='seismic',
                      vmin=-1, vmax=1, interpolation='bilinear')
        ax.set_xlabel('Trace Number')
        ax.set_ylabel('Sample')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Amplitude')

        plt.tight_layout()
        return fig

    def plot_trace_comparison(self, original: np.ndarray, noisy: np.ndarray,
                              noise: np.ndarray, timestep: int) -> Figure:
        """Plot original vs noisy trace"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        time_axis = np.arange(len(original))

        # Original
        axes[0].plot(original, time_axis, 'b-', linewidth=0.8)
        axes[0].set_title('Original Trace')
        axes[0].set_ylabel('Sample')
        axes[0].set_xlabel('Amplitude')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3)

        # Noisy
        axes[1].plot(noisy, time_axis, 'r-', linewidth=0.8)
        axes[1].set_title(f'Noisy Trace (t={timestep})')
        axes[1].set_ylabel('Sample')
        axes[1].set_xlabel('Amplitude')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3)

        # Noise component
        axes[2].plot(noise, time_axis, 'gray', linewidth=0.8, alpha=0.7)
        axes[2].set_title('Noise Added')
        axes[2].set_ylabel('Sample')
        axes[2].set_xlabel('Amplitude')
        axes[2].invert_yaxis()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_denoise_comparison(self, original: np.ndarray, noisy: np.ndarray,
                                denoised: np.ndarray) -> Figure:
        """Plot denoising results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        time_axis = np.arange(len(original))

        # Original
        axes[0].plot(original, time_axis, 'b-', linewidth=0.8)
        axes[0].set_title('Original Trace')
        axes[0].set_ylabel('Sample')
        axes[0].set_xlabel('Amplitude')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3)

        # Noisy
        axes[1].plot(noisy, time_axis, 'r-', linewidth=0.8)
        axes[1].set_title('Noisy Trace')
        axes[1].set_ylabel('Sample')
        axes[1].set_xlabel('Amplitude')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3)

        # Denoised
        axes[2].plot(denoised, time_axis, 'g-', linewidth=0.8)
        axes[2].plot(original, time_axis, 'b--', linewidth=0.5, alpha=0.5, label='Original')
        axes[2].set_title('Denoised Trace')
        axes[2].set_ylabel('Sample')
        axes[2].set_xlabel('Amplitude')
        axes[2].invert_yaxis()
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        return fig


def create_interface():
    """Create Gradio interface"""
    visualizer = SegyDiffusionVisualizer()

    with gr.Blocks(title="SEG-Y Diffusion Denoiser") as demo:
        gr.Markdown("""
        # 🌊 SEG-Y Diffusion Denoiser

        Load SEG-Y seismic data and visualize the diffusion process:
        - **Add Noise**: See how noise corrupts seismic traces (forward diffusion)
        - **Remove Noise**: Denoise traces using diffusion-based methods (reverse diffusion)
        """)

        with gr.Row():
            segy_file = gr.File(label="Upload SEG-Y File", file_types=[".sgy", ".segy"])
            load_btn = gr.Button("Load File", variant="primary")

        with gr.Row():
            preview_plot = gr.Plot(label="Data Preview")
            load_info = gr.Textbox(label="File Info", lines=3)

        load_btn.click(
            fn=visualizer.load_segy,
            inputs=[segy_file],
            outputs=[preview_plot, load_info]
        )

        with gr.Tab("Add Noise (Forward Diffusion)"):
            gr.Markdown("Visualize how noise is added to clean seismic traces")

            with gr.Row():
                with gr.Column():
                    add_trace_idx = gr.Slider(0, 100, value=0, step=1, label="Trace Index")
                    add_timestep = gr.Slider(0, 999, value=0, step=1,
                                            label="Timestep (0=clean, 999=noise)")
                    add_noise_btn = gr.Button("Add Noise", variant="primary")

                with gr.Column():
                    add_noise_plot = gr.Plot(label="Noise Addition")
                    add_noise_info = gr.Textbox(label="Info", lines=2)

            add_noise_btn.click(
                fn=visualizer.add_noise_visualization,
                inputs=[add_trace_idx, add_timestep],
                outputs=[add_noise_plot, add_noise_info]
            )

            add_timestep.change(
                fn=visualizer.add_noise_visualization,
                inputs=[add_trace_idx, add_timestep],
                outputs=[add_noise_plot, add_noise_info]
            )

        with gr.Tab("Remove Noise (Reverse Diffusion)"):
            gr.Markdown("Denoise corrupted seismic traces")

            with gr.Row():
                with gr.Column():
                    denoise_trace_idx = gr.Slider(0, 100, value=0, step=1, label="Trace Index")
                    noise_amount = gr.Slider(0.0, 1.0, value=0.3, step=0.01,
                                            label="Noise Amount to Add First")
                    denoise_strength = gr.Slider(0.0, 1.0, value=0.2, step=0.01,
                                                 label="Denoising Strength")
                    denoise_btn = gr.Button("Denoise Trace", variant="primary")

                with gr.Column():
                    denoise_plot = gr.Plot(label="Denoising Result")
                    denoise_info = gr.Textbox(label="Info", lines=2)

            denoise_btn.click(
                fn=visualizer.denoise_trace_visualization,
                inputs=[denoise_trace_idx, noise_amount, denoise_strength],
                outputs=[denoise_plot, denoise_info]
            )

        gr.Markdown("""
        ### How it works:

        **Forward Diffusion (Add Noise)**:
        - Gradually adds Gaussian noise to clean seismic traces
        - Uses DDPM noise schedule
        - Demonstrates data corruption process

        **Reverse Diffusion (Remove Noise)**:
        - Applies denoising to recover clean signal
        - Currently uses Gaussian filtering (simple method)
        - Can be extended with trained neural networks

        **Applications**:
        - Seismic data quality enhancement
        - Noise suppression in marine seismic
        - Data augmentation for training
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)
