"""
SEG-Y Diffusion Denoiser - RGB Visualization
Visualizes seismic sections as RGB images and applies diffusion models
"""

import gradio as gr
import numpy as np
import torch
from diffusers import DDPMScheduler
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import struct
import mmap
import os
from typing import Optional, Tuple
from io import BytesIO


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
        """Normalize data globally"""
        if self.data is not None:
            self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)

            # Global normalization for better visualization
            max_abs = np.percentile(np.abs(self.data), 99)  # Use 99th percentile to avoid outliers
            if max_abs > 0:
                self.data = self.data / max_abs
                self.data = np.clip(self.data, -1, 1)


class SegyToRGBConverter:
    """Convert seismic data to RGB images"""

    def __init__(self):
        self.colormaps = {
            'seismic': plt.cm.seismic,
            'gray': plt.cm.gray,
            'viridis': plt.cm.viridis,
            'RdBu': plt.cm.RdBu_r,
        }

    def data_to_rgb(self, data: np.ndarray, colormap: str = 'seismic',
                    max_size: int = 512) -> Image.Image:
        """
        Convert seismic data to RGB image

        Args:
            data: Seismic data (samples x traces)
            colormap: Matplotlib colormap name
            max_size: Maximum image dimension

        Returns:
            PIL RGB Image
        """
        # Resize if too large
        h, w = data.shape
        if max(h, w) > max_size:
            ratio = max_size / max(h, w)
            new_h = int(h * ratio)
            new_w = int(w * ratio)

            # Use PIL for high-quality resizing
            temp_img = self._array_to_image(data, colormap)
            temp_img = temp_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return temp_img

        return self._array_to_image(data, colormap)

    def _array_to_image(self, data: np.ndarray, colormap: str) -> Image.Image:
        """Convert numpy array to PIL Image using colormap"""
        # Normalize to [0, 1]
        normalized = (data + 1) / 2.0  # From [-1, 1] to [0, 1]
        normalized = np.clip(normalized, 0, 1)

        # Apply colormap
        cmap = self.colormaps.get(colormap, plt.cm.seismic)
        rgba = cmap(normalized)

        # Convert to RGB (drop alpha channel)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        return Image.fromarray(rgb)

    def rgb_to_data(self, image: Image.Image, colormap: str = 'seismic') -> np.ndarray:
        """
        Convert RGB image back to seismic data (approximate)

        Args:
            image: PIL RGB Image
            colormap: Colormap used for encoding

        Returns:
            Seismic data array
        """
        # Convert to grayscale as approximation
        img_array = np.array(image.convert('L')).astype(np.float32)

        # Normalize back to [-1, 1]
        data = (img_array / 255.0) * 2.0 - 1.0

        return data


class RGBDiffusionProcessor:
    """Process RGB seismic images with diffusion models"""

    def __init__(self):
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor normalized to [-1, 1]"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        # Normalize to [-1, 1]
        img_tensor = (img_tensor - 0.5) * 2.0

        return img_tensor

    def postprocess_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor back to PIL image"""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor / 2.0 + 0.5).clamp(0, 1)

        img_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def add_noise(self, image_tensor: torch.Tensor, timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to image at specific timestep"""
        noise = torch.randn_like(image_tensor)
        timestep_tensor = torch.tensor([timestep])
        noisy_image = self.scheduler.add_noise(image_tensor, noise, timestep_tensor)

        return noisy_image, noise

    def denoise_simple(self, noisy_image: Image.Image, strength: float = 0.3) -> Image.Image:
        """
        Simple denoising using Gaussian blur

        Args:
            noisy_image: Noisy PIL Image
            strength: Denoising strength (0-1)

        Returns:
            Denoised PIL Image
        """
        from PIL import ImageFilter

        # Apply Gaussian blur
        radius = strength * 3.0
        denoised = noisy_image.filter(ImageFilter.GaussianBlur(radius=radius))

        return denoised

    def denoise_iterative(self, noisy_image: Image.Image, start_timestep: int,
                         num_steps: int = 50) -> list:
        """
        Simulate iterative denoising process

        Args:
            noisy_image: Starting noisy image
            start_timestep: Initial noise level
            num_steps: Number of denoising steps

        Returns:
            List of images at each denoising step
        """
        images = [noisy_image]
        current_img = noisy_image

        # Simple iterative denoising (placeholder for real diffusion model)
        step_size = start_timestep / num_steps

        for i in range(num_steps):
            strength = (num_steps - i) / num_steps * 0.1
            current_img = self.denoise_simple(current_img, strength)
            if i % (num_steps // 10) == 0:  # Save every 10% progress
                images.append(current_img.copy())

        return images


class SegyDiffusionVisualizerRGB:
    """Main visualizer class for RGB-based diffusion"""

    def __init__(self):
        self.segy_reader = None
        self.converter = SegyToRGBConverter()
        self.processor = RGBDiffusionProcessor()
        self.current_data = None
        self.current_rgb = None

    def load_segy(self, file_path: str, colormap: str = 'seismic',
                  trace_range: Tuple[int, int] = None):
        """Load SEG-Y file and convert to RGB"""
        if file_path is None:
            return None, None, "Please upload a SEG-Y file"

        self.segy_reader = SegyReader()
        success = self.segy_reader.load_file(file_path)

        if success:
            self.current_data = self.segy_reader.data

            # Select trace range if specified
            if trace_range:
                start, end = trace_range
                end = min(end, self.segy_reader.num_traces)
                data_subset = self.current_data[:, start:end]
            else:
                # Use all traces (or limit for performance)
                max_traces = 500
                if self.segy_reader.num_traces > max_traces:
                    data_subset = self.current_data[:, :max_traces]
                else:
                    data_subset = self.current_data

            # Convert to RGB
            self.current_rgb = self.converter.data_to_rgb(data_subset, colormap)

            # Create bathymetric-style preview
            preview_image = self._create_bathymetric_preview(data_subset, colormap)

            info = f"Loaded: {self.segy_reader.num_traces} traces, {self.segy_reader.num_samples} samples\n"
            info += f"RGB Image: {self.current_rgb.size[0]}x{self.current_rgb.size[1]}"

            return preview_image, self.current_rgb, info
        else:
            return None, None, "Failed to load SEG-Y file"

    def _create_bathymetric_preview(self, data: np.ndarray, colormap: str) -> Image.Image:
        """Create a seismic interpretation style preview like professional seismic sections"""
        fig = plt.figure(figsize=(16, 8))
        ax = plt.subplot(111)

        # Create extent for proper aspect ratio
        num_samples, num_traces = data.shape
        extent = [0, num_traces, num_samples * self.segy_reader.sample_rate * 1000, 0]

        # Use grayscale for seismic data (like the reference image)
        im = ax.imshow(data, aspect='auto', cmap='gray',
                      extent=extent, interpolation='bilinear', vmin=-1, vmax=1)

        # Detect seafloor (first strong reflection) - simplified approach
        seafloor_line = self._detect_seafloor(data)

        # Plot seafloor line in red
        trace_indices = np.arange(num_traces)
        seafloor_times = seafloor_line * self.segy_reader.sample_rate * 1000
        ax.plot(trace_indices, seafloor_times, 'r-', linewidth=2.5, label='Seafloor', zorder=10)

        # Add geological interpretation lines (black lines following strong reflectors)
        horizon_lines = self._detect_horizons(data, num_horizons=3)
        for idx, horizon in enumerate(horizon_lines):
            horizon_times = horizon * self.segy_reader.sample_rate * 1000
            ax.plot(trace_indices, horizon_times, 'k-', linewidth=1.5,
                   alpha=0.7, zorder=9)

        # Add vertical dashed grid lines (like reference image)
        for i in range(0, num_traces, num_traces // 5):
            ax.axvline(x=i, color='brown', linestyle='--', linewidth=0.8, alpha=0.5)

        # Add horizontal dashed grid lines for depth markers
        max_time = num_samples * self.segy_reader.sample_rate * 1000
        depth_markers = [25, 50, 75]  # in meters equivalent
        for depth in depth_markers:
            if depth < max_time:
                ax.axhline(y=depth, color='brown', linestyle='--', linewidth=0.8, alpha=0.5)
                ax.text(-num_traces*0.02, depth, f'{depth} m',
                       fontsize=9, color='red', va='center')

        # Add coordinate labels at top (like reference image)
        coord_positions = np.linspace(0, num_traces, 6, dtype=int)
        for pos in coord_positions:
            coord_label = f'{5607000 + pos*100}'  # Sample coordinates
            ax.text(pos, -max_time*0.05, coord_label,
                   fontsize=8, color='red', ha='center', rotation=90)

        # Labels and styling
        ax.set_xlabel('Trace Number', fontsize=11)
        ax.set_ylabel('Time (ms) / Depth', fontsize=11)
        ax.set_title('Seismic Section - Marine Survey', fontsize=13, fontweight='bold', pad=15)

        # Set background to light cream color (like reference)
        ax.set_facecolor('#f5f5f0')
        fig.patch.set_facecolor('#f5f5f0')

        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)

        plt.tight_layout()

        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#f5f5f0')
        buf.seek(0)
        preview_img = Image.open(buf).copy()
        plt.close()

        return preview_img

    def _detect_seafloor(self, data: np.ndarray) -> np.ndarray:
        """Detect seafloor (first strong reflection) for each trace"""
        num_samples, num_traces = data.shape
        seafloor = np.zeros(num_traces)

        for i in range(num_traces):
            trace = data[:, i]
            # Find first sample with high amplitude (threshold method)
            threshold = 0.3 * np.max(np.abs(trace))
            strong_samples = np.where(np.abs(trace) > threshold)[0]

            if len(strong_samples) > 0:
                # Use first strong reflection as seafloor
                seafloor[i] = strong_samples[0]
            else:
                # Default to 10% depth if no strong reflection
                seafloor[i] = int(num_samples * 0.1)

        # Smooth the seafloor line
        from scipy.ndimage import gaussian_filter1d
        seafloor = gaussian_filter1d(seafloor, sigma=5)

        return seafloor

    def _detect_horizons(self, data: np.ndarray, num_horizons: int = 3) -> list:
        """Detect geological horizons (strong continuous reflectors)"""
        num_samples, num_traces = data.shape
        horizons = []

        # Look for horizons at different depth ranges
        depth_ranges = [
            (int(num_samples * 0.3), int(num_samples * 0.4)),
            (int(num_samples * 0.5), int(num_samples * 0.6)),
            (int(num_samples * 0.7), int(num_samples * 0.8)),
        ]

        for start, end in depth_ranges[:num_horizons]:
            horizon = np.zeros(num_traces)

            for i in range(num_traces):
                trace = data[start:end, i]
                # Find peak amplitude in this range
                peak_idx = np.argmax(np.abs(trace))
                horizon[i] = start + peak_idx

            # Smooth the horizon
            from scipy.ndimage import gaussian_filter1d
            horizon = gaussian_filter1d(horizon, sigma=10)
            horizons.append(horizon)

        return horizons

    def apply_forward_diffusion(self, timestep: int):
        """Apply forward diffusion to RGB image"""
        if self.current_rgb is None:
            return None, None, None, "Please load SEG-Y file first"

        # Convert to tensor
        img_tensor = self.processor.preprocess_image(self.current_rgb)

        # Add noise
        noisy_tensor, noise_tensor = self.processor.add_noise(img_tensor, timestep)

        # Convert back to images
        noisy_image = self.processor.postprocess_image(noisy_tensor)
        noise_image = self.processor.postprocess_image(noise_tensor)

        # Calculate noise statistics
        alpha = self.processor.scheduler.alphas_cumprod[timestep].item()
        signal_ratio = alpha * 100
        noise_ratio = (1 - alpha) * 100

        info = f"Timestep: {timestep}/1000\n"
        info += f"Signal: {signal_ratio:.1f}%, Noise: {noise_ratio:.1f}%"

        return self.current_rgb, noisy_image, noise_image, info

    def apply_reverse_diffusion(self, timestep: int, denoise_strength: float = 0.3):
        """Apply reverse diffusion (denoising)"""
        if self.current_rgb is None:
            return None, None, None, "Please load SEG-Y file first"

        # First add noise (forward diffusion)
        img_tensor = self.processor.preprocess_image(self.current_rgb)
        noisy_tensor, _ = self.processor.add_noise(img_tensor, timestep)
        noisy_image = self.processor.postprocess_image(noisy_tensor)

        # Then denoise (reverse diffusion)
        denoised_image = self.processor.denoise_simple(noisy_image, denoise_strength)

        info = f"Noise level: {timestep}/1000\n"
        info += f"Denoise strength: {denoise_strength:.2f}"

        return self.current_rgb, noisy_image, denoised_image, info

    def create_diffusion_sequence(self, num_steps: int = 10):
        """Create a sequence showing forward diffusion progression"""
        if self.current_rgb is None:
            return None, "Please load SEG-Y file first"

        timesteps = np.linspace(0, 999, num_steps, dtype=int)

        # Create figure
        cols = 5
        rows = (num_steps + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        axes = axes.flatten()

        img_tensor = self.processor.preprocess_image(self.current_rgb)

        for idx, timestep in enumerate(timesteps):
            noisy_tensor, _ = self.processor.add_noise(img_tensor, timestep)
            noisy_image = self.processor.postprocess_image(noisy_tensor)

            axes[idx].imshow(noisy_image)
            axes[idx].set_title(f't={timestep}')
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(num_steps, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        sequence_image = Image.open(buf).copy()
        plt.close()

        return sequence_image, f"Generated {num_steps} diffusion steps"

    def create_denoise_sequence(self, initial_noise: int = 500, num_steps: int = 10):
        """Create a sequence showing reverse diffusion (denoising)"""
        if self.current_rgb is None:
            return None, "Please load SEG-Y file first"

        # Add initial noise
        img_tensor = self.processor.preprocess_image(self.current_rgb)
        noisy_tensor, _ = self.processor.add_noise(img_tensor, initial_noise)
        noisy_image = self.processor.postprocess_image(noisy_tensor)

        # Generate denoising steps
        denoise_steps = self.processor.denoise_iterative(noisy_image, initial_noise, num_steps)

        # Create figure
        cols = 5
        rows = (len(denoise_steps) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        axes = axes.flatten()

        for idx, img in enumerate(denoise_steps):
            axes[idx].imshow(img)
            progress = (idx / len(denoise_steps)) * 100
            axes[idx].set_title(f'{progress:.0f}% denoised')
            axes[idx].axis('off')

        # Hide unused subplots
        for idx in range(len(denoise_steps), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        sequence_image = Image.open(buf).copy()
        plt.close()

        return sequence_image, f"Generated {len(denoise_steps)} denoising steps"


def create_interface():
    """Create Gradio interface"""
    visualizer = SegyDiffusionVisualizerRGB()

    with gr.Blocks(title="SEG-Y RGB Diffusion Visualizer") as demo:
        gr.Markdown("""
        # 🌊 SEG-Y RGB Diffusion Visualizer

        Convert seismic sections to RGB images and visualize the complete diffusion process:
        - **Load SEG-Y**: Convert seismic data to colormap-based RGB images
        - **Forward Diffusion**: Watch clean seismic images become noise
        - **Reverse Diffusion**: See noisy images being denoised step-by-step
        """)

        with gr.Row():
            segy_file = gr.File(label="Upload SEG-Y File", file_types=[".sgy", ".segy"])
            colormap_choice = gr.Dropdown(
                choices=['seismic', 'gray', 'viridis', 'RdBu'],
                value='seismic',
                label="Colormap"
            )
            load_btn = gr.Button("Load & Convert to RGB", variant="primary")

        with gr.Row():
            original_display = gr.Image(label="Original RGB Image", type="pil")
            rgb_preview = gr.Image(label="Current State", type="pil")

        load_info = gr.Textbox(label="File Info", lines=2)

        load_btn.click(
            fn=lambda f, c: visualizer.load_segy(f, c) if f else (None, None, "No file"),
            inputs=[segy_file, colormap_choice],
            outputs=[original_display, rgb_preview, load_info]
        )

        with gr.Tab("Forward Diffusion (Add Noise)"):
            gr.Markdown("""
            ### Add Noise Progressively
            Move the slider to see how Gaussian noise corrupts the seismic RGB image.
            """)

            with gr.Row():
                forward_timestep = gr.Slider(
                    0, 999, value=0, step=1,
                    label="Noise Timestep (0=clean, 999=pure noise)"
                )

            with gr.Row():
                forward_original = gr.Image(label="Original", type="pil")
                forward_noisy = gr.Image(label="Noisy Image", type="pil")
                forward_noise = gr.Image(label="Noise Component", type="pil")

            forward_info = gr.Textbox(label="Info", lines=2)
            forward_apply_btn = gr.Button("Apply Forward Diffusion", variant="primary")

            forward_apply_btn.click(
                fn=visualizer.apply_forward_diffusion,
                inputs=[forward_timestep],
                outputs=[forward_original, forward_noisy, forward_noise, forward_info]
            )

            forward_timestep.change(
                fn=visualizer.apply_forward_diffusion,
                inputs=[forward_timestep],
                outputs=[forward_original, forward_noisy, forward_noise, forward_info]
            )

            gr.Markdown("### Diffusion Sequence")
            sequence_steps = gr.Slider(4, 20, value=10, step=1, label="Number of steps")
            sequence_btn = gr.Button("Generate Forward Sequence")
            sequence_output = gr.Image(label="Diffusion Progression", type="pil")
            sequence_info = gr.Textbox(label="Info", lines=1)

            sequence_btn.click(
                fn=visualizer.create_diffusion_sequence,
                inputs=[sequence_steps],
                outputs=[sequence_output, sequence_info]
            )

        with gr.Tab("Reverse Diffusion (Denoise)"):
            gr.Markdown("""
            ### Remove Noise Progressively
            First add noise, then watch it being removed through denoising.
            """)

            with gr.Row():
                reverse_timestep = gr.Slider(
                    0, 999, value=300, step=1,
                    label="Initial Noise Level"
                )
                reverse_strength = gr.Slider(
                    0.0, 1.0, value=0.3, step=0.05,
                    label="Denoising Strength"
                )

            with gr.Row():
                reverse_original = gr.Image(label="Original", type="pil")
                reverse_noisy = gr.Image(label="Noisy Image", type="pil")
                reverse_denoised = gr.Image(label="Denoised Result", type="pil")

            reverse_info = gr.Textbox(label="Info", lines=2)
            reverse_apply_btn = gr.Button("Apply Reverse Diffusion", variant="primary")

            reverse_apply_btn.click(
                fn=visualizer.apply_reverse_diffusion,
                inputs=[reverse_timestep, reverse_strength],
                outputs=[reverse_original, reverse_noisy, reverse_denoised, reverse_info]
            )

            reverse_strength.change(
                fn=visualizer.apply_reverse_diffusion,
                inputs=[reverse_timestep, reverse_strength],
                outputs=[reverse_original, reverse_noisy, reverse_denoised, reverse_info]
            )

            gr.Markdown("### Denoising Sequence")
            denoise_initial = gr.Slider(0, 999, value=500, step=1, label="Initial noise level")
            denoise_steps = gr.Slider(5, 20, value=10, step=1, label="Number of steps")
            denoise_seq_btn = gr.Button("Generate Denoising Sequence")
            denoise_seq_output = gr.Image(label="Denoising Progression", type="pil")
            denoise_seq_info = gr.Textbox(label="Info", lines=1)

            denoise_seq_btn.click(
                fn=visualizer.create_denoise_sequence,
                inputs=[denoise_initial, denoise_steps],
                outputs=[denoise_seq_output, denoise_seq_info]
            )

        gr.Markdown("""
        ### How It Works

        **RGB Conversion**:
        - Seismic data → Normalized → Colormap applied → RGB image
        - Preserves spatial structure while enabling image-based processing

        **Forward Diffusion**:
        - `noisy_image = sqrt(α_t) × clean_image + sqrt(1 - α_t) × noise`
        - Progressive noise addition over 1000 timesteps

        **Reverse Diffusion**:
        - Iteratively removes noise to recover clean signal
        - Currently uses Gaussian filtering (can be replaced with trained model)

        **Advantages**:
        - Full section visualization (not just single traces)
        - RGB format compatible with image diffusion models
        - Can leverage pre-trained vision models
        - Preserves 2D seismic structure
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7862)
