"""
Diffusion Model Visualization Tool
Shows the forward diffusion process of adding noise to images
"""

import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from io import BytesIO


class DiffusionVisualizer:
    def __init__(self):
        # Initialize the DDPM scheduler with standard parameters
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    def preprocess_image(self, image):
        """Convert PIL image to tensor normalized to [-1, 1]"""
        # Resize if too large
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        # Normalize to [-1, 1]
        img_tensor = (img_tensor - 0.5) * 2.0

        return img_tensor

    def postprocess_image(self, tensor):
        """Convert tensor back to PIL image"""
        # Denormalize from [-1, 1] to [0, 1]
        tensor = (tensor / 2.0 + 0.5).clamp(0, 1)

        # Convert to numpy
        img_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def add_noise(self, image_tensor, timestep):
        """Add noise to image at specific timestep"""
        # Generate random noise
        noise = torch.randn_like(image_tensor)

        # Add noise according to the scheduler
        timestep_tensor = torch.tensor([timestep])
        noisy_image = self.scheduler.add_noise(image_tensor, noise, timestep_tensor)

        return noisy_image, noise

    def visualize_diffusion_step(self, image, timestep, show_noise=False):
        """
        Visualize a single diffusion step

        Args:
            image: PIL Image
            timestep: int (0-1000)
            show_noise: bool - whether to show the noise component
        """
        if image is None:
            return None, None, "Please upload an image first"

        # Preprocess image
        img_tensor = self.preprocess_image(image)

        # Add noise
        noisy_img_tensor, noise_tensor = self.add_noise(img_tensor, timestep)

        # Convert back to PIL
        noisy_image = self.postprocess_image(noisy_img_tensor)

        # Calculate noise level
        noise_level = self.scheduler.alphas_cumprod[timestep].item()
        signal_level = 1 - noise_level

        info = f"Timestep: {timestep}/1000\n"
        info += f"Signal retained: {signal_level*100:.1f}%\n"
        info += f"Noise added: {noise_level*100:.1f}%"

        if show_noise:
            # Visualize the noise component
            noise_image = self.postprocess_image(noise_tensor)
            return noisy_image, noise_image, info
        else:
            return noisy_image, None, info

    def create_diffusion_sequence(self, image, num_steps=10):
        """
        Create a sequence of images showing the diffusion process

        Args:
            image: PIL Image
            num_steps: Number of steps to visualize
        """
        if image is None:
            return None, "Please upload an image first"

        # Preprocess image
        img_tensor = self.preprocess_image(image)

        # Calculate timesteps to visualize
        timesteps = np.linspace(0, 999, num_steps, dtype=int)

        # Create figure with subplots
        fig, axes = plt.subplots(2, (num_steps + 1) // 2, figsize=(15, 6))
        axes = axes.flatten()

        for idx, timestep in enumerate(timesteps):
            noisy_img_tensor, _ = self.add_noise(img_tensor, timestep)
            noisy_image = self.postprocess_image(noisy_img_tensor)

            axes[idx].imshow(noisy_image)
            axes[idx].set_title(f't={timestep}')
            axes[idx].axis('off')

        plt.tight_layout()

        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        sequence_image = Image.open(buf)
        plt.close()

        return sequence_image, f"Generated sequence with {num_steps} steps"


def create_interface():
    """Create the Gradio interface"""
    visualizer = DiffusionVisualizer()

    with gr.Blocks(title="Diffusion Model Visualizer") as demo:
        gr.Markdown("""
        # 🎨 Diffusion Model Visualizer

        This tool visualizes the **forward diffusion process** - how noise is gradually added to an image.

        Upload an image and use the slider to see how it transforms at different timesteps.
        """)

        with gr.Tab("Single Step Visualization"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=300
                    )
                    timestep_slider = gr.Slider(
                        minimum=0,
                        maximum=999,
                        value=0,
                        step=1,
                        label="Timestep (0=clean, 999=pure noise)"
                    )
                    show_noise_check = gr.Checkbox(
                        label="Show noise component",
                        value=False
                    )
                    visualize_btn = gr.Button("Visualize", variant="primary")

                with gr.Column():
                    output_image = gr.Image(label="Noisy Image", height=300)
                    noise_image = gr.Image(label="Noise Component", height=300, visible=True)
                    info_text = gr.Textbox(label="Info", lines=3)

            visualize_btn.click(
                fn=visualizer.visualize_diffusion_step,
                inputs=[input_image, timestep_slider, show_noise_check],
                outputs=[output_image, noise_image, info_text]
            )

            # Auto-update when slider changes
            timestep_slider.change(
                fn=visualizer.visualize_diffusion_step,
                inputs=[input_image, timestep_slider, show_noise_check],
                outputs=[output_image, noise_image, info_text]
            )

        with gr.Tab("Diffusion Sequence"):
            with gr.Row():
                with gr.Column():
                    seq_input_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=300
                    )
                    num_steps_slider = gr.Slider(
                        minimum=4,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of steps to show"
                    )
                    sequence_btn = gr.Button("Generate Sequence", variant="primary")

                with gr.Column():
                    sequence_output = gr.Image(label="Diffusion Sequence", height=400)
                    seq_info_text = gr.Textbox(label="Info", lines=2)

            sequence_btn.click(
                fn=visualizer.create_diffusion_sequence,
                inputs=[seq_input_image, num_steps_slider],
                outputs=[sequence_output, seq_info_text]
            )

        gr.Markdown("""
        ### How it works:

        1. **Forward Diffusion**: The model gradually adds Gaussian noise to the image over 1000 timesteps
        2. **Timestep 0**: Original clean image (no noise)
        3. **Timestep 500**: Half signal, half noise
        4. **Timestep 999**: Almost pure noise

        The diffusion schedule controls how much noise is added at each step.
        This process is reversible - denoising models learn to reverse this process to generate images from noise!
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
