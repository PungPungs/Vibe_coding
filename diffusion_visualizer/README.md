# Diffusion Model Visualizer

A web-based interactive tool to visualize the forward diffusion process used in diffusion models like DDPM, Stable Diffusion, and DALL-E.

## Features

- **Single Step Visualization**: See how an image looks at any specific timestep (0-999)
- **Noise Component View**: Visualize the actual noise being added at each step
- **Diffusion Sequence**: Generate a multi-panel view showing the complete diffusion process
- **Interactive Slider**: Real-time updates as you move through timesteps
- **Web-based Interface**: Easy-to-use browser interface powered by Gradio

## What is Forward Diffusion?

Forward diffusion is the process of gradually adding Gaussian noise to an image over many timesteps:
- **Timestep 0**: Original clean image
- **Timestep 500**: Mixture of signal and noise
- **Timestep 999**: Almost pure noise

Diffusion models learn to reverse this process, enabling them to generate images from random noise.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: PyTorch installation may vary based on your system. Visit [pytorch.org](https://pytorch.org) for specific instructions.

## Usage

1. Run the visualizer:
```bash
python src/diffusion_viz.py
```

2. Open your browser to `http://127.0.0.1:7860`

3. Upload an image and explore:
   - Use the slider to see different timesteps
   - Toggle "Show noise component" to see the added noise
   - Switch to "Diffusion Sequence" tab to generate a multi-step visualization

## How It Works

The visualizer uses the Diffusers library's `DDPMScheduler` to add noise according to the DDPM (Denoising Diffusion Probabilistic Models) schedule:

1. Images are normalized to [-1, 1] range
2. Gaussian noise is added according to the variance schedule
3. The amount of noise increases over timesteps following: `noisy_image = sqrt(alpha) * image + sqrt(1-alpha) * noise`

## Examples

### Single Step Mode
Upload any image and drag the timestep slider to see how noise gradually obscures the original image.

### Sequence Mode
Generate a grid showing multiple timesteps at once to understand the full diffusion trajectory.

## Technical Details

- **Scheduler**: DDPM with linear beta schedule
- **Timesteps**: 1000 steps (standard for most diffusion models)
- **Image Processing**: Automatic resizing to max 512px for performance
- **Framework**: PyTorch + Diffusers

## Future Enhancements

Possible additions:
- Reverse diffusion (denoising) visualization
- Different noise schedules (cosine, sigmoid)
- Conditional diffusion with text prompts
- Video support
- Export animations as GIF/MP4

## License

MIT License

## References

- [Denoising Diffusion Probabilistic Models (DDPM) Paper](https://arxiv.org/abs/2006.11239)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
