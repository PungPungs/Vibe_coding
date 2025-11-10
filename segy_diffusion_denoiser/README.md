# SEG-Y Diffusion Denoiser

A specialized visualization tool for applying diffusion models to seismic data. Available in two versions:

1. **Trace-based** (`segy_diffusion.py`) - Individual 1D trace analysis
2. **RGB-based** (`segy_diffusion_rgb.py`) - Full section visualization with RGB conversion

## Features

### Common Features
- **SEG-Y File Support**: Load standard SEG-Y seismic files
- **Forward Diffusion**: Visualize how noise corrupts clean seismic data
- **Reverse Diffusion**: Denoise corrupted data to recover the signal
- **Real-time Visualization**: See results instantly with interactive sliders
- **Diffusion Sequences**: Multi-step progression views

### Trace-based Version Features
- **Individual Trace Analysis**: Select and analyze specific traces
- **1D Waveform Plots**: Traditional seismic wiggle trace visualization
- **Noise Component View**: See the actual noise being added

### RGB Version Features (NEW!)
- **Full Section Visualization**: View entire seismic sections as RGB images
- **Colormap Support**: Multiple colormap options (seismic, gray, viridis, RdBu)
- **2D Processing**: Apply diffusion to complete seismic images
- **Side-by-side Comparison**: Original, noisy, and denoised images
- **Denoising Sequences**: Watch step-by-step noise removal

## What Makes This Different?

Unlike general image diffusion models, this tool is specifically designed for seismic data:
- Works with 1D seismic traces instead of 2D images
- Handles SEG-Y file format (IBM float conversion, headers, etc.)
- Preserves seismic waveform characteristics
- Uses trace-appropriate normalization

## Use Cases

1. **Understanding Noise in Seismic Data**: See how different noise levels affect trace quality
2. **Denoising Research**: Test denoising algorithms on real seismic data
3. **Data Augmentation**: Generate noisy versions of clean data for training
4. **Quality Control**: Evaluate noise levels in acquired seismic data
5. **Algorithm Development**: Prototype and test diffusion-based processing

## Installation

1. Install dependencies:
```bash
cd segy_diffusion_denoiser
pip install -r requirements.txt
```

Note: PyTorch installation may vary. Visit [pytorch.org](https://pytorch.org) for system-specific instructions.

## Usage

### Trace-based Version (Individual Traces)

1. Run the application:
```bash
python src/segy_diffusion.py
```

2. Open browser to `http://127.0.0.1:7861`

3. Load a SEG-Y file and analyze individual traces

### RGB Version (Full Sections) - Recommended!

1. Run the RGB version:
```bash
python src/segy_diffusion_rgb.py
```

2. Open browser to `http://127.0.0.1:7862`

3. Load a SEG-Y file:
   - Click "Upload SEG-Y File"
   - Select colormap (seismic, gray, viridis, RdBu)
   - Click "Load & Convert to RGB"

4. **Forward Diffusion Tab**:
   - Move the timestep slider to add noise (0 = clean, 999 = pure noise)
   - See original, noisy image, and noise component side-by-side
   - Generate diffusion sequences showing multiple steps

5. **Reverse Diffusion Tab**:
   - Set initial noise level to corrupt the image
   - Adjust denoising strength
   - Compare original, noisy, and denoised results
   - Generate denoising sequences to see progressive cleanup

## Quick Start Example (RGB Version)

```bash
cd segy_diffusion_denoiser
pip install -r requirements.txt
python src/segy_diffusion_rgb.py
```

Then upload your SEG-Y file and explore!

## Detailed Workflow (Trace-based Version)

1. Load a SEG-Y file:
   - Select your .sgy or .segy file
   - Click "Load File" to process

2. **Add Noise Tab** (Forward Diffusion):
   - Select a trace index
   - Use the timestep slider (0 = clean, 999 = pure noise)
   - See original trace, noisy trace, and noise component

5. **Remove Noise Tab** (Reverse Diffusion):
   - Select a trace index
   - Set noise amount to add (simulates corrupted data)
   - Adjust denoising strength
   - Compare original, noisy, and denoised traces

## How It Works

### Forward Diffusion (Adding Noise)

Uses the DDPM (Denoising Diffusion Probabilistic Models) scheduler:

```
noisy_trace = sqrt(α_t) × clean_trace + sqrt(1 - α_t) × noise
```

Where:
- `α_t` is the noise schedule parameter at timestep t
- As t increases (0→999), more noise is added
- Noise is sampled from a Gaussian distribution

### Reverse Diffusion (Denoising)

Currently implements Gaussian filtering for denoising:
- Adaptive smoothing based on noise level
- Preserves low-frequency signal components
- Can be extended with trained neural networks

**Future Enhancement**: Train a U-Net model on seismic data to learn optimal denoising.

## Technical Details

### SEG-Y Reader
- Handles IBM float to IEEE float conversion
- Reads trace headers and binary headers
- Supports standard SEG-Y Rev 1 format
- Automatic per-trace normalization

### Diffusion Model
- **Scheduler**: DDPM with linear beta schedule
- **Timesteps**: 1000 steps
- **Noise Type**: Gaussian (standard normal distribution)
- **Denoising**: Gaussian filtering (placeholder for neural network)

### Architecture
```
SegyDiffusionVisualizer
├── SegyReader: Load and parse SEG-Y files
├── SegyDiffusionProcessor: Apply diffusion operations
│   ├── add_noise_to_trace()
│   ├── denoise_trace_simple()
│   └── denoise_trace_diffusion() (future)
└── Visualization: Matplotlib trace comparisons
```

## Example Workflow

1. Load a marine seismic line (e.g., 21_270m_20250921_065551_RAW_LF.sgy)
2. Select trace 50
3. Add noise at timestep 300 to see moderate corruption
4. Switch to denoise tab
5. Apply denoising with strength 0.2
6. Compare recovered signal to original

## Advanced: Training a Denoising Model

The code includes a placeholder `SimpleDenoisingModel` class with a 1D U-Net architecture. To train:

1. Collect clean seismic traces
2. Generate noisy versions at various timesteps
3. Train the model to predict noise:
   ```python
   loss = MSE(predicted_noise, actual_noise)
   ```
4. Use trained model in `denoise_trace_diffusion()`

## Comparison of Versions

| Aspect | Trace-based | RGB-based |
|--------|-------------|-----------|
| Data Type | 1D traces | 2D RGB images |
| Visualization | Wiggle traces | Full seismic sections |
| Processing | Single trace | Entire section |
| Use Case | Detailed analysis | Overview & context |
| Architecture | 1D processing | 2D image processing |
| Output | Line plots | Color images |

## Comparison with General Image Diffusion

| Aspect | Image Diffusion | SEG-Y RGB Diffusion |
|--------|----------------|---------------------|
| Data Type | Natural images | Seismic sections as RGB |
| Input | Photos, art | SEG-Y → Colormap → RGB |
| Normalization | [0, 255] → [-1, 1] | Seismic amplitude → Colormap → [-1, 1] |
| Domain | Computer vision | Geophysics |
| Applications | Generation, editing | Denoising, enhancement |

## Limitations

- Current denoising uses simple Gaussian filtering (not learned)
- No trained model included (would require large seismic dataset)
- Single trace processing (not full section optimization)
- Assumes stationary noise characteristics

## Future Enhancements

- [ ] Train neural network denoiser on real seismic data
- [ ] Support for 2D section denoising
- [ ] Multiple noise types (coherent, random, etc.)
- [ ] Conditional generation based on geology
- [ ] Batch processing multiple traces
- [ ] Export denoised SEG-Y files
- [ ] Integration with your existing SEG-Y viewers

## References

- [DDPM Paper](https://arxiv.org/abs/2006.11239) - Denoising Diffusion Probabilistic Models
- [SEG-Y Format](https://seg.org/Portals/0/SEG/News%20and%20Resources/Technical%20Standards/seg_y_rev1.pdf) - Technical standard
- [Diffusers Library](https://github.com/huggingface/diffusers) - HuggingFace implementation

## Integration with Your Project

This tool complements your existing SEG-Y viewers:
- `segy_2d_viewer`: View full sections
- `segy_trace_viewer`: Analyze individual traces
- `segy_diffusion_denoiser`: **Process and denoise traces**

All use the same SEG-Y reading foundation!

## License

MIT License
