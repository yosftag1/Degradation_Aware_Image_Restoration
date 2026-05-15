# Degradation-Aware Image Restoration

This is an image restoration project built around a small degradation-aware model. The goal of this project is to implement and refine state-of-the-art methods for image transformation and restoration. This is mainly done by synthesizing corruptions, estimating the degradation, and using that estimation to guide the restoration.

## Project Overview

### Architecture

**NAFNet-inspired Lightweight U-Net**
- Uses the paper's simple-block philosophy
- 4 lightweight NAF-style blocks instead of the full architecture
- Channel attention and skip connections

**Degradation Estimator Branch**
- Predicts degradation type and severity
- Produces a small embedding that conditions the decoder
- Keeps the model degradation-aware instead of blind

### Degradation Pipeline

Custom synthesis on top of DIV2K:
- **Gaussian Noise**: Additive noise with controllable variance
- **Motion Blur**: Directional blur with random angle and kernel size
- **JPEG Compression**: Quality-based compression artifacts
- **Color Degradations**: White balance shifts, saturation loss, and hue shifts
- **Exposure Issues**: Underexposure, overexposure, and contrast changes
- All degradations applied with variable severity

### Loss Function

Multi-component loss stack:
1. **L1 Loss** (weight: 1.0): Pixel-level reconstruction
2. **Perceptual Loss** (weight: 0.1): VGG-based feature matching at relu3_4
3. **Frequency Loss** (weight: 0.05): FFT magnitude spectrum matching
4. **Auxiliary losses**: Degradation type and severity classification (encourage interpretability)

### Design Philosophy

- Lightweight and easy to train
- Clear enough to explain in an interview
- Focused on the restoration idea rather than extra complexity

## Project Structure

```
degradation_aware_restoration/
├── config.py                 # Configuration management
├── requirements.txt          # Dependencies
├── train.py                  # Main training script
├── inference.py              # Inference and evaluation
├── download_dataset.py       # DIV2K dataset download
├── models/
│   └── architecture.py       # Model implementation
├── losses/
│   └── loss_functions.py     # Loss functions
├── data/
│   └── dataset.py            # Data pipeline and degradation synthesis
└── checkpoints/              # Saved models
```

## Getting Started

### 1. Installation

```bash
cd degradation_aware_restoration
pip install -r requirements.txt
```

### 2. Download Dataset

Download DIV2K (2K high-resolution images):

```bash
python download_dataset.py --dataset-dir ./data/DIV2K
```

This downloads the DIV2K train and validation images.

### 3. Quick Start (Debug Mode)

Test on small dataset with fast iteration:

```bash
python train.py --debug
```

Runs for 3 epochs on 10 images to verify everything works.

### 4. Full Training

```bash
python train.py
```

Trains for 50 epochs with checkpoint saving.

### 5. Resume Training

```bash
python train.py --resume checkpoints/model_best.pt
```

### 6. Inference

Restore a single image:

```bash
python inference.py \
    --model checkpoints/model_best.pt \
    --image degraded_image.png \
    --output-dir ./restored
```

Restore entire directory:

```bash
python inference.py \
    --model checkpoints/model_best.pt \
    --batch-dir ./images \
    --output-dir ./restored
```

Evaluate restoration quality (PSNR/SSIM):

```bash
python inference.py \
    --model checkpoints/model_best.pt \
    --eval ./clean_images \
    --eval-output results.json
```

## Model Architecture Details

### Encoder

Progressive downsampling with residual NAFNet blocks:
```
Input (256×256×3)
  ↓ Conv
  ↓ NAFNetBlock × 2
  ↓ Downsample (2× stride)
  ↓ NAFNetBlock × 2
  ↓ Downsample (2× stride)
  ↓ NAFNetBlock × 2
  ↓ Downsample (2× stride)
  ↓ NAFNetBlock × 2  ← Bottleneck
```

### Degradation Estimator

Lightweight auxiliary network:
```
Input (256×256×3)
  ↓ Conv (stride=2) → 16 channels
  ↓ Conv (stride=2) → 32 channels
  ↓ Conv (stride=2) → 64 channels
  ↓ Global Average Pool
  ├→ FC → Degradation Type (3-way classification)
  ├→ FC → Severity Level (5-way classification)
  └→ FC → Embedding (16D conditioning vector)
```

### Decoder

Progressive upsampling with skip connections and degradation conditioning:
```
Bottleneck (from encoder)
  ↓ Add degradation embedding
  ↓ NAFNetBlock × 2
  ↓ Upsample (2×) + Skip + Concat
  ↓ NAFNetBlock × 2
  ↓ Upsample (2×) + Skip + Concat
  ↓ NAFNetBlock × 2
  ↓ Upsample (2×) + Skip + Concat
  ↓ NAFNetBlock × 2
  ↓ Conv → Output (256×256×3)
```

## Configuration

Edit `config.py` to adjust:

```python
# Architecture
BASE_CHANNELS = 32          # Increase for larger model
NUM_BLOCKS = 4              # Increase for deeper model
ATTENTION_REDUCTION = 16    # Attention bottleneck

# Training
LEARNING_RATE = 1e-3        # Initial learning rate
BATCH_SIZE = 4              # Batch size (adjust for GPU memory)
EPOCHS = 50                 # Number of epochs

# Loss weights
LOSS_L1_WEIGHT = 1.0        # L1 importance
LOSS_PERCEPTUAL_WEIGHT = 0.1    # Perceptual importance
LOSS_FREQUENCY_WEIGHT = 0.05    # Frequency importance
```

## Key Implementation Insights

### 1. Degradation Awareness
- Helps the decoder adapt to the input corruption.
- Keeps the model easier to inspect and explain.

### 2. Loss Weighting
- L1 does most of the reconstruction work.
- Perceptual and frequency losses help with detail.
- Auxiliary losses teach the degradation branch.

### 3. Channel Attention
- Reweights features so the network can focus on useful channels.
- Adds a small amount of capacity without making the model heavy.

### 4. NAFNet-style Design
Inspired by [NAFNet](https://arxiv.org/abs/2204.04676):
- Uses simple blocks and fewer layers.
- Keeps the model fast enough to train and explain.

## Training Monitoring

Monitor training in several ways:

1. **Terminal output**: Real-time epoch/loss information
2. **Checkpoints**: Saved every 5 epochs + best model
3. **History**: Training/validation metrics saved to `logs/history_*.json`

Example history structure:
```json
{
  "train": [
    {
      "epoch": 1,
      "loss": 0.125,
      "components": {
        "l1": 0.10,
        "perceptual": 0.015,
        "frequency": 0.01,
        ...
      }
    }
  ],
  "val": [...]
}
```

## References

- NAFNet: [https://arxiv.org/abs/2204.04676](https://arxiv.org/abs/2204.04676)
- NAFNet: [https://github.com/chxy95/NAFNet](https://github.com/chxy95/NAFNet)
- DF2K + OST Dataset: [https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost](https://www.kaggle.com/datasets/thaihoa1476050/df2k-ost)
- VGG Perceptual Loss: [https://arxiv.org/abs/1603.08155](https://arxiv.org/abs/1603.08155)
