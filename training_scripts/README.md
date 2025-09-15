# Training Scripts

This directory contains essential training scripts for the Mario RL project.

## Available Scripts

- `train_mario_production.py` - **Main training script** with optimized output and Apple Silicon MPS support
- `train_mario_extended.py` - Extended training with checkpointing and resume capability
- `record_mario_video.py` - Record gameplay videos from trained models

## Usage

Run training scripts from the project root directory:

```bash
# Main production training (recommended)
python3 training_scripts/train_mario_production.py

# Extended training with checkpoints
python3 training_scripts/train_mario_extended.py

# Record video from trained model
python3 training_scripts/record_mario_video.py
```

## Apple Silicon GPU Support

All scripts automatically detect and use Apple Silicon MPS (Metal Performance Shaders) for GPU acceleration when available, providing significant performance improvements over CPU-only training.
- `train_mario_live_visual.py`

These provide 2x performance improvement on M1/M2 Macs.