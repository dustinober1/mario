# Training Scripts

This directory contains various training scripts for the Mario RL project. Each script represents different approaches and optimizations for training the Mario agent.

## Scripts Overview

- `train_mario.py` - Original training script
- `train_mario_production.py` - Clean production version with optimized output
- `train_mario_live_visual.py` - Training with live visual rendering
- `train_mario_gpu_simple.py` - GPU-optimized simple training
- `train_mario_optimized.py` - Performance-optimized training
- `train_mario_fixed.py` - Fixed version addressing buffer issues
- `train_mario_visual.py` - Training with visual rendering support
- `train_mario_manual_render.py` - Manual rendering control for demonstrations

## Usage

Run any training script from the project root directory:

```bash
# From the mario/ root directory
cd ..
python training_scripts/train_mario_production.py
```

## GPU Acceleration

Scripts with GPU optimization (using Apple Silicon MPS):
- `train_mario_gpu_simple.py`
- `train_mario_production.py`
- `train_mario_live_visual.py`

These provide 2x performance improvement on M1/M2 Macs.