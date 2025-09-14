# Docker Setup for Mario RL

This directory contains Docker configuration and scripts for running the Mario RL project in containers, providing a consistent development environment across different machines.

## Quick Start

1. **Setup the environment:**
   ```bash
   ./scripts/setup.sh
   ```

2. **Run basic training:**
   ```bash
   ./scripts/run.sh train
   ```

3. **Start development environment:**
   ```bash
   ./scripts/run.sh --service mario-rl-dev
   ```

## Architecture

### Docker Services

- **mario-rl**: Production runtime environment for training and evaluation
- **mario-rl-dev**: Development environment with additional tools (Jupyter, debugging tools)

### Docker Images

- **Base stage**: Python environment with system dependencies
- **Runtime stage**: Minimal image with ML dependencies for production
- **Development stage**: Extended image with development tools

## Files Overview

```
├── Dockerfile              # Multi-stage Docker build configuration
├── docker-compose.yml      # Container orchestration setup
├── .dockerignore           # Build context optimization
└── scripts/
    ├── setup.sh            # Complete environment setup
    ├── build.sh            # Docker image building
    └── run.sh              # Container execution wrapper
```

## Usage Guide

### Initial Setup

```bash
# Run the complete setup (recommended for first-time users)
./scripts/setup.sh

# Or manual setup:
./scripts/build.sh --target runtime      # Build runtime image
./scripts/build.sh --target development  # Build development image
```

### Training Commands

```bash
# Basic training session
./scripts/run.sh train

# Extended 100K episode training
./scripts/run.sh train-100k

# Background training (detached)
./scripts/run.sh -d train-100k
```

### Development Workflow

```bash
# Interactive development shell
./scripts/run.sh --service mario-rl-dev bash

# Start Jupyter Lab (available at http://localhost:8888)
./scripts/run.sh --service mario-rl-dev jupyter

# Run tests
./scripts/run.sh test

# Start TensorBoard (available at http://localhost:6006)
./scripts/run.sh tensorboard
```

### Container Management

```bash
# Start containers with docker-compose
docker-compose up mario-rl          # Runtime container
docker-compose up mario-rl-dev      # Development container
docker-compose up -d mario-rl-dev   # Development in background

# Stop containers
docker-compose down
docker-compose stop mario-rl

# View logs
docker-compose logs -f mario-rl
```

## Data Persistence

All important data is persisted through Docker volumes:

- **models/**: Trained model checkpoints and final models
- **videos/**: Training and evaluation video recordings
- **logs/**: Training logs and TensorBoard data
- **data/**: Cached data and temporary files

Data persists between container restarts and rebuilds.

## Script Options

### build.sh Options

```bash
./scripts/build.sh [OPTIONS]

Options:
  -t, --target TARGET     Build target (runtime|development) [default: runtime]
  -p, --python VERSION    Python version (3.9|3.10|3.11) [default: 3.10]
  -f, --force            Force rebuild (remove existing images)
  --no-cache             Build without using cache
  -h, --help             Show help message
```

### run.sh Options

```bash
./scripts/run.sh [OPTIONS] [COMMAND]

Options:
  -s, --service SERVICE   Service to run (mario-rl|mario-rl-dev) [default: mario-rl]
  -d, --detach           Run in detached mode (background)
  -b, --build            Build images before running
  --no-interactive       Run without interactive mode
  -h, --help             Show help message

Commands:
  train                  Run basic training script
  train-100k             Run 100K episode training
  evaluate               Run model evaluation
  visualize              Run visualization
  bash                   Open interactive shell (default)
  jupyter                Start Jupyter Lab (dev service only)
  tensorboard           Start TensorBoard
  test                   Run test suite
```

## Environment Variables

Key environment variables for customization:

```bash
# Display forwarding (automatically set on macOS)
export DISPLAY=host.docker.internal:0

# GPU configuration
export CUDA_VISIBLE_DEVICES=0

# PyTorch MPS fallback (for Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## GPU Support

### NVIDIA GPU (Linux)

Uncomment the GPU-related sections in `docker-compose.yml`:

```yaml
# Uncomment for NVIDIA GPU support
runtime: nvidia
devices:
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-uvm:/dev/nvidia-uvm
```

### Apple Silicon (MPS)

MPS support is automatically configured for Apple Silicon Macs. The containers will use Apple's Metal Performance Shaders when available.

## Troubleshooting

### Common Issues

1. **Docker not running:**
   ```bash
   # Start Docker Desktop or Docker daemon
   open -a Docker  # macOS
   ```

2. **Permission errors:**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   ```

3. **Build failures:**
   ```bash
   # Force rebuild without cache
   ./scripts/build.sh --force --no-cache
   ```

4. **GPU not detected:**
   ```bash
   # Check GPU availability in container
   docker-compose run mario-rl python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
   ```

### Debug Mode

Run containers with additional debugging:

```bash
# Enable verbose logging
docker-compose run mario-rl bash -c "export PYTHONPATH=/workspace && python -v training_scripts/train_mario.py"

# Check environment setup
docker-compose run mario-rl python -c "
import sys, torch, stable_baselines3, gym_super_mario_bros
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'SB3: {stable_baselines3.__version__}')
print(f'Mario: {gym_super_mario_bros.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'MPS: {torch.backends.mps.is_available()}')
"
```

### Performance Optimization

1. **Resource limits**: Adjust memory limits in `docker-compose.yml`
2. **Build optimization**: Use `.dockerignore` to exclude unnecessary files
3. **Volume optimization**: Use specific volume mounts instead of full directory mounts

## Integration with Host System

### X11 Forwarding (macOS)

For GUI applications and rendering:

```bash
# Install XQuartz if needed
brew install --cask xquartz

# Allow localhost connections
xhost +localhost
```

### File Permissions

The container runs as root by default. To maintain file ownership:

```bash
# Change ownership after container operations
sudo chown -R $(whoami):$(whoami) models/ videos/ logs/
```

## Advanced Usage

### Custom Training Scripts

Add your own training scripts to `training_scripts/` and run them:

```bash
./scripts/run.sh --service mario-rl python training_scripts/my_custom_script.py
```

### Multi-Stage Development

```bash
# Develop in dev container
./scripts/run.sh --service mario-rl-dev bash

# Test in runtime container
./scripts/run.sh --service mario-rl bash

# Production deployment
docker-compose up -d mario-rl
```

### Continuous Integration

Use the containers in CI/CD pipelines:

```bash
# Build and test
./scripts/build.sh --target runtime
docker-compose run --rm mario-rl python -m pytest tests/

# Production deployment
docker tag mario-rl:latest myregistry/mario-rl:v1.0
docker push myregistry/mario-rl:v1.0
```