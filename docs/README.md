# Mario RL Documentation

Welcome to the Mario RL documentation! This directory contains comprehensive documentation for the Mario RL project.

## 📚 Documentation Structure

### API Reference
- **`api/`** - Complete API documentation for all modules
- **`examples/`** - Code examples and tutorials
- **`tutorials/`** - Step-by-step guides

## 🚀 Quick Start

### Installation
```bash
# Install the package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Basic Usage
```python
from mario_rl.environments import MarioEnvironment
from mario_rl.agents import MarioAgent
from mario_rl.configs import TrainingConfig

# Create environment
env = MarioEnvironment(level="1-1", movement_type="simple")
vec_env = env.create_vectorized_env(n_envs=1)

# Create agent
config = TrainingConfig()
agent = MarioAgent(vec_env, config)

# Train
model = agent.create_model()
agent.train(100000)
```

### Command Line Interface
```bash
# Train a new agent
mario-rl train --level 1-1 --timesteps 100000

# Evaluate a trained model
mario-rl evaluate --model checkpoints/ppo_mario_final.zip

# Visualize results
mario-rl visualize --log-file mario_training.log
```

## 📖 Available Documentation

### Core Modules
- **Environments** - Mario game environment wrappers
- **Agents** - Reinforcement learning agent implementations
- **Models** - PPO and other RL model implementations
- **Configs** - Configuration management
- **Utils** - Logging, plotting, and utility functions

### Examples
- Basic training scripts
- Environment customization
- Model evaluation
- Result visualization

### Tutorials
- Getting started guide
- Advanced training techniques
- Custom environment creation
- Performance optimization

## 🔧 Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mario_rl --cov-report=html
```

### Code Quality
```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/
```

## 📝 Contributing

Please see the main [CONTRIBUTING.md](../CONTRIBUTING.md) file for guidelines on contributing to the documentation.

## 🐛 Issues

If you find issues with the documentation, please:
1. Check if the issue has already been reported
2. Create a new issue with a clear description
3. Include steps to reproduce the problem

## 📞 Support

For questions about the documentation:
- Check the examples and tutorials first
- Search existing issues
- Create a new issue for specific questions
