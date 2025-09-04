# 🍄 Mario RL - Professional Deep Reinforcement Learning

A professional-grade deep reinforcement learning implementation for training AI agents to play Super Mario Bros using PPO (Proximal Policy Optimization) and stable-baselines3.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-Latest-green.svg)](https://stable-baselines3.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/dustinober1/mario/workflows/CI/badge.svg)](https://github.com/dustinober1/mario/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

This project demonstrates advanced reinforcement learning techniques applied to the classic Super Mario Bros game. The agent learns to navigate through levels, avoid obstacles, defeat enemies, and reach the flag pole through trial and error, gradually improving its performance over thousands of episodes.

### ✨ Key Features

- **🏗️ Professional Architecture**: Modular, well-documented codebase with proper error handling
- **🚀 Advanced Training**: PPO algorithm with optimized hyperparameters and callbacks
- **📊 Comprehensive Evaluation**: Detailed performance metrics and visualization tools
- **💾 Model Management**: Automatic checkpointing, model comparison, and version control
- **📈 Tensorboard Integration**: Real-time training monitoring and logging
- **🎮 Multi-level Support**: Train on different Mario levels with configurable difficulty
- **⚡ Parallel Training**: Support for multiple environment instances
- **🛠️ CLI Interface**: Easy-to-use command-line tools for training and evaluation

## 🏗️ Project Structure

```
mario/
├── src/
│   └── mario_rl/              # Main package
│       ├── __init__.py         # Package initialization
│       ├── agents/             # RL agent implementations
│       │   ├── __init__.py
│       │   └── mario_agent.py  # PPO-based Mario agent
│       ├── environments/       # Game environment wrappers
│       │   ├── __init__.py
│       │   └── mario_env.py    # Mario environment wrapper
│       ├── models/             # Model implementations
│       │   ├── __init__.py
│       │   └── ppo_model.py    # PPO model wrapper
│       ├── configs/            # Configuration management
│       │   ├── __init__.py
│       │   └── training_config.py
│       ├── utils/              # Utility functions
│       │   ├── __init__.py
│       │   ├── logging_utils.py
│       │   └── plotting_utils.py
│       └── cli/                # Command-line interface
│           ├── __init__.py
│           ├── main.py         # Main CLI entry point
│           ├── train.py        # Training command
│           ├── evaluate.py     # Evaluation command
│           └── visualize.py    # Visualization command
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Test configuration
│   └── test_mario.py          # Basic tests
├── examples/                   # Example scripts
│   └── basic_training.py      # Basic training example
├── docs/                      # Documentation
│   ├── README.md              # Documentation overview
│   ├── api/                   # API reference
│   ├── examples/              # Code examples
│   └── tutorials/             # Step-by-step guides
├── scripts/                   # Utility scripts
├── notebooks/                 # Jupyter notebooks
├── .github/                   # GitHub configuration
│   └── workflows/             # CI/CD workflows
├── setup.py                   # Package installation
├── pyproject.toml             # Modern Python packaging
├── requirements.txt            # Python dependencies
├── Makefile                   # Development commands
├── pytest.ini                 # Test configuration
├── .pre-commit-config.yaml    # Code quality hooks
├── README.md                  # This file
├── CHANGELOG.md               # Version history
├── CONTRIBUTING.md            # Contribution guidelines
├── CODE_OF_CONDUCT.md         # Community standards
└── SECURITY.md                # Security policy
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dustinober1/mario.git
   cd mario
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv mario_env
   source mario_env/bin/activate  # On Windows: mario_env\Scripts\activate
   ```

3. **Install the package**:
   ```bash
   # Install in development mode
   pip install -e .
   
   # Install with development dependencies
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## 🚀 Usage

### Command Line Interface

The project provides a comprehensive CLI for easy interaction:

```bash
# Train a new agent on World 1-1
mario-rl train --level 1-1 --timesteps 100000

# Train with complex movements and multiple environments
mario-rl train --movement complex --n-envs 4 --timesteps 500000

# Continue training from a checkpoint
mario-rl train --load-model checkpoints/ppo_mario_100000.zip

# Evaluate a trained model
mario-rl evaluate --model checkpoints/ppo_mario_final.zip --episodes 100

# Visualize training results
mario-rl visualize --log-file mario_training.log --type training
```

### Python API

For more control, use the Python API directly:

```python
from mario_rl.environments import MarioEnvironment
from mario_rl.agents import MarioAgent
from mario_rl.configs import TrainingConfig

# Create environment
env = MarioEnvironment(level="1-1", movement_type="simple")
vec_env = env.create_vectorized_env(n_envs=1)

# Create agent with configuration
config = TrainingConfig(
    level="1-1",
    movement_type="simple",
    total_timesteps=100000,
    learning_rate=3e-4
)
agent = MarioAgent(vec_env, config)

# Train the agent
model = agent.create_model()
agent.train()

# Evaluate performance
results = agent.evaluate(n_eval_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

### Examples

Run the included examples:

```bash
# Basic training example
python examples/basic_training.py

# Or use the Makefile
make run-mario
```

## 📊 Performance Monitoring

### Tensorboard Integration

Monitor training in real-time:

```bash
tensorboard --logdir mario_tensorboard/
```

### Training Logs

Detailed logging includes:
- Episode-by-episode performance
- Model checkpoint information
- Evaluation results with statistical analysis
- Error tracking and debugging information

## 🎮 Game Environment

### Supported Levels

The agent can be trained on any Super Mario Bros level:
- World 1: `1-1`, `1-2`, `1-3`, `1-4`
- World 2: `2-1`, `2-2`, `2-3`, `2-4`
- And all subsequent worlds (3-1 through 8-4)

### Movement Options

- **Simple**: 7 basic actions (walk, run, jump combinations)
- **Complex**: 12 advanced actions (including precise movements)

### Reward Structure

The agent receives rewards for:
- Forward progress (x-position advancement)
- Level completion (reaching the flag)
- Enemy elimination
- Coin collection

Penalties for:
- Time running out
- Losing a life
- Backward movement

## 🔧 Configuration

### Hyperparameters

Key training parameters in `TrainingConfig`:

```python
learning_rate: 3e-4        # PPO learning rate
n_steps: 512               # Steps per environment per update
batch_size: 64             # Minibatch size
n_epochs: 10               # PPO epochs per update
gamma: 0.99                # Discount factor
gae_lambda: 0.95           # GAE lambda
clip_range: 0.2            # PPO clipping parameter
```

### Environment Variables

Override configuration via environment variables:

```bash
export MARIO_LEARNING_RATE=5e-4
export MARIO_TIMESTEPS=1000000
export MARIO_LEVEL=2-1
export MARIO_N_ENVS=8
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mario_rl --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make ci
```

### Development Commands

```bash
# Install development dependencies
make install-dev

# Clean build artifacts
make clean

# Build package
make build

# Quick start development environment
make quick-start
```

## 📈 Results and Performance

### Typical Training Results

After 200,000 timesteps on World 1-1:
- **Completion Rate**: 85-95%
- **Average Reward**: 10-15
- **Max X-Position**: 3200+ (level length: 3266)
- **Training Time**: 2-4 hours (depending on hardware)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Contribution Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/mario.git
cd mario

# Install development dependencies
make install-dev

# Run tests to ensure everything works
make test

# Make your changes and run quality checks
make ci
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Gym](https://gym.openai.com/) for the RL environment framework
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO implementation
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) for the Mario environment
- The reinforcement learning community for continuous innovation

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- GitHub Issues: [Report bugs or request features](https://github.com/dustinober1/mario/issues)
- GitHub Discussions: [Join the conversation](https://github.com/dustinober1/mario/discussions)

---

⭐ **Star this repository if you find it useful for your RL projects!**
