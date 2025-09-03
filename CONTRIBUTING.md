# Contributing to Mario RL

Thank you for your interest in contributing to the Mario RL project! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Code Review Guidelines](#code-review-guidelines)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of reinforcement learning concepts
- Familiarity with PyTorch and stable-baselines3 (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mario.git
   cd mario
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/dustinober1/mario.git
   ```

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv mario_env
source mario_env/bin/activate  # On Windows: mario_env\Scripts\activate
```

### 2. Install Development Dependencies

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

### 4. Verify Setup

```bash
# Run tests to ensure everything is working
pytest tests/

# Check code formatting
black --check .

# Run linting
flake8 .

# Type checking
mypy .
```

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **String quotes**: Double quotes for docstrings, single quotes for strings
- **Import sorting**: Use `isort` for consistent import ordering

### Code Formatting

We use [Black](https://black.readthedocs.io/) for automatic code formatting:

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

### Import Sorting

We use [isort](https://pycqa.github.io/isort/) for import organization:

```bash
# Sort imports
isort .

# Check import sorting
isort --check-only .
```

### Type Hints

We use type hints throughout the codebase:

```python
from typing import List, Optional, Tuple
import numpy as np

def process_frames(frames: List[np.ndarray], 
                  normalize: bool = True) -> np.ndarray:
    """Process a list of frames.
    
    Args:
        frames: List of input frames
        normalize: Whether to normalize pixel values
        
    Returns:
        Processed frames as a single array
    """
    # Implementation here
    pass
```

### Documentation

- All public functions and classes must have docstrings
- Use Google-style docstring format
- Include type hints in docstrings
- Provide examples for complex functions

Example:
```python
def train_agent(env: gym.Env, 
                model: BaseAlgorithm,
                total_timesteps: int = 1000000) -> BaseAlgorithm:
    """Train a reinforcement learning agent.
    
    Args:
        env: The training environment
        model: The model to train
        total_timesteps: Total number of training timesteps
        
    Returns:
        The trained model
        
    Example:
        >>> env = make_mario_env("1-1")
        >>> model = PPO("CnnPolicy", env)
        >>> trained_model = train_agent(env, model, 100000)
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=mario --cov-report=html

# Run specific test file
pytest tests/test_mario.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Tests should be in the `tests/` directory
- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Use descriptive test names that explain what is being tested

Example:
```python
import pytest
import numpy as np
from mario.environment import MarioEnvironment

def test_mario_environment_initialization():
    """Test that Mario environment initializes correctly."""
    env = MarioEnvironment("1-1")
    assert env.level == "1-1"
    assert env.action_space is not None
    assert env.observation_space is not None

def test_mario_environment_reset():
    """Test that environment reset returns valid observation."""
    env = MarioEnvironment("1-1")
    obs = env.reset()
    assert obs.shape == (84, 84, 4)
    assert obs.dtype == np.uint8
```

### Test Coverage

We aim for at least 80% test coverage. To check coverage:

```bash
pytest --cov=mario --cov-report=term-missing
```

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, focused commits
- Each commit should represent a logical change
- Use descriptive commit messages

### 3. Run Tests and Quality Checks

```bash
# Run all tests
pytest

# Check code quality
pre-commit run --all-files
```

### 4. Update Documentation

- Update README.md if needed
- Add docstrings for new functions
- Update CHANGELOG.md for new features

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Reference to related issues
- Screenshots if UI changes
- Test results

### 6. Code Review

- Address review comments promptly
- Make requested changes in new commits
- Keep the PR focused and manageable

## Reporting Bugs

### Before Reporting

1. Check if the bug has already been reported
2. Try to reproduce the bug with the latest version
3. Check the documentation and existing issues

### Bug Report Template

Use the issue template and include:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment information
- Error messages and logs
- Minimal code example if applicable

## Feature Requests

### Before Requesting

1. Check if the feature already exists
2. Search existing issues and discussions
3. Consider if the feature fits the project scope

### Feature Request Template

- Clear description of the feature
- Use cases and benefits
- Implementation suggestions if possible
- Mockups or examples if applicable

## Code Review Guidelines

### For Reviewers

- Be constructive and respectful
- Focus on code quality and functionality
- Provide specific, actionable feedback
- Consider the contributor's experience level

### For Contributors

- Respond to review comments promptly
- Ask questions if feedback is unclear
- Be open to suggestions and improvements
- Don't take feedback personally

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH**
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are updated
- [ ] Release notes are prepared
- [ ] GitHub release is created

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check the README and docstrings first

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to Mario RL! Your contributions help make this project better for everyone.
