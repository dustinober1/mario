# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD pipeline
- Issue and Pull Request templates
- Changelog documentation
- Code of Conduct
- Contributing guidelines

### Changed
- Improved project documentation structure

## [1.0.0] - 2024-01-15

### Added
- Initial release of Mario RL project
- PPO algorithm implementation with stable-baselines3
- Mario environment integration with gym-super-mario-bros
- Comprehensive training pipeline with callbacks
- Model evaluation and comparison tools
- Advanced visualization utilities
- Configuration management system
- Tensorboard integration for training monitoring
- Multi-environment training support
- Automatic checkpointing and model saving
- Performance metrics and logging

### Features
- Support for multiple Mario levels (World 1-1 through 8-4)
- Configurable movement options (simple and complex)
- Reward structure optimization for Mario gameplay
- Hyperparameter tuning capabilities
- Model versioning and comparison
- Training progress visualization
- Performance benchmarking tools

### Technical Details
- CNN-based policy and value networks
- Frame stacking for temporal information
- Reward clipping and normalization
- Action space discretization
- Environment preprocessing pipeline

## [0.1.0] - 2024-01-01

### Added
- Project initialization
- Basic Mario environment setup
- Initial PPO implementation
- Basic training loop

---

## Version History

- **1.0.0**: First stable release with full feature set
- **0.1.0**: Initial development version

## Migration Guide

### From 0.1.0 to 1.0.0
- Updated import statements for new module structure
- New configuration system requires config file updates
- Enhanced training pipeline with new callback system
- Improved model saving/loading format

## Deprecation Notices

- None currently

## Breaking Changes

- None currently
