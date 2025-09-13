"""
Mario RL - Deep Reinforcement Learning for Super Mario Bros

A professional-grade deep reinforcement learning implementation for training
AI agents to play Super Mario Bros using PPO and stable-baselines3.
"""

__version__ = "1.0.0"
__author__ = "Reinforcement Learning Engineer"
__email__ = "developer@example.com"

# Main imports
from .agents import MarioAgent
from .environments import MarioEnvironment
from .configs import TrainingConfig, EnvironmentConfig

__all__ = [
    "MarioAgent",
    "MarioEnvironment", 
    "TrainingConfig",
    "EnvironmentConfig"
]