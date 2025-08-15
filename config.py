"""Configuration settings for Mario RL training."""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for RL training parameters."""
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training settings
    total_timesteps: int = 200000
    save_freq: int = 50000
    eval_freq: int = 50000
    n_eval_episodes: int = 10
    
    # Environment settings
    level: str = "1-1"
    movement_type: str = "simple"
    n_envs: int = 1
    
    # Paths
    model_name: str = "ppo_mario"
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./mario_tensorboard"
    log_file: str = "mario_training.log"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm
        }


@dataclass
class EnvironmentConfig:
    """Configuration for game environment."""
    
    # Available Mario levels
    LEVELS = [
        "1-1", "1-2", "1-3", "1-4",
        "2-1", "2-2", "2-3", "2-4",
        "3-1", "3-2", "3-3", "3-4",
        "4-1", "4-2", "4-3", "4-4",
        "5-1", "5-2", "5-3", "5-4",
        "6-1", "6-2", "6-3", "6-4",
        "7-1", "7-2", "7-3", "7-4",
        "8-1", "8-2", "8-3", "8-4"
    ]
    
    # Movement types
    MOVEMENT_TYPES = ["simple", "complex"]
    
    # Reward thresholds
    COMPLETION_REWARD = 15.0
    TIME_PENALTY = -0.1
    DEATH_PENALTY = -15.0


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def load_config_from_env() -> TrainingConfig:
    """Load configuration from environment variables."""
    config = TrainingConfig()
    
    # Override with environment variables if they exist
    config.learning_rate = float(os.getenv('MARIO_LEARNING_RATE', config.learning_rate))
    config.total_timesteps = int(os.getenv('MARIO_TIMESTEPS', config.total_timesteps))
    config.level = os.getenv('MARIO_LEVEL', config.level)
    config.movement_type = os.getenv('MARIO_MOVEMENT', config.movement_type)
    config.n_envs = int(os.getenv('MARIO_N_ENVS', config.n_envs))
    
    return config