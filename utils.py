"""Utility functions for Mario RL project."""

import os
import json
import pickle
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def setup_logging(log_file: str = "mario_training.log", level: int = logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_directories(dirs: List[str]):
    """Create directories if they don't exist."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_training_history(history: List[Dict[str, Any]], filepath: str):
    """Save training history to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(history, f)


def load_training_history(filepath: str) -> List[Dict[str, Any]]:
    """Load training history from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_training_metrics(
    rewards: List[float], 
    episode_lengths: List[int],
    save_path: Optional[str] = None
):
    """Plot training metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Training Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def calculate_moving_average(data: List[float], window: int = 100) -> List[float]:
    """Calculate moving average of data."""
    if len(data) < window:
        return data
    
    moving_avg = []
    for i in range(len(data)):
        if i < window:
            moving_avg.append(np.mean(data[:i+1]))
        else:
            moving_avg.append(np.mean(data[i-window+1:i+1]))
    
    return moving_avg


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_model_size(model_path: str) -> str:
    """Get model file size in human readable format."""
    size_bytes = os.path.getsize(model_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


class TrainingLogger:
    """Enhanced logging for training progress."""
    
    def __init__(self, log_file: str = "training_progress.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("TrainingLogger")
        self.start_time = datetime.now()
        self.episode_count = 0
        
    def log_episode(self, reward: float, length: int, info: Dict[str, Any]):
        """Log episode information."""
        self.episode_count += 1
        elapsed_time = datetime.now() - self.start_time
        
        self.logger.info(
            f"Episode {self.episode_count}: "
            f"Reward={reward:.2f}, Length={length}, "
            f"Elapsed={format_time(elapsed_time.total_seconds())}"
        )
        
        if 'x_pos' in info:
            self.logger.info(f"  Max X Position: {info['x_pos']}")
        if 'time' in info:
            self.logger.info(f"  Time Remaining: {info['time']}")
    
    def log_checkpoint(self, timestep: int, model_path: str):
        """Log checkpoint information."""
        model_size = get_model_size(model_path)
        self.logger.info(
            f"Checkpoint saved at timestep {timestep}: "
            f"{model_path} (Size: {model_size})"
        )
    
    def log_evaluation(self, mean_reward: float, std_reward: float, n_episodes: int):
        """Log evaluation results."""
        self.logger.info(
            f"Evaluation over {n_episodes} episodes: "
            f"Mean={mean_reward:.2f}±{std_reward:.2f}"
        )