"""Command-line interface module for Mario RL."""

from .train import train_command
from .evaluate import evaluate_command
from .visualize import visualize_command

__all__ = ["train_command", "evaluate_command", "visualize_command"]
