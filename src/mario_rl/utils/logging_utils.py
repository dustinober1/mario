"""Logging utilities for Mario RL."""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
        log_format: Log message format
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('mario_rl')
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(f'mario_rl.{name}')


def log_training_start(config: dict) -> None:
    """Log training configuration at start."""
    logger = get_logger('training')
    logger.info("Starting Mario RL training")
    logger.info(f"Configuration: {config}")


def log_training_progress(episode: int, reward: float, info: dict) -> None:
    """Log training progress."""
    logger = get_logger('training')
    logger.info(f"Episode {episode}: Reward = {reward:.2f}, Info = {info}")


def log_evaluation_results(results: dict) -> None:
    """Log evaluation results."""
    logger = get_logger('evaluation')
    logger.info(f"Evaluation results: {results}")


def log_model_save(path: str) -> None:
    """Log model save operation."""
    logger = get_logger('model')
    logger.info(f"Model saved to: {path}")


def log_model_load(path: str) -> None:
    """Log model load operation."""
    logger = get_logger('model')
    logger.info(f"Model loaded from: {path}")
