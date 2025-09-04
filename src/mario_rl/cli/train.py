"""Training command for Mario RL CLI."""

import argparse
import logging
from pathlib import Path

from ..environments import MarioEnvironment
from ..agents import MarioAgent
from ..configs import TrainingConfig, load_config_from_env
from ..utils import setup_logging


def train_command(args: argparse.Namespace) -> None:
    """Execute training command."""
    # Setup logging
    logger = setup_logging(
        log_file=args.log_file,
        log_level=getattr(logging, args.log_level.upper())
    )
    
    # Load configuration
    config = load_config_from_env()
    
    # Override with command line arguments
    if args.level:
        config.level = args.level
    if args.movement:
        config.movement_type = args.movement
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.n_envs:
        config.n_envs = args.n_envs
    if args.model_name:
        config.model_name = args.model_name
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.tensorboard_dir:
        config.tensorboard_dir = args.tensorboard_dir
    
    logger.info(f"Training configuration: {config}")
    
    try:
        # Create environment
        env_wrapper = MarioEnvironment(
            level=config.level,
            movement_type=config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(config.n_envs)
        
        # Create agent
        agent = MarioAgent(vec_env, config)
        
        # Create model
        model = agent.create_model()
        
        # Load existing model if specified
        if args.load_model:
            agent.load_model(args.load_model)
            logger.info(f"Loaded existing model from: {args.load_model}")
        
        # Train agent
        agent.train(config.total_timesteps)
        
        # Save final model
        agent.save_model()
        
        # Evaluate final model
        if args.eval_episodes > 0:
            results = agent.evaluate(args.eval_episodes)
            logger.info(f"Final evaluation results: {results}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add training parser to subparsers."""
    train_parser = subparsers.add_parser(
        'train',
        help='Train a Mario RL agent'
    )
    
    # Environment options
    train_parser.add_argument(
        '--level',
        type=str,
        default='1-1',
        help='Mario level to train on (e.g., 1-1, 1-2)'
    )
    train_parser.add_argument(
        '--movement',
        type=str,
        choices=['simple', 'complex'],
        default='simple',
        help='Movement type (simple or complex)'
    )
    train_parser.add_argument(
        '--n-envs',
        type=int,
        default=1,
        help='Number of parallel environments'
    )
    
    # Training options
    train_parser.add_argument(
        '--timesteps',
        type=int,
        default=200000,
        help='Total training timesteps'
    )
    train_parser.add_argument(
        '--model-name',
        type=str,
        default='ppo_mario',
        help='Model name for saving/loading'
    )
    
    # Paths
    train_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory for model checkpoints'
    )
    train_parser.add_argument(
        '--tensorboard-dir',
        type=str,
        default='./mario_tensorboard',
        help='Directory for tensorboard logs'
    )
    train_parser.add_argument(
        '--log-file',
        type=str,
        default='mario_training.log',
        help='Log file path'
    )
    
    # Model options
    train_parser.add_argument(
        '--load-model',
        type=str,
        help='Path to load existing model from'
    )
    
    # Evaluation options
    train_parser.add_argument(
        '--eval-episodes',
        type=int,
        default=10,
        help='Number of episodes for final evaluation'
    )
    
    # Logging options
    train_parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    train_parser.set_defaults(func=train_command)
