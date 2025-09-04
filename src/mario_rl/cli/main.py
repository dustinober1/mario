"""Main CLI entry point for Mario RL."""

import argparse
import sys
from pathlib import Path

from .train import add_train_parser
from .evaluate import add_evaluate_parser
from .visualize import add_visualize_parser


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description='Mario RL - Deep Reinforcement Learning for Super Mario Bros',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new agent on World 1-1
  mario-rl train --level 1-1 --timesteps 100000

  # Train with complex movements
  mario-rl train --movement complex --n-envs 4

  # Continue training from checkpoint
  mario-rl train --load-model checkpoints/ppo_mario_100000.zip

  # Evaluate a trained model
  mario-rl evaluate --model checkpoints/ppo_mario_final.zip

  # Visualize training results
  mario-rl visualize --log-file mario_training.log
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )
    
    add_train_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_visualize_parser(subparsers)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
