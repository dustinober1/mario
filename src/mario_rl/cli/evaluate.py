"""Evaluation command for Mario RL CLI."""

import argparse


def evaluate_command(args: argparse.Namespace) -> None:
    """Execute evaluation command."""
    print("Evaluation command not yet implemented")
    print(f"Args: {args}")


def add_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add evaluation parser to subparsers."""
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained Mario RL agent'
    )
    
    eval_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the model to evaluate'
    )
    
    eval_parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes to evaluate'
    )
    
    eval_parser.add_argument(
        '--render',
        action='store_true',
        help='Render the evaluation'
    )
    
    eval_parser.set_defaults(func=evaluate_command)
