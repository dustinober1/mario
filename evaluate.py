"""Evaluation script for trained Mario RL models."""

import os
import argparse
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from mario import MarioEnvironment
from utils import setup_logging, plot_training_metrics, TrainingLogger
from config import TrainingConfig


def evaluate_model_comprehensive(
    model_path: str,
    config: TrainingConfig,
    n_episodes: int = 100,
    render: bool = False,
    save_video: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a trained model.
    
    Args:
        model_path: Path to the trained model
        config: Training configuration
        n_episodes: Number of episodes to evaluate
        render: Whether to render the game
        save_video: Whether to save evaluation videos
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger = logging.getLogger(__name__)
    
    # Create environment
    mario_env = MarioEnvironment(level=config.level, movement_type=config.movement_type)
    env = mario_env.create_env()
    
    # Load model
    try:
        model = PPO.load(model_path)
        logger.info(f"Loaded model from: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Evaluation metrics
    rewards = []
    episode_lengths = []
    x_positions = []
    completion_rates = []
    times_remaining = []
    
    logger.info(f"Starting comprehensive evaluation over {n_episodes} episodes")
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        max_x_pos = 0
        completed = False
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Track max x position
            if 'x_pos' in info:
                max_x_pos = max(max_x_pos, info['x_pos'])
            
            # Check completion
            if 'flag_get' in info and info['flag_get']:
                completed = True
            
            if render:
                env.render()
            
            if done:
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        x_positions.append(max_x_pos)
        completion_rates.append(1.0 if completed else 0.0)
        
        if 'time' in info:
            times_remaining.append(info['time'])
        else:
            times_remaining.append(0)
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Completed {episode + 1}/{n_episodes} episodes")
    
    env.close()
    
    # Calculate statistics
    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'median_reward': np.median(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_x_position': np.mean(x_positions),
        'max_x_position': np.max(x_positions),
        'completion_rate': np.mean(completion_rates),
        'mean_time_remaining': np.mean(times_remaining),
        'total_episodes': n_episodes,
        'level': config.level,
        'movement_type': config.movement_type
    }
    
    logger.info("Evaluation completed!")
    logger.info(f"Results: {results}")
    
    return results


def compare_models(model_paths: List[str], config: TrainingConfig, n_episodes: int = 50):
    """Compare multiple trained models."""
    logger = logging.getLogger(__name__)
    
    results = {}
    
    for model_path in model_paths:
        logger.info(f"Evaluating model: {model_path}")
        model_results = evaluate_model_comprehensive(
            model_path, config, n_episodes, render=False
        )
        results[model_path] = model_results
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(results.keys())
    mean_rewards = [results[model]['mean_reward'] for model in models]
    completion_rates = [results[model]['completion_rate'] for model in models]
    mean_x_positions = [results[model]['mean_x_position'] for model in models]
    mean_lengths = [results[model]['mean_episode_length'] for model in models]
    
    # Plot comparisons
    axes[0, 0].bar(range(len(models)), mean_rewards)
    axes[0, 0].set_title('Mean Reward Comparison')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels([os.path.basename(m) for m in models], rotation=45)
    
    axes[0, 1].bar(range(len(models)), completion_rates)
    axes[0, 1].set_title('Completion Rate Comparison')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels([os.path.basename(m) for m in models], rotation=45)
    
    axes[1, 0].bar(range(len(models)), mean_x_positions)
    axes[1, 0].set_title('Mean X Position Comparison')
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels([os.path.basename(m) for m in models], rotation=45)
    
    axes[1, 1].bar(range(len(models)), mean_lengths)
    axes[1, 1].set_title('Mean Episode Length Comparison')
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels([os.path.basename(m) for m in models], rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate Mario RL models')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--level', type=str, default='1-1',
                       help='Mario level to evaluate on')
    parser.add_argument('--movement', choices=['simple', 'complex'],
                       default='simple', help='Movement type')
    parser.add_argument('--render', action='store_true',
                       help='Render evaluation episodes')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple models')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    setup_logging("evaluation.log")
    logger = logging.getLogger(__name__)
    
    config = TrainingConfig()
    config.level = args.level
    config.movement_type = args.movement
    
    try:
        if args.compare:
            logger.info(f"Comparing models: {args.compare}")
            results = compare_models(args.compare, config, args.episodes)
        else:
            logger.info(f"Evaluating single model: {args.model_path}")
            results = evaluate_model_comprehensive(
                args.model_path, config, args.episodes, args.render
            )
        
        # Save results
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()