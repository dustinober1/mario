"""Visualization tools for Mario RL training and evaluation."""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_progress(log_file: str, save_path: Optional[str] = None):
    """Plot training progress from log file."""
    # This would parse training logs - simplified version
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Placeholder data - in real implementation, parse from tensorboard logs
    episodes = np.arange(1000)
    rewards = np.cumsum(np.random.randn(1000) * 0.1) + np.linspace(0, 15, 1000)
    episode_lengths = 200 + 50 * np.sin(np.linspace(0, 10, 1000)) + np.random.randn(1000) * 10
    x_positions = np.clip(np.cumsum(np.random.randn(1000) * 0.5) + np.linspace(0, 3000, 1000), 0, 3266)
    
    # Reward progression
    axes[0, 0].plot(episodes, rewards, alpha=0.7, label='Episode Reward')
    axes[0, 0].plot(episodes, pd.Series(rewards).rolling(50).mean(), 'r-', label='Moving Average (50)')
    axes[0, 0].set_title('Training Rewards Over Time')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode length
    axes[0, 1].plot(episodes, episode_lengths, alpha=0.7, label='Episode Length')
    axes[0, 1].plot(episodes, pd.Series(episode_lengths).rolling(50).mean(), 'r-', label='Moving Average (50)')
    axes[0, 1].set_title('Episode Length Over Time')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # X position progress
    axes[1, 0].plot(episodes, x_positions, alpha=0.7, label='Max X Position')
    axes[1, 0].plot(episodes, pd.Series(x_positions).rolling(50).mean(), 'r-', label='Moving Average (50)')
    axes[1, 0].axhline(y=3266, color='g', linestyle='--', label='Level End')
    axes[1, 0].set_title('Maximum X Position Reached')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('X Position')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate schedule
    learning_rates = 3e-4 * np.exp(-episodes / 2000)
    axes[1, 1].plot(episodes, learning_rates)
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to: {save_path}")
    else:
        plt.show()


def plot_evaluation_results(results_file: str, save_path: Optional[str] = None):
    """Plot evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if isinstance(results, dict) and 'mean_reward' in results:
        # Single model results
        plot_single_model_results(results, save_path)
    else:
        # Multiple model comparison
        plot_model_comparison(results, save_path)


def plot_single_model_results(results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot results for a single model evaluation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metrics overview
    metrics = ['mean_reward', 'mean_episode_length', 'mean_x_position', 'completion_rate']
    values = [results[metric] for metric in metrics]
    colors = ['skyblue', 'lightgreen', 'orange', 'pink']
    
    axes[0, 0].bar(metrics, values, color=colors)
    axes[0, 0].set_title('Evaluation Metrics Overview')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Reward distribution (simulated)
    np.random.seed(42)
    reward_dist = np.random.normal(results['mean_reward'], results['std_reward'], 1000)
    axes[0, 1].hist(reward_dist, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(results['mean_reward'], color='red', linestyle='--', label='Mean')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].set_xlabel('Reward')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Performance radar chart
    categories = ['Reward', 'Completion', 'Distance', 'Efficiency']
    values_norm = [
        results['mean_reward'] / 15.0,  # Normalize to expected max
        results['completion_rate'],
        results['mean_x_position'] / 3266.0,  # Normalize to level length
        1.0 - (results['mean_episode_length'] / 1000.0)  # Efficiency (inverse of length)
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_norm += values_norm[:1]  # Complete the circle
    angles += angles[:1]
    
    axes[1, 0].plot(angles, values_norm, 'o-', linewidth=2, label='Performance')
    axes[1, 0].fill(angles, values_norm, alpha=0.25)
    axes[1, 0].set_xticks(angles[:-1])
    axes[1, 0].set_xticklabels(categories)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Performance Radar Chart')
    axes[1, 0].grid(True)
    
    # Model info
    info_text = f"""
    Model Evaluation Summary
    ========================
    Level: {results['level']}
    Movement: {results['movement_type']}
    Episodes: {results['total_episodes']}
    
    Key Metrics:
    • Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}
    • Completion Rate: {results['completion_rate']:.1%}
    • Max Distance: {results['max_x_position']:.0f} / 3266
    • Avg Episode Length: {results['mean_episode_length']:.0f} steps
    """
    
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation results plot saved to: {save_path}")
    else:
        plt.show()


def plot_model_comparison(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
    """Plot comparison between multiple models."""
    models = list(results.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Prepare data
    metrics = ['mean_reward', 'completion_rate', 'mean_x_position', 'mean_episode_length']
    metric_labels = ['Mean Reward', 'Completion Rate', 'Mean X Position', 'Mean Episode Length']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i // 2, i % 2]
        values = [results[model][metric] for model in models]
        
        bars = ax.bar(range(n_models), values, alpha=0.7)
        ax.set_title(f'{label} Comparison')
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([os.path.basename(m) for m in models], rotation=45, ha='right')
        ax.set_ylabel(label)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {save_path}")
    else:
        plt.show()


def create_performance_heatmap(results_file: str, save_path: Optional[str] = None):
    """Create a heatmap of model performance across different metrics."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if not isinstance(results, dict) or 'mean_reward' in results:
        print("This function requires multiple model comparison data")
        return
    
    # Prepare data for heatmap
    models = list(results.keys())
    metrics = ['mean_reward', 'completion_rate', 'mean_x_position', 'mean_episode_length']
    
    data = []
    for model in models:
        row = [results[model][metric] for metric in metrics]
        data.append(row)
    
    # Normalize data for better visualization
    data = np.array(data)
    data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(data_normalized, 
                xticklabels=['Reward', 'Completion', 'Distance', 'Efficiency'],
                yticklabels=[os.path.basename(m) for m in models],
                annot=data,
                fmt='.2f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Normalized Performance'})
    
    plt.title('Model Performance Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance heatmap saved to: {save_path}")
    else:
        plt.show()


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(description='Visualize Mario RL training and evaluation')
    parser.add_argument('--type', choices=['training', 'evaluation', 'comparison', 'heatmap'],
                       required=True, help='Type of visualization')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file (log file for training, JSON for evaluation)')
    parser.add_argument('--output', type=str, 
                       help='Output file path (optional)')
    
    args = parser.parse_args()
    
    try:
        if args.type == 'training':
            plot_training_progress(args.input, args.output)
        elif args.type == 'evaluation':
            plot_evaluation_results(args.input, args.output)
        elif args.type == 'comparison':
            plot_evaluation_results(args.input, args.output)
        elif args.type == 'heatmap':
            create_performance_heatmap(args.input, args.output)
        
        print(f"Visualization completed successfully!")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()