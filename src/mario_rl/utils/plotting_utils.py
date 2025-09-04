"""Plotting utilities for Mario RL."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


def plot_training_progress(
    rewards: List[float],
    losses: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training progress.
    
    Args:
        rewards: List of episode rewards
        losses: Dictionary of loss values (optional)
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2 if losses else 1, figsize=(15, 5))
    if not losses:
        axes = [axes]
    
    # Plot rewards
    ax1 = axes[0]
    ax1.plot(rewards, alpha=0.6, color='blue')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot losses if provided
    if losses:
        ax2 = axes[1]
        for loss_name, loss_values in losses.items():
            ax2.plot(loss_values, label=loss_name, alpha=0.7)
        ax2.set_title('Training Losses')
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_evaluation_results(
    results: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot evaluation results.
    
    Args:
        results: Dictionary of evaluation metrics
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(results.items()):
        ax = axes[i]
        
        # Create histogram
        ax.hist(values, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.set_title(f'{metric_name} Distribution')
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        ax.axvline(mean_val, color='red', linestyle='--', 
                   label=f'Mean: {mean_val:.2f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', 
                   label=f'+1σ: {mean_val + std_val:.2f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', 
                   label=f'-1σ: {mean_val - std_val:.2f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_model_comparison(
    models: Dict[str, Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot model comparison.
    
    Args:
        models: Dictionary of model results
        metrics: List of metrics to compare
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Prepare data
    data = []
    for model_name, model_results in models.items():
        for metric in metrics:
            if metric in model_results:
                data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': model_results[metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Value', hue='Model')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curves(
    training_data: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot learning curves.
    
    Args:
        training_data: Dictionary of training metrics over time
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    for metric_name, values in training_data.items():
        plt.plot(values, label=metric_name, alpha=0.8)
    
    plt.title('Learning Curves')
    plt.xlabel('Training Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_training_summary_plot(
    rewards: List[float],
    losses: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Create a comprehensive training summary plot.
    
    Args:
        rewards: List of episode rewards
        losses: Dictionary of loss values (optional)
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    n_plots = 2 if losses else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # Plot rewards with moving average
    ax1 = axes[0]
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, alpha=0.6, color='blue', label='Episode Reward')
    
    # Add moving average
    if len(rewards) > 10:
        window = min(100, len(rewards) // 10)
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        ax1.plot(episodes, moving_avg, color='red', linewidth=2, 
                 label=f'Moving Average (window={window})')
    
    ax1.set_title('Training Progress - Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot losses if provided
    if losses:
        ax2 = axes[1]
        for loss_name, loss_values in losses.items():
            ax2.plot(loss_values, label=loss_name, alpha=0.7)
        ax2.set_title('Training Losses')
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
