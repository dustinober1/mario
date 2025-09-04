#!/usr/bin/env python3
"""
Basic Mario RL Training Example

This script demonstrates how to train a Mario RL agent using the new package structure.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mario_rl.environments import MarioEnvironment
from mario_rl.agents import MarioAgent
from mario_rl.configs import TrainingConfig
from mario_rl.utils import setup_logging


def main():
    """Run basic training example."""
    # Setup logging
    logger = setup_logging(log_file="basic_training.log")
    logger.info("Starting basic Mario RL training example")
    
    # Create configuration
    config = TrainingConfig(
        level="1-1",
        movement_type="simple",
        n_envs=1,
        total_timesteps=50000,  # Reduced for example
        save_freq=10000,
        eval_freq=10000
    )
    
    try:
        # Create environment
        logger.info("Creating Mario environment...")
        env_wrapper = MarioEnvironment(
            level=config.level,
            movement_type=config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(config.n_envs)
        
        # Create agent
        logger.info("Creating Mario agent...")
        agent = MarioAgent(vec_env, config)
        
        # Create and train model
        logger.info("Creating PPO model...")
        model = agent.create_model()
        
        logger.info("Starting training...")
        agent.train(config.total_timesteps)
        
        # Save final model
        logger.info("Saving trained model...")
        agent.save_model()
        
        # Evaluate final model
        logger.info("Evaluating trained model...")
        results = agent.evaluate(n_eval_episodes=5)
        logger.info(f"Evaluation results: {results}")
        
        logger.info("Training example completed successfully!")
        
    except Exception as e:
        logger.error(f"Training example failed: {e}")
        raise


if __name__ == "__main__":
    main()
