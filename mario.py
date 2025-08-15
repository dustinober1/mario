"""
Super Mario Bros Reinforcement Learning Agent

This module implements a PPO-based reinforcement learning agent for playing
Super Mario Bros using stable-baselines3 and gym-super-mario-bros.
"""

import os
import logging
import argparse
from typing import Optional, Tuple
from datetime import datetime

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mario_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarioEnvironment:
    """Wrapper class for creating and managing Mario game environments."""
    
    def __init__(self, level: str = '1-1', movement_type: str = 'simple'):
        """
        Initialize Mario environment.
        
        Args:
            level: Game level (e.g., '1-1', '1-2', etc.)
            movement_type: 'simple' or 'complex' movement actions
        """
        self.level = level
        self.movement = SIMPLE_MOVEMENT if movement_type == 'simple' else COMPLEX_MOVEMENT
        
    def create_env(self) -> gym.Env:
        """Create a single Mario environment instance."""
        try:
            env_name = f'SuperMarioBros-{self.level}-v0'
            env = make(env_name)
            env = JoypadSpace(env, self.movement)
            env = Monitor(env)
            logger.info(f"Created environment: {env_name}")
            return env
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            raise
    
    def create_vectorized_env(self, n_envs: int = 1) -> DummyVecEnv:
        """Create vectorized environment for parallel training."""
        try:
            if n_envs == 1:
                vec_env = DummyVecEnv([lambda: self.create_env()])
            else:
                vec_env = SubprocVecEnv([lambda: self.create_env() for _ in range(n_envs)])
            logger.info(f"Created vectorized environment with {n_envs} processes")
            return vec_env
        except Exception as e:
            logger.error(f"Failed to create vectorized environment: {e}")
            raise


class MarioAgent:
    """PPO-based reinforcement learning agent for Mario."""
    
    def __init__(self, env: DummyVecEnv, model_name: str = "ppo_mario"):
        """
        Initialize the Mario RL agent.
        
        Args:
            env: Vectorized environment
            model_name: Name for saving/loading the model
        """
        self.env = env
        self.model_name = model_name
        self.model: Optional[PPO] = None
        self.training_history = []
        
    def create_model(self, **kwargs) -> PPO:
        """Create a new PPO model."""
        try:
            default_kwargs = {
                'policy': 'CnnPolicy',
                'env': self.env,
                'verbose': 1,
                'tensorboard_log': './mario_tensorboard/',
                'learning_rate': 3e-4,
                'n_steps': 512,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01
            }
            default_kwargs.update(kwargs)
            
            self.model = PPO(**default_kwargs)
            logger.info("Created new PPO model with parameters:")
            for key, value in default_kwargs.items():
                if key != 'env':
                    logger.info(f"  {key}: {value}")
            return self.model
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def train(self, total_timesteps: int, save_freq: int = 50000) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of training timesteps
            save_freq: Frequency for saving model checkpoints
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call create_model() first.")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = f"./checkpoints/{timestamp}/"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=checkpoint_dir,
                name_prefix=self.model_name
            )
            
            eval_callback = EvalCallback(
                self.env,
                best_model_save_path=checkpoint_dir,
                log_path=checkpoint_dir,
                eval_freq=save_freq,
                deterministic=True,
                render=False
            )
            
            logger.info(f"Starting training for {total_timesteps} timesteps")
            logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
            
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=True
            )
            
            final_model_path = f"{self.model_name}_{timestamp}"
            self.model.save(final_model_path)
            logger.info(f"Training completed. Final model saved as: {final_model_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        try:
            self.model = PPO.load(model_path)
            logger.info(f"Loaded model from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate(self, n_eval_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the trained agent.
        
        Args:
            n_eval_episodes: Number of episodes for evaluation
            
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        try:
            logger.info(f"Evaluating model over {n_eval_episodes} episodes")
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.env, 
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            )
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            return mean_reward, std_reward
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Mario RL Agent')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], 
                       default='both', help='Mode to run')
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Number of training timesteps')
    parser.add_argument('--level', type=str, default='1-1',
                       help='Mario level to play')
    parser.add_argument('--movement', choices=['simple', 'complex'], 
                       default='simple', help='Movement complexity')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to pre-trained model')
    parser.add_argument('--n_envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    return parser.parse_args()


def main():
    """Main training and evaluation pipeline."""
    args = parse_arguments()
    
    try:
        logger.info("Initializing Mario RL training environment")
        logger.info(f"Configuration: Level={args.level}, Movement={args.movement}, "
                   f"Timesteps={args.timesteps}, Environments={args.n_envs}")
        
        # Create environment
        mario_env = MarioEnvironment(level=args.level, movement_type=args.movement)
        vec_env = mario_env.create_vectorized_env(n_envs=args.n_envs)
        
        # Create agent
        agent = MarioAgent(vec_env)
        
        if args.mode in ['train', 'both']:
            if args.load_model:
                logger.info(f"Loading existing model: {args.load_model}")
                agent.load_model(args.load_model)
            else:
                agent.create_model()
            
            agent.train(total_timesteps=args.timesteps)
        
        if args.mode in ['evaluate', 'both']:
            if args.load_model and args.mode == 'evaluate':
                agent.load_model(args.load_model)
            
            mean_reward, std_reward = agent.evaluate(n_eval_episodes=args.eval_episodes)
            
            print(f"\n{'='*50}")
            print(f"FINAL EVALUATION RESULTS")
            print(f"{'='*50}")
            print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Episodes: {args.eval_episodes}")
            print(f"{'='*50}")
        
        vec_env.close()
        logger.info("Training/evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()