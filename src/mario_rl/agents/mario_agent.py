"""PPO-based reinforcement learning agent for Mario."""

import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from ..configs import TrainingConfig

logger = logging.getLogger(__name__)


class MarioAgent:
    """PPO-based reinforcement learning agent for Mario."""
    
    def __init__(self, env: DummyVecEnv, config: TrainingConfig):
        """
        Initialize the Mario RL agent.
        
        Args:
            env: Vectorized environment
            config: Training configuration
        """
        self.env = env
        self.config = config
        self.model: Optional[PPO] = None
        self.training_history: List[Dict[str, Any]] = []
        
    def create_model(self, **kwargs) -> PPO:
        """Create a new PPO model."""
        try:
            # Use config values as defaults, allow override via kwargs
            model_kwargs = {
                'policy': 'CnnPolicy',
                'env': self.env,
                'verbose': 1,
                'learning_rate': self.config.learning_rate,
                'n_steps': self.config.n_steps,
                'batch_size': self.config.batch_size,
                'n_epochs': self.config.n_epochs,
                'gamma': self.config.gamma,
                'gae_lambda': self.config.gae_lambda,
                'clip_range': self.config.clip_range,
                'ent_coef': self.config.ent_coef,
                'vf_coef': self.config.vf_coef,
                'max_grad_norm': self.config.max_grad_norm,
                'tensorboard_log': self.config.tensorboard_dir
            }
            model_kwargs.update(kwargs)
            
            self.model = PPO(**model_kwargs)
            logger.info("Created new PPO model")
            return self.model
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def train(self, total_timesteps: Optional[int] = None) -> None:
        """Train the agent."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        try:
            logger.info(f"Starting training for {total_timesteps} timesteps")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _create_callbacks(self) -> List:
        """Create training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=self.config.checkpoint_dir,
            name_prefix=self.config.model_name
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=self.config.checkpoint_dir,
            log_path=self.config.checkpoint_dir,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        return callbacks
    
    def evaluate(self, n_eval_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the trained agent."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        try:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True
            )
            
            results = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'n_episodes': n_eval_episodes
            }
            
            logger.info(f"Evaluation results: {results}")
            return results
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def save_model(self, path: Optional[str] = None) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{self.config.checkpoint_dir}/{self.config.model_name}_{timestamp}.zip"
        
        try:
            self.model.save(path)
            logger.info(f"Model saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        try:
            self.model = PPO.load(path, env=self.env)
            logger.info(f"Model loaded from: {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple:
        """Make a prediction given an observation."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get the training history."""
        return self.training_history.copy()
