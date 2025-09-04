"""Mario game environment wrapper for reinforcement learning."""

import logging
from typing import Optional

import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace

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
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space."""
        return len(self.movement)
    
    def get_observation_space(self) -> gym.Space:
        """Get the observation space of the environment."""
        env = self.create_env()
        obs_space = env.observation_space
        env.close()
        return obs_space
    
    def validate_level(self, level: str) -> bool:
        """Validate if a level string is in the correct format."""
        try:
            world, stage = level.split('-')
            world_num = int(world)
            stage_num = int(stage)
            return 1 <= world_num <= 8 and 1 <= stage_num <= 4
        except (ValueError, AttributeError):
            return False
