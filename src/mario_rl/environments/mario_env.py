"""Mario game environment wrapper for reinforcement learning."""

import logging
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
import cv2

logger = logging.getLogger(__name__)


class FrameSkipWrapper(gym.Wrapper):
    """Wrapper to skip frames and return only every nth frame."""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleWrapper(gym.ObservationWrapper):
    """Convert RGB observation to grayscale."""
    
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(old_shape[0], old_shape[1], 1), dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(observation, axis=-1)


class ResizeWrapper(gym.ObservationWrapper):
    """Resize observation to specified shape."""
    
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shape[0], shape[1], 1), dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(observation, axis=-1)


class NormalizeWrapper(gym.ObservationWrapper):
    """Normalize observations to [0, 1] range."""
    
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=old_shape, dtype=np.float32
        )

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

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
            
            # Apply preprocessing wrappers
            env = FrameSkipWrapper(env, skip=4)
            env = GrayScaleWrapper(env)
            env = ResizeWrapper(env, shape=(84, 84))
            env = NormalizeWrapper(env)
            
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
