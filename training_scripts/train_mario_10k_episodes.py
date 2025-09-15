#!/usr/bin/env python3
"""
🚀 Mario RL Training - 10,000 Episodes with Apple Silicon MPS GPU
"""

import os
import time
import warnings
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

# Suppress gym deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

print("🚀 Mario RL Training - 10,000 Episodes")
print("=" * 55)

print("1. Loading libraries...")
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make
print("   ✓ Libraries loaded")

# Add project root to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


class GymToGymnasiumWrapper(gym.Env):
    """Wrapper to make old gym environments compatible with gymnasium."""
    
    def __init__(self, env):
        super().__init__()
        self._env = env
        # Convert old gym spaces to gymnasium spaces
        if hasattr(env.observation_space, 'shape'):
            self.observation_space = gym.spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype
            )
        else:
            self.observation_space = env.observation_space
            
        # Convert action space
        if hasattr(env.action_space, 'n'):
            self.action_space = gym.spaces.Discrete(env.action_space.n)
        else:
            self.action_space = env.action_space
        
    def reset(self, **kwargs):
        obs = self._env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, False, info
    
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()


def create_mario_env():
    """Create a Mario environment with minimal warnings."""
    # Suppress environment-specific warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make('SuperMarioBros-1-1-v3')  # Use v3 instead of v0
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GymToGymnasiumWrapper(env)
    return env


class EpisodeCounterCallback(BaseCallback):
    """
    Custom callback to count episodes and stop after reaching target
    """
    def __init__(self, target_episodes=10000, verbose=1):
        super().__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Progress updates
            if self.episode_count % 100 == 0:
                elapsed = time.time() - self.start_time
                progress = (self.episode_count / self.target_episodes) * 100
                eta = (elapsed / self.episode_count) * (self.target_episodes - self.episode_count)
                
                print(f"   📊 Episode {self.episode_count:,}/{self.target_episodes:,} "
                      f"({progress:.1f}%) - ETA: {eta/60:.1f}m")
            
            # Stop when target reached
            if self.episode_count >= self.target_episodes:
                print(f"\n🎯 Target reached! Completed {self.episode_count:,} episodes")
                return False
                
        return True


def check_mps_availability():
    """Check if MPS (Metal Performance Shaders) is available"""
    if torch.backends.mps.is_available():
        print("   🍎 Using Metal Performance Shaders: Apple Silicon GPU (MPS)")
        return True
    else:
        print("   💻 MPS not available, using CPU")
        return False


def main():
    # Check MPS availability
    print("\n2. Checking device...")
    use_mps = check_mps_availability()
    device = "mps" if use_mps else "cpu"
    
    # Create environment
    print("\n3. Creating Mario environment...")
    env = create_mario_env()
    print("   ✓ Mario environment ready")
    
    # Set up model paths
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Load existing model if available
    print("\n4. Setting up AI agent...")
    existing_model = models_dir / "mario_production_gpu.zip"
    
    if existing_model.exists():
        print(f"   📂 Loading existing model: {existing_model.name}")
        model = PPO.load(str(existing_model), env=env, device=device)
        print("   ✓ Model loaded successfully")
    else:
        print("   🆕 Creating new PPO agent")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=0,  # Reduce verbosity for cleaner output
            tensorboard_log="./logs/",
            device=device,
            batch_size=256 if use_mps else 64,
            n_steps=512 if use_mps else 256,
            learning_rate=0.0003,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            gamma=0.99
        )
        print("   ✓ New PPO agent created")
    
    print(f"   🔥 Device: {device.upper()}")
    print(f"   🎯 Target: 10,000 episodes")
    
    # Set up callback
    episode_callback = EpisodeCounterCallback(target_episodes=10000)
    
    # Start training
    print("\n🚀 Starting 10,000 Episode Training")
    print("-" * 55)
    
    start_time = time.time()
    
    try:
        # Suppress output during training for cleaner experience
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.learn(
                total_timesteps=10_000_000,  # Large number, but callback will stop at 10k episodes
                callback=episode_callback,
                progress_bar=False,  # We have custom progress
                reset_num_timesteps=False
            )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Calculate training time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n🎉 Training Complete!")
    print("=" * 55)
    print(f"   Episodes completed: {episode_callback.episode_count:,}")
    print(f"   Training time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print(f"   Device used: {device.upper()}")
    
    # Save the trained model
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = f"mario_10k_episodes_{timestamp}.zip"
    model_path = models_dir / model_name
    
    print(f"\n💾 Saving model...")
    model.save(str(model_path))
    print(f"   ✓ Model saved: {model_name}")
    
    # Also save as latest
    latest_path = models_dir / "mario_latest_10k.zip"
    model.save(str(latest_path))
    print(f"   ✓ Latest model saved: {latest_path.name}")
    
    env.close()
    print("\n🏁 Training session completed successfully!")
    

if __name__ == "__main__":
    main()