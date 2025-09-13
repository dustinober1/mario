#!/usr/bin/env python3
"""
Basic Mario RL Training Example

This script demonstrates how to train a Mario RL agent using stable-baselines3 and PPO.
"""

import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace

from mario_rl.utils import setup_logging


class GymToGymnasiumWrapper(gym.Env):
    """Wrapper to make old gym environments compatible with gymnasium/stable-baselines3."""
    
    def __init__(self, env):
        super().__init__()
        self._env = env
        self.observation_space = env.observation_space
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
    """Create a Mario environment with proper compatibility."""
    env = make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymToGymnasiumWrapper(env)
    return env


def main():
    """Run basic training example."""
    # Setup logging
    logger = setup_logging(log_file="mario_training.log")
    logger.info("Starting Mario RL training example")
    
    print("🎮 Mario RL Training Example")
    print("=" * 40)
    
    try:
        # Create environment
        logger.info("Creating Mario environment...")
        print("1. Creating Mario environment...")
        env = DummyVecEnv([lambda: Monitor(create_mario_env())])
        print(f"   ✓ Environment ready for World 1-1")
        
        # Create agent
        logger.info("Creating PPO agent...")
        print("2. Setting up AI agent...")
        model = PPO(
            'CnnPolicy',
            env,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            device='auto'
        )
        print("   ✓ PPO agent created")
        
        # Train model
        logger.info("Starting training...")
        print("3. Training Mario...")
        print("   This will take a few minutes...")
        
        start_time = time.time()
        model.learn(total_timesteps=50000)
        training_time = time.time() - start_time
        
        print(f"   ✅ Training completed in {training_time:.1f} seconds")
        
        # Save final model
        logger.info("Saving trained model...")
        model.save("mario_trained_model")
        print("4. Model saved as 'mario_trained_model'")
        
        # Quick evaluation
        logger.info("Evaluating trained model...")
        print("5. Testing Mario's performance...")
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        max_x = 0
        
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if info and len(info) > 0:
                x_pos = info[0].get('x_pos', 0)
                max_x = max(max_x, x_pos)
            
            if done:
                break
        
        print(f"   📊 Results:")
        print(f"      Distance reached: {max_x:,} pixels")
        print(f"      Total reward: {total_reward:.1f}")
        print(f"      Steps taken: {steps}")
        
        if max_x > 1000:
            print("   🎉 Excellent! Mario learned to navigate well!")
        elif max_x > 500:
            print("   👍 Good! Mario made solid progress!")
        else:
            print("   📈 Mario learned the basics. Try longer training for better results!")
        
        logger.info("Training example completed successfully!")
        print("\n✅ Training example completed!")
        
    except Exception as e:
        logger.error(f"Training example failed: {e}")
        print(f"❌ Error: {e}")
        raise
    finally:
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    main()
