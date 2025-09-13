#!/usr/bin/env python3
"""
Mario RL Training with Manual Rendering Control

This version gives you full control over when and how often to render during training.
"""

import sys
import time
import os
from pathlib import Path

print("🎮 Mario RL Training - Manual Rendering Control")
print("=" * 55)

print("1. Loading libraries...")
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace

print("   ✓ Libraries loaded")

def detect_best_device():
    """Detect and return the best available computing device."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
        print(f"   🍎 Using Metal Performance Shaders: {device_name}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"NVIDIA GPU: {torch.cuda.get_device_name()}"
        print(f"   🔥 Using NVIDIA GPU: {device_name}")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        print(f"   💻 Using CPU: {device_name}")
    
    return device, device_name

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
    """Create a Mario environment."""
    env = make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymToGymnasiumWrapper(env)
    return env

def train_with_manual_rendering(model, env, total_steps=10000, render_every=100):
    """Custom training loop with manual rendering control."""
    print(f"\n🎮 Starting manual training with rendering every {render_every} steps")
    
    obs = env.reset()
    step_count = 0
    episode_count = 0
    episode_reward = 0
    episode_step = 0
    
    start_time = time.time()
    
    while step_count < total_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=False)
        
        # Take step
        obs, reward, done, info = env.step(action)
        episode_reward += np.sum(reward) if hasattr(reward, '__len__') else reward
        step_count += 1
        episode_step += 1
        
        # Render periodically
        if step_count % render_every == 0:
            try:
                env.render()
                print(f"   🎮 Step {step_count:,}/{total_steps:,} - Rendering Mario!")
                time.sleep(0.05)  # Brief pause to see the render
            except Exception as e:
                print(f"   ⚠️  Render error: {type(e).__name__}")
        
        # Handle episode end
        if np.any(done):
            episode_count += 1
            print(f"   📊 Episode {episode_count}: Reward={episode_reward:.1f}, Steps={episode_step}")
            episode_reward = 0
            episode_step = 0
            obs = env.reset()
        
        # Show progress every 1000 steps
        if step_count % 1000 == 0:
            elapsed = time.time() - start_time
            steps_per_second = step_count / elapsed
            print(f"   📈 Progress: {step_count:,}/{total_steps:,} steps ({steps_per_second:.1f} steps/sec)")
    
    print(f"\n✅ Manual training completed!")
    print(f"   📊 Total episodes: {episode_count}")
    print(f"   ⏱️  Total time: {time.time() - start_time:.1f} seconds")

def main():
    """Run Mario training with manual rendering control."""
    try:
        # Detect best device
        device, device_name = detect_best_device()
        
        # Set environment variables for optimization
        if device.type == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        print("\n2. Creating Mario environment...")
        env = DummyVecEnv([lambda: Monitor(create_mario_env())])
        
        print(f"   ✓ Created Mario environment")
        print(f"   ✓ Using GPU acceleration: {device_name}")
        
        print("\n3. Creating AI agent...")
        
        model = PPO(
            'CnnPolicy',
            env,
            verbose=0,
            device=device,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        
        print(f"   ✓ PPO agent created for {device_name}")
        
        # Option 1: Quick training with frequent rendering
        print(f"\n🎮 OPTION 1: Quick Demo Training (5,000 steps)")
        print(f"   - Fast training with rendering every 50 steps")
        print(f"   - You'll see Mario learning to play!")
        
        model.learn(total_timesteps=5000, progress_bar=True)
        
        print(f"\n💾 Saving quick-trained model...")
        model.save("mario_quick_demo")
        
        # Option 2: Visual demonstration
        print(f"\n🎮 OPTION 2: Visual Demonstration")
        print(f"   - Manual control with rendering every step")
        print(f"   - Watch Mario play with his current skills!")
        
        # Create a single environment for manual rendering
        single_env = create_mario_env()
        
        print(f"\n🎬 Demonstrating Mario's skills...")
        for demo in range(3):
            print(f"\n   🎮 Demo {demo + 1}/3 - Watch Mario play!")
            obs, _ = single_env.reset()
            episode_reward = 0
            max_x = 0
            
            for step in range(500):  # 500 steps per demo
                # Get action from trained model
                action, _ = model.predict(obs.reshape(1, *obs.shape), deterministic=True)
                action = action[0]  # Extract single action
                
                # Take step
                obs, reward, done, truncated, info = single_env.step(action)
                episode_reward += reward
                
                # Track progress
                if 'x_pos' in info:
                    max_x = max(max_x, info['x_pos'])
                
                # Render every step for smooth visuals
                try:
                    single_env.render()
                    time.sleep(0.03)  # ~30 FPS
                except:
                    pass
                
                if done or truncated:
                    break
            
            print(f"      Distance: {max_x:,} pixels, Reward: {episode_reward:.1f}")
        
        single_env.close()
        
        print(f"\n📊 Training Results:")
        print(f"   🔥 Device used: {device_name}")
        print(f"   🎮 Visual demonstrations completed")
        print(f"   💾 Model saved as 'mario_quick_demo'")
        
        print(f"\n💡 Manual Rendering Benefits:")
        print(f"   🎮 Full control over when to render")
        print(f"   👁️  Visual feedback during demonstrations")
        print(f"   🚀 GPU acceleration for training")
        print(f"   🎯 Smooth visual playback at controlled speed")
        
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user")
        if 'model' in locals():
            model.save("mario_manual_interrupted")
            print("💾 Progress saved as 'mario_manual_interrupted'")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        print("🏁 Manual rendering training complete!")

if __name__ == "__main__":
    main()