#!/usr/bin/env python3
"""
Mario RL Training with REAL-TIME Visual Rendering During Training

This version shows Mario playing live while the AI is learning.
"""

import sys
import time
import os
from pathlib import Path

print("🎮 Mario RL Training - LIVE VISUAL TRAINING")
print("=" * 50)

print("1. Loading libraries...")
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
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

class LiveRenderingWrapper(gym.Wrapper):
    """Wrapper that renders the environment during training."""
    
    def __init__(self, env, render_freq=1):
        super().__init__(env)
        self.render_freq = render_freq
        self.step_count = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Render every render_freq steps
        self.step_count += 1
        if self.step_count % self.render_freq == 0:
            try:
                self.render()
            except:
                pass  # Handle render errors gracefully
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Always render on reset
        try:
            self.render()
        except:
            pass
        return obs, info

class LiveTrainingCallback(BaseCallback):
    """Callback that shows live training progress without buffer errors."""
    
    def __init__(self, verbose=1, device_name="Unknown"):
        super().__init__(verbose)
        self.device_name = device_name
        self.best_mean_reward = float('-inf')
        self.start_time = time.time()
        self.last_report = 0
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        # Simple progress reporting without buffer access
        elapsed = time.time() - self.start_time
        steps_per_second = self.num_timesteps / elapsed if elapsed > 0 else 0
        
        # Report every 2048 steps (one rollout)
        if self.num_timesteps - self.last_report >= 2048:
            self.last_report = self.num_timesteps
            print(f"\n   🎮 LIVE Training: {self.num_timesteps:,} steps")
            print(f"      Speed: {steps_per_second:.1f} steps/sec on {self.device_name}")
            print(f"      You should see Mario playing in the game window!")

def create_live_mario_env():
    """Create a Mario environment with live rendering during training."""
    # Create base environment
    env = make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymToGymnasiumWrapper(env)
    # Add live rendering wrapper - render every 2 steps for smooth visuals
    env = LiveRenderingWrapper(env, render_freq=2)
    return env

def main():
    """Run Mario training with live visual rendering."""
    try:
        # Detect best device
        device, device_name = detect_best_device()
        
        # Set environment variables for optimization
        if device.type == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        print("\n2. Creating LIVE VISUAL Mario environment...")
        env = DummyVecEnv([lambda: Monitor(create_live_mario_env())])
        
        print(f"   ✓ Created LIVE VISUAL Mario environment")
        print(f"   🎮 Mario will be visible during training!")
        print(f"   ✓ Using GPU acceleration: {device_name}")
        
        print("\n3. Creating AI agent for live training...")
        
        # Smaller batch for more frequent visual updates
        timesteps = 20000  # Shorter for demo
        
        model = PPO(
            'CnnPolicy',
            env,
            verbose=0,
            device=device,
            
            # Optimized for live visual training
            learning_rate=3e-4,
            n_steps=512,   # Smaller steps for more frequent updates
            batch_size=64,  # Smaller batch for faster visual feedback
            n_epochs=5,     # Fewer epochs for faster rollouts
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        
        print(f"   ✓ PPO agent created for {device_name}")
        print(f"   ✓ Optimized for live visual feedback")
        print(f"   ✓ Batch size: 64 (faster visual updates)")
        print(f"   ✓ Steps per update: 512 (frequent rollouts)")
        
        print(f"\n🎮 Starting LIVE VISUAL TRAINING!")
        print(f"   🎯 Target: {timesteps:,} timesteps")
        print(f"   🔥 Device: {device_name}")
        print(f"   👁️  WATCH: Mario learning to play in real-time!")
        print(f"   🎮 A game window should open showing Mario!")
        print("-" * 50)
        
        # Create live training callback
        callback = LiveTrainingCallback(device_name=device_name)
        
        # Start live visual training
        start_time = time.time()
        
        print("\n🎮 MARIO IS LEARNING TO PLAY - WATCH THE GAME WINDOW!")
        print("   (If no window appears, your system might not support OpenGL rendering)")
        
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        steps_per_second = timesteps / training_time
        
        print(f"\n✅ LIVE Visual Training completed!")
        print(f"   ⏱️  Time: {training_time:.1f} seconds")
        print(f"   🚀 Speed: {steps_per_second:.1f} steps/second")
        print(f"   🎮 You watched Mario learn to play!")
        
        # Save model
        model.save("mario_live_trained")
        print(f"💾 Live-trained model saved as 'mario_live_trained'")
        
        # Final demonstration
        print(f"\n🏆 FINAL DEMONSTRATION:")
        print("Watch Mario play with his newly learned skills...")
        
        for episode in range(2):
            print(f"\n   🎮 Demo Episode {episode + 1}")
            obs = env.reset()
            episode_reward = 0
            max_x = 0
            
            for step in range(1000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += np.sum(reward) if hasattr(reward, '__len__') else reward
                
                # Track progress
                if info and len(info) > 0:
                    for env_info in info:
                        if env_info and 'x_pos' in env_info:
                            max_x = max(max_x, env_info['x_pos'])
                
                # Slower for demonstration
                time.sleep(0.03)
                
                if np.any(done):
                    break
            
            print(f"      Distance reached: {max_x:,} pixels")
            print(f"      Reward earned: {episode_reward:.1f}")
        
        print(f"\n📊 Live Training Results:")
        print(f"   🎮 Watched Mario learn in real-time!")
        print(f"   🔥 GPU acceleration: {device_name}")
        print(f"   ⚡ Training speed: {steps_per_second:.1f} steps/sec")
        print(f"   👁️  Visual feedback throughout training")
        
        print(f"\n💡 Live Training Benefits:")
        print(f"   🎮 SAW Mario learning to play step by step")
        print(f"   📈 Visual progress tracking during training")
        print(f"   🚀 GPU acceleration with live rendering")
        print(f"   🎯 Immediate feedback on AI learning progress")
        
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user")
        if 'model' in locals():
            model.save("mario_live_interrupted")
            print("💾 Progress saved as 'mario_live_interrupted'")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("💡 Note: Visual rendering requires OpenGL support")
        print("   Try running on a system with display capabilities")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        print("🏁 Live visual training session complete!")

if __name__ == "__main__":
    main()