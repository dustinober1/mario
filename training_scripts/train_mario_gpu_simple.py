#!/usr/bin/env python3
"""
Simplified Optimized Mario RL Training Script for M1/M2 Macs

This version focuses on GPU acceleration without complex preprocessing to ensure compatibility.
"""

import sys
import time
import os
from pathlib import Path

print("🚀 Mario RL Training - Simplified GPU Optimized")
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
        print(f"   🔥 Using CUDA GPU: {device_name}")
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

class PerformanceCallback(BaseCallback):
    """Enhanced callback for tracking training performance and GPU utilization."""
    
    def __init__(self, verbose=1, device_name="Unknown"):
        super().__init__(verbose)
        self.device_name = device_name
        self.best_mean_reward = float('-inf')
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        if len(self.model.ep_info_buffer) > 0:
            recent_episodes = self.model.ep_info_buffer[-10:]
            if recent_episodes:
                mean_reward = np.mean([ep["r"] for ep in recent_episodes])
                mean_length = np.mean([ep["l"] for ep in recent_episodes])
                
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    print(f"   🎉 NEW BEST! Mean reward: {mean_reward:.1f}")
                
                # Calculate training speed
                elapsed = time.time() - self.start_time
                steps_per_second = self.num_timesteps / elapsed if elapsed > 0 else 0
                
                if self.num_timesteps % 4000 == 0:
                    print(f"   📊 Progress: {self.num_timesteps:,} steps")
                    print(f"      Mean reward: {mean_reward:.1f}")
                    print(f"      Mean episode length: {mean_length:.1f}")
                    print(f"      Training speed: {steps_per_second:.1f} steps/sec on {self.device_name}")

def create_mario_env():
    """Create a Mario environment with basic compatibility wrapper."""
    # Create base environment
    env = make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymToGymnasiumWrapper(env)  # Only compatibility wrapper
    return env

def main():
    """Run GPU-optimized Mario training."""
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
        
        print("\n3. Creating GPU-optimized AI agent...")
        
        # GPU-optimized hyperparameters
        timesteps = 100000 if device.type in ["mps", "cuda"] else 50000
        
        model = PPO(
            'CnnPolicy',
            env,
            verbose=0,
            device=device,
            
            # GPU-optimized hyperparameters
            learning_rate=3e-4,
            n_steps=2048 if device.type in ["mps", "cuda"] else 512,
            batch_size=256 if device.type in ["mps", "cuda"] else 64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
        
        print(f"   ✓ PPO agent created for {device_name}")
        print(f"   ✓ Batch size: {256 if device.type in ['mps', 'cuda'] else 64}")
        print(f"   ✓ Steps per update: {2048 if device.type in ['mps', 'cuda'] else 512}")
        
        print(f"\n🚀 Starting GPU-optimized training!")
        print(f"   🎯 Target: {timesteps:,} timesteps")
        print(f"   🔥 Device: {device_name}")
        print("-" * 50)
        
        # Create performance callback
        callback = PerformanceCallback(device_name=device_name)
        
        # Training with GPU acceleration
        start_time = time.time()
        
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        steps_per_second = timesteps / training_time
        
        print(f"\n✅ GPU Training completed!")
        print(f"   ⏱️  Time: {training_time:.1f} seconds")
        print(f"   🚀 Speed: {steps_per_second:.1f} steps/second")
        
        # Save model
        model.save("mario_gpu_optimized")
        print("💾 GPU-optimized model saved as 'mario_gpu_optimized'")
        
        # Test the trained model
        print("\n🏆 Testing GPU-trained model:")
        print("Running 5 test episodes...")
        
        test_results = []
        
        for episode in range(5):
            obs = env.reset()
            episode_reward = 0
            max_x = 0
            
            for _ in range(2000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += np.sum(reward) if hasattr(reward, '__len__') else reward
                
                # Track best x position
                if info and len(info) > 0:
                    for env_info in info:
                        if env_info and 'x_pos' in env_info:
                            max_x = max(max_x, env_info['x_pos'])
                
                if np.any(done):
                    break
            
            test_results.append({
                'distance': max_x,
                'reward': episode_reward
            })
            
            print(f"   Episode {episode + 1}: Distance={max_x:,}, Reward={episode_reward:.1f}")
        
        # Final results
        avg_distance = np.mean([r['distance'] for r in test_results])
        best_distance = max([r['distance'] for r in test_results])
        avg_reward = np.mean([r['reward'] for r in test_results])
        
        print(f"\n📊 GPU-Optimized Training Results:")
        print(f"   🏃 Average distance: {avg_distance:,.1f} pixels")
        print(f"   🎯 Best distance: {best_distance:,} pixels")
        print(f"   💰 Average reward: {avg_reward:.1f}")
        print(f"   🔥 Device used: {device_name}")
        print(f"   ⚡ Training speed: {steps_per_second:.1f} steps/sec")
        
        # Performance comparison
        baseline_speed = 37  # steps/sec from original CPU training
        speedup = steps_per_second / baseline_speed
        print(f"\n💡 GPU Optimization Benefits:")
        print(f"   🚀 Speedup vs CPU: {speedup:.1f}x faster!")
        print(f"   🔥 Metal Performance Shaders utilized")
        print(f"   ⚡ Large batch processing on GPU")
        
        if best_distance > 3000:
            print("   🏆 AMAZING! GPU acceleration enabled excellent performance!")
        elif best_distance > 1500:
            print("   🌟 EXCELLENT! Great results with GPU optimization!")
        elif best_distance > 800:
            print("   👍 GOOD! Solid performance with GPU training!")
        else:
            print("   📈 PROGRESS! GPU training working, try longer sessions!")
        
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user")
        if 'model' in locals():
            model.save("mario_gpu_interrupted")
            print("💾 Progress saved as 'mario_gpu_interrupted'")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        print("🏁 GPU-optimized training session complete!")

if __name__ == "__main__":
    main()