#!/usr/bin/env python3
"""
Fixed Optimized Mario RL Training Script for M1/M2 Macs and GPU Acceleration

This script leverages:
- Metal Performance Shaders (MPS) for M1/M2 Macs
- CUDA for NVIDIA GPUs
- Optimized environment preprocessing
- Single environment with optimizations (avoiding multiprocessing issues)
- Enhanced hyperparameters for GPU training

Usage:
    python train_mario_optimized_fixed.py

The script automatically detects the best available device and optimizes training accordingly.
"""

import sys
import time
import os
from pathlib import Path

print("🚀 Mario RL Training - GPU Optimized (Fixed)")
print("=" * 45)

print("1. Loading optimized libraries...")
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
import cv2

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

class OptimizedFrameStack(gym.Wrapper):
    """Stack multiple frames and convert to grayscale for better temporal understanding."""
    
    def __init__(self, env, num_stack=4, resize_shape=(84, 84)):
        super().__init__(env)
        self.num_stack = num_stack
        self.resize_shape = resize_shape
        self.frames = []
        
        # Update observation space for stacked grayscale frames
        # Format: (height, width, channels) for CNN processing
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(resize_shape[1], resize_shape[0], num_stack),  # (height, width, channels)
            dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Convert to grayscale and resize
        frame = self._process_frame(obs)
        # Initialize stack with the same frame
        self.frames = [frame] * self.num_stack
        return self._get_stacked_frames(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self._process_frame(obs)
        self.frames.append(frame)
        if len(self.frames) > self.num_stack:
            self.frames.pop(0)
        return self._get_stacked_frames(), reward, terminated, truncated, info
    
    def _process_frame(self, frame):
        """Convert frame to grayscale and resize."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to target shape
        resized = cv2.resize(gray, self.resize_shape, interpolation=cv2.INTER_AREA)
        return resized
    
    def _get_stacked_frames(self):
        """Stack frames along the channel dimension."""
        return np.stack(self.frames, axis=-1)

class FrameSkipWrapper(gym.Wrapper):
    """Skip frames to reduce computational load and improve training speed."""
    
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    
    def step(self, action):
        total_reward = 0.0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class NormalizeWrapper(gym.ObservationWrapper):
    """Normalize observations to [0, 1] range for better neural network training."""
    
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=old_shape, dtype=np.float32
        )
    
    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

class PerformanceCallback(BaseCallback):
    """Enhanced callback for tracking training performance and GPU utilization."""
    
    def __init__(self, verbose=1, device_name="Unknown"):
        super().__init__(verbose)
        self.device_name = device_name
        self.episode_count = 0
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

def create_optimized_mario_env():
    """Create an optimized Mario environment with preprocessing."""
    # Create base environment
    env = make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Apply wrappers in correct order: Gym->Gymnasium first, then other optimizations
    env = GymToGymnasiumWrapper(env)     # Compatibility wrapper FIRST
    env = FrameSkipWrapper(env, skip=4)  # Skip frames for speed
    env = OptimizedFrameStack(env, num_stack=4, resize_shape=(84, 84))  # Stack frames
    env = NormalizeWrapper(env)          # Normalize for better training
    
    return env

def main():
    """Run optimized Mario training with GPU acceleration."""
    try:
        # Detect best device
        device, device_name = detect_best_device()
        
        # Set environment variables for optimization
        if device.type == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        print("\n2. Creating optimized Mario environment...")
        
        # Use single environment to avoid multiprocessing issues for now
        env = DummyVecEnv([lambda: Monitor(create_optimized_mario_env())])
        
        print(f"   ✓ Created optimized environment")
        print(f"   ✓ Optimized preprocessing pipeline:")
        print(f"      - Frame skipping (4x speedup)")
        print(f"      - Frame stacking (temporal awareness)")
        print(f"      - Grayscale conversion (3x memory reduction)")
        print(f"      - Resizing to 84x84 (faster processing)")
        print(f"      - Normalization (better training)")
        
        print("\n3. Creating optimized AI agent...")
        
        # Optimized hyperparameters for GPU training
        timesteps = 100000 if device.type in ["mps", "cuda"] else 50000
        
        model = PPO(
            'CnnPolicy',
            env,
            verbose=0,  # Reduce logging for performance
            device=device,
            
            # Optimized hyperparameters for GPU
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
            
            # Policy network optimizations
            policy_kwargs=dict(
                features_extractor_kwargs=dict(features_dim=512),
                net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Use dict instead of list
                activation_fn=torch.nn.ReLU,
                normalize_images=False  # We already normalize
            )
        )
        
        print(f"   ✓ PPO agent optimized for {device_name}")
        print(f"   ✓ Batch size: {256 if device.type in ['mps', 'cuda'] else 64}")
        print(f"   ✓ Steps per update: {2048 if device.type in ['mps', 'cuda'] else 512}")
        
        print(f"\n🚀 Starting optimized training!")
        print(f"   🎯 Target: {timesteps:,} timesteps")
        print(f"   🔥 Device: {device_name}")
        print(f"   ⚡ Optimizations: Frame skip, stacking, GPU acceleration")
        print("-" * 45)
        
        # Create performance callback
        callback = PerformanceCallback(device_name=device_name)
        
        # Training with optimization
        start_time = time.time()
        
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        steps_per_second = timesteps / training_time
        
        print(f"\n✅ Training completed!")
        print(f"   ⏱️  Time: {training_time:.1f} seconds")
        print(f"   🚀 Speed: {steps_per_second:.1f} steps/second")
        
        # Save optimized model
        model.save("mario_optimized_model")
        print("💾 Optimized model saved as 'mario_optimized_model'")
        
        # Enhanced evaluation
        print("\n🏆 Performance Evaluation:")
        print("Running 5 test episodes on optimized model...")
        
        test_results = []
        
        for episode in range(5):
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            max_x = 0
            
            for _ in range(2000):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += np.sum(reward) if hasattr(reward, '__len__') else reward
                episode_steps += 1
                
                # Track best x position
                if info and len(info) > 0:
                    for env_info in info:
                        if env_info and 'x_pos' in env_info:
                            max_x = max(max_x, env_info['x_pos'])
                
                if np.any(done):
                    break
            
            test_results.append({
                'distance': max_x,
                'reward': episode_reward,
                'steps': episode_steps
            })
            
            print(f"   Episode {episode + 1}: Distance={max_x:,}, Reward={episode_reward:.1f}")
        
        # Final statistics
        avg_distance = np.mean([r['distance'] for r in test_results])
        best_distance = max([r['distance'] for r in test_results])
        avg_reward = np.mean([r['reward'] for r in test_results])
        
        print(f"\n📊 Optimized Training Results:")
        print(f"   🏃 Average distance: {avg_distance:,.1f} pixels")
        print(f"   🎯 Best distance: {best_distance:,} pixels")
        print(f"   💰 Average reward: {avg_reward:.1f}")
        print(f"   🔥 Device utilized: {device_name}")
        print(f"   ⚡ Training speed: {steps_per_second:.1f} steps/sec")
        
        # Performance assessment
        if best_distance > 3000:
            print("   🏆 AMAZING! GPU optimization enabled excellent performance!")
        elif best_distance > 2000:
            print("   🌟 EXCELLENT! Great results with optimized training!")
        elif best_distance > 1000:
            print("   👍 GOOD! Solid performance improvement!")
        else:
            print("   📈 PROGRESS! Optimization working, try longer training!")
        
        print(f"\n💡 Optimization Benefits:")
        print(f"   - Frame skipping speedup (4x faster)")
        print(f"   - Memory reduction with grayscale (3x less)")
        print(f"   - GPU acceleration with {device_name}")
        print(f"   - Optimized batch processing")
        print(f"   - Enhanced neural network architecture")
        
        # Performance comparison
        baseline_speed = 37  # steps/sec from original CPU training
        speedup = steps_per_second / baseline_speed
        print(f"   - Speedup vs CPU baseline: {speedup:.1f}x faster!")
        
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user")
        if 'model' in locals():
            model.save("mario_optimized_interrupted")
            print("💾 Progress saved as 'mario_optimized_interrupted'")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        print("🏁 Optimized training session complete!")

if __name__ == "__main__":
    main()