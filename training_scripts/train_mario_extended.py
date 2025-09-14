#!/usr/bin/env python3
"""
🎮 Mario RL - Extended Training with Video Recording
====================================================

Long-duration training (100,000 episodes) with:
- Video recording of test episodes
- Model checkpointing every 10,000 steps
- Comprehensive logging and progress tracking
- Resume capability from saved checkpoints
- GPU acceleration (MPS on Apple Silicon)

Usage:
    python training_scripts/train_mario_extended.py [--resume MODEL_PATH]
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

class MarioVideoWrapper(gym.Wrapper):
    """Wrapper to ensure Mario environment works with video recording"""
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, **kwargs):
        """Reset environment and return observation in correct format"""
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """Take environment step and ensure 5-value return"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

def create_mario_env(render_mode=None):
    """Create a Mario environment with proper wrappers"""
    # Create base environment (older gym-super-mario-bros doesn't use render_mode in make())
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    
    # Apply action space wrapper
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Apply custom wrapper for video compatibility
    env = MarioVideoWrapper(env)
    
    # Add monitor for episode statistics
    env = Monitor(env)
    
    return env

class ExtendedTrainingCallback:
    """Custom callback for extended training with enhanced logging"""
    
    def __init__(self, save_path="models/"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        self.start_time = time.time()
        self.episode_count = 0
        self.best_reward = -float('inf')
        
    def on_step(self, locals_, globals_):
        """Called at each training step"""
        self.episode_count += 1
        
        # Log progress every 1000 episodes
        if self.episode_count % 1000 == 0:
            elapsed_time = time.time() - self.start_time
            episodes_per_min = self.episode_count / (elapsed_time / 60)
            
            print(f"📊 Episode {self.episode_count:,} | "
                  f"⏱️  {elapsed_time/60:.1f}min | "
                  f"🚀 {episodes_per_min:.1f} eps/min")
        
        return True

def setup_directories():
    """Create necessary directories for training artifacts"""
    directories = [
        "models/checkpoints",
        "models/final", 
        "videos/training",
        "videos/evaluation",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Created training directories")

def main():
    parser = argparse.ArgumentParser(description='Extended Mario RL Training')
    parser.add_argument('--resume', type=str, help='Path to model to resume training from')
    parser.add_argument('--total-episodes', type=int, default=100000, 
                       help='Total training episodes (default: 100000)')
    parser.add_argument('--checkpoint-freq', type=int, default=10000,
                       help='Save checkpoint every N episodes (default: 10000)')
    args = parser.parse_args()
    
    print("🍄 Mario RL - Extended Training")
    print("=" * 50)
    print(f"🎯 Target Episodes: {args.total_episodes:,}")
    print(f"💾 Checkpoint Frequency: {args.checkpoint_freq:,} episodes")
    print(f"🎬 Video Recording: Enabled")
    print(f"🧠 GPU Acceleration: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print()
    
    # Setup directories
    setup_directories()
    
    # Device configuration for optimal performance
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using Apple Silicon GPU (MPS) acceleration")
    else:
        device = "cpu"
        print("💻 Using CPU (consider GPU for faster training)")
    
    # Create timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create training environment
    print("🎮 Creating Mario training environment...")
    env = make_vec_env(
        lambda: create_mario_env(),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    # Create evaluation environment with video recording
    print("📹 Setting up video recording environment...")
    eval_env = make_vec_env(
        lambda: create_mario_env(),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    
    # Wrap evaluation environment for video recording
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder=f"videos/evaluation/mario_training_{timestamp}",
        record_video_trigger=lambda x: x % 5000 == 0,  # Record every 5000 steps
        video_length=1000,  # Record 1000 steps per video
        name_prefix="mario_eval"
    )
    
    # Configure PPO with optimized hyperparameters for long training
    model_config = {
        'policy': 'CnnPolicy',
        'env': env,
        'learning_rate': 2.5e-4,
        'n_steps': 2048,         # Larger batch for stability
        'batch_size': 256,       # GPU-optimized batch size
        'n_epochs': 4,           # Multiple passes through data
        'gamma': 0.9,            # Discount factor
        'gae_lambda': 0.95,      # GAE parameter
        'clip_range': 0.1,       # PPO clipping
        'ent_coef': 0.01,        # Entropy coefficient for exploration
        'vf_coef': 0.5,          # Value function coefficient
        'max_grad_norm': 0.5,    # Gradient clipping
        'verbose': 1,
        'device': device,
        'tensorboard_log': f"logs/mario_training_{timestamp}"
    }
    
    # Create or load model
    if args.resume and os.path.exists(args.resume):
        print(f"🔄 Resuming training from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
        # Update tensorboard log for continued training
        model.tensorboard_log = f"logs/mario_training_{timestamp}_continued"
    else:
        print("🆕 Creating new PPO model...")
        model = PPO(**model_config)
    
    # Setup callbacks for checkpointing and evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=f"models/checkpoints/mario_{timestamp}",
        name_prefix="mario_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/final/mario_best_{timestamp}",
        log_path=f"logs/eval_{timestamp}",
        eval_freq=5000,        # Evaluate every 5000 steps
        n_eval_episodes=5,     # Number of episodes to evaluate
        deterministic=True,
        render=False
    )
    
    # Training configuration
    total_timesteps = args.total_episodes * 1000  # Approximate timesteps per episode
    
    print(f"🏃 Starting training for {total_timesteps:,} timesteps...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Start training with callbacks
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False if args.resume else True
        )
        
        # Save final model
        final_model_path = f"models/final/mario_final_{timestamp}.zip"
        model.save(final_model_path)
        
        print()
        print("🎉 Training completed successfully!")
        print(f"💾 Final model saved: {final_model_path}")
        print(f"📹 Videos saved in: videos/evaluation/mario_training_{timestamp}/")
        print(f"📊 Logs available in: logs/mario_training_{timestamp}/")
        print()
        
        # Test the trained model
        print("🎮 Testing trained model...")
        test_env = create_mario_env()
        obs, _ = test_env.reset()
        
        total_reward = 0
        steps = 0
        
        for _ in range(1000):  # Test for 1000 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"🏆 Test Results: {total_reward:.1f} reward in {steps} steps")
        
        # Save test video
        print("🎬 Saving test demonstration video...")
        test_video_env = VecVideoRecorder(
            DummyVecEnv([lambda: create_mario_env()]),
            video_folder=f"videos/training/mario_final_test_{timestamp}",
            record_video_trigger=lambda x: True,
            video_length=1000,
            name_prefix="mario_final_test"
        )
        
        obs = test_video_env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_video_env.step(action)
            if terminated.any():
                break
        
        test_video_env.close()
        test_env.close()
        
        print(f"✅ Test video saved in: videos/training/mario_final_test_{timestamp}/")
        
    except KeyboardInterrupt:
        print()
        print("⏸️  Training interrupted by user")
        
        # Save interrupted model
        interrupted_model_path = f"models/final/mario_interrupted_{timestamp}.zip"
        model.save(interrupted_model_path)
        print(f"💾 Progress saved: {interrupted_model_path}")
        
    except Exception as e:
        print()
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)
        
    finally:
        # Cleanup
        env.close()
        eval_env.close()
        
        print()
        print("🏁 Training session complete!")
        print(f"⏰ Session duration: {(time.time() - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    start_time = time.time()
    main()