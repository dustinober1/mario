#!/usr/bin/env python3
"""
🎮 Mario RL - 100K Episode Training

Simple, reliable extended training based on the proven production script.
Features:
- 100,000 episodes (≈40 million timesteps)
- Checkpointing every 1 million steps
- Resume capability
- GPU acceleration (MPS on Apple Silicon)
- Final model testing and saving

Usage:
    python training_scripts/train_mario_100k.py [--resume MODEL_PATH]
"""

import sys
import time
import os
import warnings
import argparse
from pathlib import Path
from datetime import datetime

# Suppress gym deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("🍄 Mario RL - 100K Episode Training")
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

class ExtendedTrainingCallback(BaseCallback):
    """Extended callback for 100K episode training with checkpointing"""
    
    def __init__(self, checkpoint_freq=1000000, verbose=1, device_name="Unknown"):
        super().__init__(verbose)
        self.device_name = device_name
        self.checkpoint_freq = checkpoint_freq
        self.best_mean_reward = float('-inf')
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_count = 0
        self.last_checkpoint = 0
        
        # Create directories
        Path("models/extended").mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"📊 Extended training initialized")
        print(f"💾 Checkpoints every {checkpoint_freq:,} steps")
        print()
    
    def _on_step(self) -> bool:
        return True
        
    def _on_rollout_end(self) -> None:
        # Extended progress reporting
        elapsed = time.time() - self.start_time
        steps_per_second = self.num_timesteps / elapsed if elapsed > 0 else 0
        
        # Calculate progress percentage for 100K episodes (≈40M steps)
        progress_pct = (self.num_timesteps / 40000000) * 100
        
        # Save checkpoint if needed
        if self.num_timesteps - self.last_checkpoint >= self.checkpoint_freq:
            self.last_checkpoint = self.num_timesteps
            checkpoint_path = f"models/extended/mario_checkpoint_{self.num_timesteps}_{self.timestamp}.zip"
            self.model.save(checkpoint_path)
            
            elapsed_hours = elapsed / 3600
            print()
            print(f"💾 CHECKPOINT SAVED at step {self.num_timesteps:,}")
            print(f"   📁 {checkpoint_path}")
            print(f"   ⏰ Training time: {elapsed_hours:.1f} hours")
            print(f"   🚀 Speed: {steps_per_second:.1f} steps/sec")
            print()
        
        # Progress report every 10,000 steps
        if self.num_timesteps % 10000 == 0:
            print(f"📈 Step {self.num_timesteps:,} | "
                  f"Progress: {progress_pct:.1f}% | "
                  f"⏱️  {elapsed/60:.0f}min | "
                  f"🚀 {steps_per_second:.1f} steps/sec | "
                  f"Device: {self.device_name}")

def create_mario_env():
    """Create a Mario environment."""
    env = make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GymToGymnasiumWrapper(env)
    return env

def main():
    """Run extended 100K episode Mario RL training."""
    parser = argparse.ArgumentParser(description='100K Episode Mario RL Training')
    parser.add_argument('--resume', type=str, help='Path to model to resume training from')
    parser.add_argument('--timesteps', type=int, default=40000000, help='Total timesteps (default: 40M)')
    parser.add_argument('--checkpoint-freq', type=int, default=1000000, help='Checkpoint frequency (default: 1M)')
    args = parser.parse_args()
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Target Timesteps: {args.timesteps:,}")
    print(f"   Checkpoint Frequency: {args.checkpoint_freq:,} steps")
    print(f"   Resume from: {args.resume if args.resume else 'New training'}")
    print(f"   Estimated Episodes: {args.timesteps // 400:,}")
    print()
    
    try:
        # Detect best device
        device, device_name = detect_best_device()
        
        # Set environment variables for optimization
        if device.type == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        print("2. Creating Mario environment...")
        env = DummyVecEnv([lambda: Monitor(create_mario_env())])
        print("   ✓ Mario environment ready")
        
        # Configure PPO for extended training (same config as working production)
        print("3. Setting up PPO model...")
        ppo_config = {
            'policy': 'CnnPolicy',
            'env': env,
            'learning_rate': 2.5e-4,
            'n_steps': 2048,        # Large batch for GPU efficiency
            'batch_size': 256,      # GPU-optimized batch size
            'n_epochs': 4,          # Multiple epochs for stability
            'gamma': 0.9,           # Discount factor
            'gae_lambda': 0.95,     # GAE lambda
            'clip_range': 0.1,      # PPO clipping range
            'ent_coef': 0.01,       # Entropy coefficient for exploration
            'vf_coef': 0.5,         # Value function coefficient
            'max_grad_norm': 0.5,   # Gradient clipping
            'verbose': 0,           # Quiet - callback handles logging
            'device': device
        }
        
        # Create or load model
        if args.resume and os.path.exists(args.resume):
            print(f"🔄 Loading model from: {args.resume}")
            model = PPO.load(args.resume, env=env, device=device)
            print("   ✓ Model loaded successfully")
        else:
            print("🆕 Creating new PPO model...")
            model = PPO(**ppo_config)
            print("   ✓ New model initialized")
        
        # Setup extended training callback
        callback = ExtendedTrainingCallback(
            checkpoint_freq=args.checkpoint_freq,
            device_name=device_name,
            verbose=1
        )
        
        print(f"4. Starting extended training...")
        print(f"   🎯 Target timesteps: {args.timesteps:,}")
        print(f"   ⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        print()
        
        start_time = time.time()
        
        # Start the extended training
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False,  # Use our custom progress tracking
            reset_num_timesteps=False if args.resume else True
        )
        
        training_time = time.time() - start_time
        
        print()
        print("🎉 EXTENDED TRAINING COMPLETED!")
        print("=" * 50)
        print(f"⏰ Total training time: {training_time/3600:.1f} hours")
        print(f"📊 Total timesteps: {args.timesteps:,}")
        print(f"🚀 Average speed: {args.timesteps/training_time:.0f} steps/sec")
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = f"models/extended/mario_final_100k_{timestamp}.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        print()
        print("✅ 100K EPISODE TRAINING COMPLETE!")
        print(f"📁 Final model: {final_model_path}")
        print(f"📂 Checkpoints: models/extended/")
        
    except KeyboardInterrupt:
        print()
        print("⏸️  Training interrupted by user")
        
        # Save interrupted state
        interrupted_time = time.time() - start_time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interrupted_path = f"models/extended/mario_interrupted_{timestamp}.zip"
        model.save(interrupted_path)
        
        print(f"💾 Progress saved: {interrupted_path}")
        print(f"⏰ Training time: {interrupted_time/3600:.1f} hours")
        
    except Exception as e:
        print()
        print(f"❌ Training failed: {str(e)}")
        
        # Try to save current state
        try:
            error_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_path = f"models/extended/mario_error_save_{error_timestamp}.zip"
            model.save(error_path)
            print(f"💾 Emergency save: {error_path}")
        except:
            print("❌ Could not save model state")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        try:
            env.close()
        except:
            pass
        print("🏁 Session ended")

if __name__ == "__main__":
    main()