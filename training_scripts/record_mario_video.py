#!/usr/bin/env python3
"""
🎬 Mario Video Recorder

Record video of a trained Mario model for demonstration purposes.
Works with any saved Mario RL model (.zip file).

Usage:
    python training_scripts/record_mario_video.py MODEL_PATH [--output VIDEO_NAME]
"""

import os
import sys
import argparse
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("🎬 Mario Video Recorder")
print("=" * 30)

try:
    import cv2
    print("   ✓ OpenCV available for video recording")
except ImportError:
    print("   ❌ OpenCV not available - install with: pip install opencv-python")
    sys.exit(1)

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace

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

def record_mario_video(model_path, output_path, num_episodes=3, max_steps=1000):
    """Record video of Mario model playing"""
    
    print(f"📁 Loading model: {model_path}")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        print("   🍎 Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("   🔥 Using NVIDIA GPU")
    else:
        device = "cpu"
        print("   💻 Using CPU")
    
    # Load the trained model
    try:
        model = PPO.load(model_path, device=device)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return False
    
    # Create environment
    env = create_mario_env()
    print("   ✓ Mario environment ready")
    
    # Video setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frame_size = (256, 240)  # Mario screen size
    
    print(f"🎬 Recording video: {output_path}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Max steps per episode: {max_steps}")
    
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    total_frames = 0
    
    for episode in range(num_episodes):
        print(f"   📹 Recording episode {episode + 1}/{num_episodes}...")
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_frames = 0
        max_distance = 0
        
        for step in range(max_steps):
            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_frames += 1
            
            # Track progress
            if 'x_pos' in info:
                max_distance = max(max_distance, info['x_pos'])
            
            # Get frame and write to video
            try:
                frame = env.render()
                if frame is not None and frame.size > 0:
                    # Ensure frame is the right size
                    if frame.shape[:2] != (240, 256):
                        frame = cv2.resize(frame, frame_size)
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                    total_frames += 1
            except Exception as e:
                # Skip problematic frames
                pass
            
            if done or truncated:
                print(f"      Episode {episode + 1}: {max_distance:,} pixels, {episode_reward:.1f} reward, {episode_frames} frames")
                break
    
    out.release()
    env.close()
    
    print(f"✅ Video recording complete!")
    print(f"   📁 Saved: {output_path}")
    print(f"   🎞️  Total frames: {total_frames}")
    print(f"   ⏱️  Duration: ~{total_frames/fps:.1f} seconds")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Record Mario RL model video')
    parser.add_argument('model_path', help='Path to the trained model (.zip file)')
    parser.add_argument('--output', default='mario_demo.mp4', help='Output video filename')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to record')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create videos directory
    Path("videos").mkdir(exist_ok=True)
    
    # Ensure output path is in videos directory
    output_path = f"videos/{args.output}" if not args.output.startswith("videos/") else args.output
    
    # Record the video
    success = record_mario_video(
        model_path=args.model_path,
        output_path=output_path,
        num_episodes=args.episodes,
        max_steps=args.max_steps
    )
    
    if success:
        print("\n🎉 Video recording successful!")
        print(f"🎬 Play with: open {output_path}")
    else:
        print("\n❌ Video recording failed")
        sys.exit(1)

if __name__ == "__main__":
    main()