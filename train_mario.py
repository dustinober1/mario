#!/usr/bin/env python3
"""
Mario RL Training Script

Train an AI agent to play Super Mario Bros using PPO (Proximal Policy Optimization).
This script demonstrates deep reinforcement learning applied to the classic Mario game.

Usage:
    python train_mario.py

The script will train for 20,000 timesteps with live progress updates and save
the final trained model as 'mario_trained_model.zip'.
"""

import sys
import time
from pathlib import Path

print("🎮 Mario RL Training")
print("=" * 30)

print("1. Loading libraries...")
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace
print("   ✓ Libraries loaded")

class GymToGymnasiumWrapper(gym.Env):
    """Proper gymnasium wrapper for old gym environments"""
    
    def __init__(self, env):
        super().__init__()
        self._env = env
        
        # Convert spaces to gymnasium format
        import gymnasium.spaces as gym_spaces
        
        # Convert observation space
        if hasattr(env.observation_space, 'shape'):
            self.observation_space = gym_spaces.Box(
                low=env.observation_space.low,
                high=env.observation_space.high,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype
            )
        else:
            self.observation_space = env.observation_space
            
        # Convert action space
        if hasattr(env.action_space, 'n'):
            self.action_space = gym_spaces.Discrete(env.action_space.n)
        else:
            self.action_space = env.action_space
        
    def reset(self, **kwargs):
        obs = self._env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        # Convert to new gymnasium format: obs, reward, terminated, truncated, info
        return obs, reward, done, False, info
    
    def render(self):
        return self._env.render()
    
    def close(self):
        return self._env.close()

def create_mario_env():
    """Create Mario environment with proper gymnasium compatibility."""
    print("\n2. Creating Mario environment...")
    
    # Create the base environment
    env = make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Wrap with gymnasium compatibility
    env = GymToGymnasiumWrapper(env)
    
    print(f"   ✓ Environment created with {len(SIMPLE_MOVEMENT)} actions")
    print(f"   ✓ Training on World 1-1 (the first level)")
    print(f"   ✓ Observation space: {env.observation_space.shape}")
    
    return env

def main():
    """Run Mario training with live progress updates."""
    try:
        # Create vectorized environment
        env = DummyVecEnv([lambda: Monitor(create_mario_env())])
        
        print("\n3. Creating AI agent...")
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
        print("   ✓ PPO agent ready to learn!")
        
        print("\n🚀 Training begins!")
        print("   🎯 Goal: Teach Mario to complete World 1-1")
        print("   📚 Method: Proximal Policy Optimization (PPO)")
        print("   🎮 Actions: 7 possible moves (right, jump, etc.)")
        print("   ⏱️  Training: 20,000 timesteps with live updates")
        print("-" * 30)
        
        # Training configuration
        total_timesteps = 20000  # Shorter for demo purposes
        update_interval = 2000   # Show progress every 2000 steps
        
        start_time = time.time()
        best_distance = 0
        
        for i in range(0, total_timesteps, update_interval):
            current_steps = min(update_interval, total_timesteps - i)
            
            batch_num = i // update_interval + 1
            total_batches = total_timesteps // update_interval
            
            print(f"\n📈 Training Batch {batch_num}/{total_batches}")
            print(f"   Steps: {i:,} → {i + current_steps:,}")
            
            # Train the model
            batch_start = time.time()
            model.learn(total_timesteps=current_steps, reset_num_timesteps=False)
            batch_time = time.time() - batch_start
            
            print(f"   ⏱️  Batch training time: {batch_time:.1f} seconds")
            
            # Test current performance
            print("   🧪 Testing current skill level...")
            obs = env.reset()
            episode_reward = 0
            episode_steps = 0
            max_x_position = 0
            
            # Run a test episode
            for _ in range(750):  # Max steps for test
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_steps += 1
                
                # Track Mario's progress
                if info and len(info) > 0:
                    x_pos = info[0].get('x_pos', 0)
                    max_x_position = max(max_x_position, x_pos)
                
                if done:
                    break
            
            # Update best performance
            if max_x_position > best_distance:
                best_distance = max_x_position
                improvement = "🆕 NEW RECORD!"
            else:
                improvement = ""
            
            # Display results
            elapsed_total = time.time() - start_time
            progress = ((i + current_steps) / total_timesteps) * 100
            
            print(f"   📊 Results:")
            print(f"      Distance: {max_x_position:,} pixels {improvement}")
            print(f"      Reward: {episode_reward:.1f}")
            print(f"      Episode length: {episode_steps} steps")
            print(f"      Best distance so far: {best_distance:,}")
            print(f"   📈 Overall progress: {progress:.1f}%")
            print(f"   ⏱️  Total time: {elapsed_total:.1f} seconds")
            
            # Provide encouraging feedback based on performance
            if max_x_position > 2500:
                print("   🎉 INCREDIBLE! Mario is almost completing the level!")
            elif max_x_position > 1500:
                print("   🌟 EXCELLENT! Mario is navigating complex obstacles!")
            elif max_x_position > 1000:
                print("   🚀 GREAT! Mario is making solid progress!")
            elif max_x_position > 500:
                print("   👍 GOOD! Mario is learning to move forward!")
            elif max_x_position > 200:
                print("   📈 PROGRESS! Mario understands basic movement!")
            else:
                print("   🤔 Mario is still learning the fundamentals...")
            
            # Save checkpoint
            model.save(f"mario_checkpoint_{i + current_steps}")
            print(f"   💾 Checkpoint saved")
        
        print("\n" + "="*50)
        print("🏆 TRAINING COMPLETED!")
        print("="*50)
        
        # Save final model
        model.save("mario_trained_model")
        print("💾 Final model saved as 'mario_trained_model'")
        
        # Comprehensive final evaluation
        print("\n🎯 FINAL EVALUATION:")
        print("Running 3 test episodes to assess Mario's skills...")
        
        test_results = []
        
        for test_num in range(3):
            print(f"\n   Test Episode {test_num + 1}:")
            obs = env.reset()
            total_reward = 0
            steps = 0
            max_x = 0
            
            for _ in range(1500):  # Longer test
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                steps += 1
                
                if info and len(info) > 0:
                    x_pos = info[0].get('x_pos', 0)
                    max_x = max(max_x, x_pos)
                
                if done:
                    break
            
            test_results.append({
                'distance': max_x,
                'reward': total_reward,
                'steps': steps
            })
            
            print(f"      Distance: {max_x:,} pixels")
            print(f"      Reward: {total_reward:.1f}")
            print(f"      Steps: {steps}")
        
        # Calculate final statistics
        avg_distance = np.mean([r['distance'] for r in test_results])
        best_distance_final = max([r['distance'] for r in test_results])
        avg_reward = np.mean([r['reward'] for r in test_results])
        
        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Average distance: {avg_distance:,.1f} pixels")
        print(f"   Best distance: {best_distance_final:,} pixels")
        print(f"   Average reward: {avg_reward:.1f}")
        
        # Final assessment
        print(f"\n🎖️  PERFORMANCE ASSESSMENT:")
        if best_distance_final > 3000:
            print("   🏆 LEGENDARY! Mario mastered most of the level!")
            print("   🌟 Your AI agent is highly skilled!")
        elif best_distance_final > 2000:
            print("   🥇 EXCELLENT! Mario navigated far into the level!")
            print("   👏 Great training results!")
        elif best_distance_final > 1200:
            print("   🥈 VERY GOOD! Mario learned solid gameplay!")
            print("   📈 Impressive progress!")
        elif best_distance_final > 600:
            print("   🥉 GOOD! Mario grasped the basics well!")
            print("   👍 Solid foundation established!")
        elif best_distance_final > 300:
            print("   📚 LEARNING! Mario understood basic controls!")
            print("   🎯 Ready for advanced training!")
        else:
            print("   🔰 BEGINNER! Mario learned fundamental movement!")
            print("   💪 More training will improve performance!")
        
        training_time = time.time() - start_time
        print(f"\n⏱️  Total training time: {training_time:.1f} seconds")
        print(f"🎮 Mario is now trained and ready to play!")
        
    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user")
        if 'model' in locals():
            model.save("mario_interrupted")
            print("💾 Progress saved as 'mario_interrupted'")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()
        print("\n🏁 Training session ended!")

if __name__ == "__main__":
    main()