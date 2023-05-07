import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros import make
from nes_py.wrappers import JoypadSpace

# Create the custom Mario environment
def create_mario_env():
    env = make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

env = create_mario_env()

# Vectorize the environment
vec_env = DummyVecEnv([lambda: env])

# Initialize the RL model (Proximal Policy Optimization in this case)
model = PPO("CnnPolicy", vec_env, verbose=3)

# Train the model
training_timesteps = 200000
model.learn(total_timesteps=training_timesteps)

# Save the trained model
model.save("ppo_mario")

# Load the trained model for evaluation
model = PPO.load("ppo_mario")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")