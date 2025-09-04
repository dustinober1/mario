"""Comprehensive tests for Mario environment functionality."""

import pytest
import numpy as np
import gym
from unittest.mock import Mock, patch, MagicMock
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mario_rl.environments import MarioEnvironment


class TestMarioEnvironment:
    """Test Mario environment functionality."""

    def test_environment_initialization(self):
        """Test environment initialization with different parameters."""
        # Test default initialization
        env = MarioEnvironment()
        assert env.level == '1-1'
        assert env.movement == env.movement  # Should be SIMPLE_MOVEMENT
        
        # Test custom initialization
        env = MarioEnvironment(level='2-1', movement_type='complex')
        assert env.level == '2-1'
        assert env.movement == env.movement  # Should be COMPLEX_MOVEMENT

    def test_environment_initialization_invalid_level(self):
        """Test environment initialization with invalid level."""
        # This should not raise an error during initialization
        # Validation happens during environment creation
        env = MarioEnvironment(level='invalid-level')
        assert env.level == 'invalid-level'

    def test_environment_initialization_invalid_movement(self):
        """Test environment initialization with invalid movement type."""
        # Should default to simple movement
        env = MarioEnvironment(movement_type='invalid')
        assert env.movement == env.movement  # Should be SIMPLE_MOVEMENT

    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_create_env_success(self, mock_monitor, mock_joypad, mock_make):
        """Test successful environment creation."""
        # Setup mocks
        mock_gym_env = Mock()
        mock_gym_env.observation_space = Mock()
        mock_gym_env.action_space = Mock()
        mock_make.return_value = mock_gym_env
        
        mock_joypad_env = Mock()
        mock_joypad_env.observation_space = Mock()
        mock_joypad_env.action_space = Mock()
        mock_joypad.return_value = mock_joypad_env
        
        mock_monitor_env = Mock()
        mock_monitor_env.observation_space = Mock()
        mock_monitor_env.action_space = Mock()
        mock_monitor.return_value = mock_monitor_env
        
        # Create environment
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        env = env_wrapper.create_env()
        
        # Verify calls
        mock_make.assert_called_once_with('SuperMarioBros-1-1-v0')
        mock_joypad.assert_called_once_with(mock_gym_env, env_wrapper.movement)
        mock_monitor.assert_called_once_with(mock_joypad_env)
        
        assert env == mock_monitor_env

    @patch('mario_rl.environments.make')
    def test_create_env_failure(self, mock_make):
        """Test environment creation failure."""
        # Setup mock to raise exception
        mock_make.side_effect = Exception("Environment creation failed")
        
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        
        with pytest.raises(Exception, match="Environment creation failed"):
            env_wrapper.create_env()

    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_create_vectorized_env_single(self, mock_monitor, mock_joypad, mock_make):
        """Test single environment vectorization."""
        # Setup mocks
        mock_gym_env = Mock()
        mock_gym_env.observation_space = Mock()
        mock_gym_env.action_space = Mock()
        mock_make.return_value = mock_gym_env
        
        mock_joypad_env = Mock()
        mock_joypad_env.observation_space = Mock()
        mock_joypad_env.action_space = Mock()
        mock_joypad.return_value = mock_joypad_env
        
        mock_monitor_env = Mock()
        mock_monitor_env.observation_space = Mock()
        mock_monitor_env.action_space = Mock()
        mock_monitor.return_value = mock_monitor_env
        
        # Create vectorized environment
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        vec_env = env_wrapper.create_vectorized_env(n_envs=1)
        
        # Verify it's a DummyVecEnv
        assert isinstance(vec_env, DummyVecEnv)
        assert vec_env.num_envs == 1

    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_create_vectorized_env_multiple(self, mock_monitor, mock_joypad, mock_make):
        """Test multiple environment vectorization."""
        # Setup mocks
        mock_gym_env = Mock()
        mock_gym_env.observation_space = Mock()
        mock_gym_env.action_space = Mock()
        mock_make.return_value = mock_gym_env
        
        mock_joypad_env = Mock()
        mock_joypad_env.observation_space = Mock()
        mock_joypad_env.action_space = Mock()
        mock_joypad.return_value = mock_joypad_env
        
        mock_monitor_env = Mock()
        mock_monitor_env.observation_space = Mock()
        mock_monitor_env.action_space = Mock()
        mock_monitor.return_value = mock_monitor_env
        
        # Create vectorized environment
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        vec_env = env_wrapper.create_vectorized_env(n_envs=4)
        
        # Verify it's a SubprocVecEnv
        assert isinstance(vec_env, SubprocVecEnv)
        assert vec_env.num_envs == 4

    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_create_vectorized_env_failure(self, mock_monitor, mock_joypad, mock_make):
        """Test vectorized environment creation failure."""
        # Setup mock to raise exception
        mock_make.side_effect = Exception("Vectorized environment creation failed")
        
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        
        with pytest.raises(Exception, match="Vectorized environment creation failed"):
            env_wrapper.create_vectorized_env(n_envs=2)

    def test_get_action_space_size(self):
        """Test action space size retrieval."""
        env_wrapper = MarioEnvironment(movement_type='simple')
        action_size = env_wrapper.get_action_space_size()
        assert action_size == 7  # SIMPLE_MOVEMENT has 7 actions
        
        env_wrapper = MarioEnvironment(movement_type='complex')
        action_size = env_wrapper.get_action_space_size()
        assert action_size == 12  # COMPLEX_MOVEMENT has 12 actions

    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_get_observation_space(self, mock_monitor, mock_joypad, mock_make):
        """Test observation space retrieval."""
        # Setup mocks
        mock_gym_env = Mock()
        mock_obs_space = Mock()
        mock_obs_space.shape = (84, 84, 4)
        mock_gym_env.observation_space = mock_obs_space
        mock_make.return_value = mock_gym_env
        
        mock_joypad_env = Mock()
        mock_joypad_env.observation_space = mock_obs_space
        mock_joypad.return_value = mock_joypad_env
        
        mock_monitor_env = Mock()
        mock_monitor_env.observation_space = mock_obs_space
        mock_monitor.return_value = mock_monitor_env
        
        # Mock close method
        mock_monitor_env.close = Mock()
        
        # Test observation space retrieval
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        obs_space = env_wrapper.get_observation_space()
        
        assert obs_space == mock_obs_space
        mock_monitor_env.close.assert_called_once()

    def test_validate_level_valid(self):
        """Test level validation with valid levels."""
        env_wrapper = MarioEnvironment()
        
        # Test valid levels
        valid_levels = ['1-1', '1-2', '1-3', '1-4', '2-1', '8-4']
        for level in valid_levels:
            assert env_wrapper.validate_level(level) == True

    def test_validate_level_invalid(self):
        """Test level validation with invalid levels."""
        env_wrapper = MarioEnvironment()
        
        # Test invalid levels
        invalid_levels = [
            '0-1',  # World 0 doesn't exist
            '1-0',  # Stage 0 doesn't exist
            '1-5',  # Stage 5 doesn't exist
            '9-1',  # World 9 doesn't exist
            '1-',   # Incomplete format
            '-1',   # Incomplete format
            '1',    # Missing stage
            'a-1',  # Non-numeric world
            '1-b',  # Non-numeric stage
            'invalid',  # Completely invalid
            '',     # Empty string
            None    # None value
        ]
        
        for level in invalid_levels:
            assert env_wrapper.validate_level(level) == False

    def test_validate_level_edge_cases(self):
        """Test level validation edge cases."""
        env_wrapper = MarioEnvironment()
        
        # Test edge cases
        assert env_wrapper.validate_level('1-1') == True   # Minimum valid
        assert env_wrapper.validate_level('8-4') == True   # Maximum valid
        assert env_wrapper.validate_level('1-4') == True   # Last stage of world 1
        assert env_wrapper.validate_level('8-1') == True   # First stage of world 8

    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_environment_reset_and_step(self, mock_monitor, mock_joypad, mock_make):
        """Test environment reset and step functionality."""
        # Setup mocks
        mock_gym_env = Mock()
        mock_gym_env.observation_space = Mock()
        mock_gym_env.action_space = Mock()
        mock_make.return_value = mock_gym_env
        
        mock_joypad_env = Mock()
        mock_joypad_env.observation_space = Mock()
        mock_joypad_env.action_space = Mock()
        mock_joypad.return_value = mock_joypad_env
        
        mock_monitor_env = Mock()
        mock_monitor_env.observation_space = Mock()
        mock_monitor_env.action_space = Mock()
        
        # Mock reset and step methods
        mock_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        mock_monitor_env.reset.return_value = mock_obs
        mock_monitor_env.step.return_value = (mock_obs, 1.0, False, {})
        mock_monitor.return_value = mock_monitor_env
        
        # Create environment and test
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        env = env_wrapper.create_env()
        
        # Test reset
        obs = env.reset()
        assert obs is not None
        mock_monitor_env.reset.assert_called_once()
        
        # Test step
        action = 0
        obs, reward, done, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        mock_monitor_env.step.assert_called_once_with(action)

    def test_movement_type_switching(self):
        """Test switching between movement types."""
        # Test simple movement
        env_simple = MarioEnvironment(movement_type='simple')
        assert env_simple.get_action_space_size() == 7
        
        # Test complex movement
        env_complex = MarioEnvironment(movement_type='complex')
        assert env_complex.get_action_space_size() == 12

    def test_level_switching(self):
        """Test switching between different levels."""
        levels = ['1-1', '1-2', '2-1', '3-1', '8-4']
        
        for level in levels:
            env = MarioEnvironment(level=level)
            assert env.level == level
            assert env.validate_level(level) == True

    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_environment_cleanup(self, mock_monitor, mock_joypad, mock_make):
        """Test environment cleanup and resource management."""
        # Setup mocks
        mock_gym_env = Mock()
        mock_gym_env.observation_space = Mock()
        mock_gym_env.action_space = Mock()
        mock_make.return_value = mock_gym_env
        
        mock_joypad_env = Mock()
        mock_joypad_env.observation_space = Mock()
        mock_joypad_env.action_space = Mock()
        mock_joypad.return_value = mock_joypad_env
        
        mock_monitor_env = Mock()
        mock_monitor_env.observation_space = Mock()
        mock_monitor_env.action_space = Mock()
        mock_monitor_env.close = Mock()
        mock_monitor.return_value = mock_monitor_env
        
        # Create environment
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        env = env_wrapper.create_env()
        
        # Test cleanup
        env.close()
        mock_monitor_env.close.assert_called_once()

    def test_environment_properties(self):
        """Test environment properties and attributes."""
        env_wrapper = MarioEnvironment(level='2-3', movement_type='complex')
        
        # Test properties
        assert env_wrapper.level == '2-3'
        assert env_wrapper.movement is not None
        assert len(env_wrapper.movement) == 12  # COMPLEX_MOVEMENT

    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_environment_error_handling(self, mock_monitor, mock_joypad, mock_make):
        """Test environment error handling."""
        # Test with invalid level
        mock_make.side_effect = Exception("Invalid level")
        
        env_wrapper = MarioEnvironment(level='invalid-level')
        
        with pytest.raises(Exception, match="Invalid level"):
            env_wrapper.create_env()

    def test_environment_immutability(self):
        """Test that environment wrapper properties are immutable after creation."""
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        
        # Store original values
        original_level = env_wrapper.level
        original_movement = env_wrapper.movement
        
        # Try to modify (should not affect the wrapper)
        env_wrapper.level = '2-1'
        env_wrapper.movement = 'complex'
        
        # Properties should remain unchanged
        assert env_wrapper.level == original_level
        assert env_wrapper.movement == original_movement


class TestMarioEnvironmentIntegration:
    """Integration tests for Mario environment."""

    @pytest.mark.slow
    @patch('mario_rl.environments.make')
    @patch('mario_rl.environments.JoypadSpace')
    @patch('mario_rl.environments.Monitor')
    def test_full_environment_workflow(self, mock_monitor, mock_joypad, mock_make):
        """Test complete environment workflow."""
        # Setup comprehensive mocks
        mock_gym_env = Mock()
        mock_gym_env.observation_space = Mock()
        mock_gym_env.action_space = Mock()
        mock_make.return_value = mock_gym_env
        
        mock_joypad_env = Mock()
        mock_joypad_env.observation_space = Mock()
        mock_joypad_env.action_space = Mock()
        mock_joypad.return_value = mock_joypad_env
        
        mock_monitor_env = Mock()
        mock_monitor_env.observation_space = Mock()
        mock_monitor_env.action_space = Mock()
        
        # Mock environment methods
        mock_obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        mock_monitor_env.reset.return_value = mock_obs
        mock_monitor_env.step.return_value = (mock_obs, 1.0, False, {})
        mock_monitor_env.close = Mock()
        mock_monitor.return_value = mock_monitor_env
        
        # Test workflow
        env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
        
        # Create environment
        env = env_wrapper.create_env()
        
        # Test multiple resets and steps
        for _ in range(5):
            obs = env.reset()
            assert obs is not None
            
            for _ in range(10):
                action = np.random.randint(0, 7)
                obs, reward, done, info = env.step(action)
                assert obs is not None
                assert isinstance(reward, (int, float))
                assert isinstance(done, bool)
                assert isinstance(info, dict)
                
                if done:
                    break
        
        # Cleanup
        env.close()
        mock_monitor_env.close.assert_called()

    @pytest.mark.slow
    def test_vectorized_environment_workflow(self):
        """Test vectorized environment workflow."""
        with patch('mario_rl.environments.make') as mock_make, \
             patch('mario_rl.environments.JoypadSpace') as mock_joypad, \
             patch('mario_rl.environments.Monitor') as mock_monitor:
            
            # Setup mocks
            mock_gym_env = Mock()
            mock_gym_env.observation_space = Mock()
            mock_gym_env.action_space = Mock()
            mock_make.return_value = mock_gym_env
            
            mock_joypad_env = Mock()
            mock_joypad_env.observation_space = Mock()
            mock_joypad_env.action_space = Mock()
            mock_joypad.return_value = mock_joypad_env
            
            mock_monitor_env = Mock()
            mock_monitor_env.observation_space = Mock()
            mock_monitor_env.action_space = Mock()
            mock_monitor_env.close = Mock()
            mock_monitor.return_value = mock_monitor_env
            
            # Create vectorized environment
            env_wrapper = MarioEnvironment(level='1-1', movement_type='simple')
            vec_env = env_wrapper.create_vectorized_env(n_envs=4)
            
            # Test vectorized operations
            assert vec_env.num_envs == 4
            assert isinstance(vec_env, SubprocVecEnv)
            
            # Test reset
            obs = vec_env.reset()
            assert obs is not None
            
            # Test step
            actions = [0, 1, 2, 3]
            obs, rewards, dones, infos = vec_env.step(actions)
            assert obs is not None
            assert len(rewards) == 4
            assert len(dones) == 4
            assert len(infos) == 4
            
            # Cleanup
            vec_env.close()
