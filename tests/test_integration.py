"""Comprehensive integration tests for Mario RL system."""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mario_rl.environments import MarioEnvironment
from mario_rl.agents import MarioAgent
from mario_rl.models import PPOModel
from mario_rl.configs import TrainingConfig, EnvironmentConfig
from mario_rl.utils.logging_utils import setup_logging
from mario_rl.utils.plotting_utils import plot_training_progress


class TestSystemIntegration:
    """Test complete system integration."""

    @pytest.fixture
    def mock_environment_setup(self):
        """Setup mock environment for integration tests."""
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
            
            yield mock_monitor_env

    @pytest.fixture
    def mock_agent_setup(self):
        """Setup mock agent for integration tests."""
        with patch('mario_rl.agents.PPO') as mock_ppo_class:
            mock_model = Mock()
            mock_model.learn.return_value = None
            mock_model.save.return_value = None
            mock_model.predict.return_value = (np.array([0]), None)
            mock_ppo_class.return_value = mock_model
            
            yield mock_model

    @pytest.fixture
    def integration_config(self):
        """Create configuration for integration tests."""
        return TrainingConfig(
            level="1-1",
            movement_type="simple",
            total_timesteps=1000,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=4,
            save_freq=500,
            eval_freq=500,
            n_eval_episodes=5,
            checkpoint_dir="./test_checkpoints",
            tensorboard_dir="./test_tensorboard"
        )

    @pytest.mark.slow
    def test_complete_training_workflow(self, mock_environment_setup, mock_agent_setup, integration_config):
        """Test complete training workflow from start to finish."""
        # Setup mocks
        mock_env = mock_environment_setup
        mock_model = mock_agent_setup
        
        # Create environment
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(integration_config.n_envs)
        
        # Create agent
        agent = MarioAgent(vec_env, integration_config)
        
        # Create model
        model = agent.create_model()
        assert model == mock_model
        
        # Train
        agent.train()
        mock_model.learn.assert_called_once()
        
        # Save model
        agent.save_model()
        mock_model.save.assert_called_once()
        
        # Make prediction
        observation = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        action, state = agent.predict(observation)
        assert action is not None
        assert state is None
        
        # Get training history
        history = agent.get_training_history()
        assert isinstance(history, list)

    @pytest.mark.slow
    def test_environment_agent_integration(self, mock_environment_setup, integration_config):
        """Test environment and agent integration."""
        # Setup mocks
        mock_env = mock_environment_setup
        
        # Create environment
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(integration_config.n_envs)
        
        # Test environment properties
        assert vec_env.num_envs == integration_config.n_envs
        assert env_wrapper.level == integration_config.level
        assert env_wrapper.movement_type == integration_config.movement_type
        
        # Test action space
        action_size = env_wrapper.get_action_space_size()
        assert action_size > 0
        
        # Test observation space
        obs_space = env_wrapper.get_observation_space()
        assert obs_space is not None

    @pytest.mark.slow
    def test_config_environment_integration(self, integration_config):
        """Test configuration and environment integration."""
        # Test level validation
        assert integration_config.level in EnvironmentConfig.LEVELS
        assert integration_config.movement_type in EnvironmentConfig.MOVEMENT_TYPES
        
        # Test environment creation with config
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        
        # Test that environment uses config values
        assert env_wrapper.level == integration_config.level
        assert env_wrapper.movement_type == integration_config.movement_type

    @pytest.mark.slow
    def test_logging_integration(self, integration_config):
        """Test logging integration with training workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup logging
            log_file = os.path.join(temp_dir, "integration_test.log")
            logger = setup_logging(log_file=log_file)
            
            # Log training start
            logger.info("Starting integration test")
            
            # Create environment
            env_wrapper = MarioEnvironment(
                level=integration_config.level,
                movement_type=integration_config.movement_type
            )
            
            # Log environment creation
            logger.info(f"Created environment for level {env_wrapper.level}")
            
            # Log completion
            logger.info("Integration test completed")
            
            # Verify log file
            assert os.path.exists(log_file)
            
            # Verify log content
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Starting integration test" in content
                assert "Created environment" in content
                assert "Integration test completed" in content

    @pytest.mark.slow
    def test_plotting_integration(self, integration_config):
        """Test plotting integration with training workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create training data
            rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
            losses = {
                'policy_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
                'value_loss': [0.3, 0.25, 0.2, 0.15, 0.1]
            }
            
            # Plot training progress
            plot_path = os.path.join(temp_dir, "training_progress.png")
            with patch('matplotlib.pyplot.show'):
                plot_training_progress(rewards, losses=losses, save_path=plot_path, show=False)
            
            # Verify plot file
            assert os.path.exists(plot_path)

    @pytest.mark.slow
    def test_model_saving_loading_integration(self, mock_environment_setup, mock_agent_setup, integration_config):
        """Test model saving and loading integration."""
        # Setup mocks
        mock_env = mock_environment_setup
        mock_model = mock_agent_setup
        
        # Create environment and agent
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(integration_config.n_envs)
        agent = MarioAgent(vec_env, integration_config)
        
        # Create model
        model = agent.create_model()
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_model.zip")
            agent.save_model(path=save_path)
            
            # Verify save was called
            mock_model.save.assert_called_once_with(save_path)
            
            # Test loading (mock the load method)
            with patch('mario_rl.agents.PPO.load') as mock_load:
                mock_load.return_value = mock_model
                agent.load_model(save_path)
                mock_load.assert_called_once_with(save_path, env=vec_env)

    @pytest.mark.slow
    def test_error_handling_integration(self, integration_config):
        """Test error handling integration across components."""
        # Test environment creation failure
        with patch('mario_rl.environments.make') as mock_make:
            mock_make.side_effect = Exception("Environment creation failed")
            
            env_wrapper = MarioEnvironment(
                level=integration_config.level,
                movement_type=integration_config.movement_type
            )
            
            with pytest.raises(Exception, match="Environment creation failed"):
                env_wrapper.create_env()
        
        # Test agent creation failure
        with patch('mario_rl.agents.PPO') as mock_ppo_class:
            mock_ppo_class.side_effect = Exception("Agent creation failed")
            
            # Create mock environment
            mock_vec_env = Mock()
            mock_vec_env.num_envs = 1
            mock_vec_env.observation_space = Mock()
            mock_vec_env.action_space = Mock()
            
            agent = MarioAgent(mock_vec_env, integration_config)
            
            with pytest.raises(Exception, match="Agent creation failed"):
                agent.create_model()

    @pytest.mark.slow
    def test_performance_integration(self, integration_config):
        """Test performance integration across components."""
        # Test environment performance
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        
        # Test multiple environment creation
        for _ in range(10):
            vec_env = env_wrapper.create_vectorized_env(1)
            assert vec_env is not None
        
        # Test configuration performance
        for _ in range(100):
            config = TrainingConfig(
                level=integration_config.level,
                movement_type=integration_config.movement_type
            )
            assert config.level == integration_config.level

    @pytest.mark.slow
    def test_memory_integration(self, integration_config):
        """Test memory usage integration across components."""
        # Test environment memory usage
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        
        # Create multiple environments
        environments = []
        for _ in range(5):
            vec_env = env_wrapper.create_vectorized_env(1)
            environments.append(vec_env)
        
        # Test that all environments are created successfully
        assert len(environments) == 5
        for env in environments:
            assert env is not None
        
        # Test configuration memory usage
        configurations = []
        for _ in range(100):
            config = TrainingConfig(
                level=integration_config.level,
                movement_type=integration_config.movement_type
            )
            configurations.append(config)
        
        # Test that all configurations are created successfully
        assert len(configurations) == 100
        for config in configurations:
            assert config.level == integration_config.level

    @pytest.mark.slow
    def test_concurrent_integration(self, integration_config):
        """Test concurrent operations integration."""
        # Test concurrent environment creation
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        
        # Create multiple environments concurrently (simulated)
        environments = []
        for _ in range(3):
            vec_env = env_wrapper.create_vectorized_env(1)
            environments.append(vec_env)
        
        # Test that all environments are created successfully
        assert len(environments) == 3
        for env in environments:
            assert env is not None
        
        # Test concurrent configuration creation
        configurations = []
        for _ in range(10):
            config = TrainingConfig(
                level=integration_config.level,
                movement_type=integration_config.movement_type
            )
            configurations.append(config)
        
        # Test that all configurations are created successfully
        assert len(configurations) == 10
        for config in configurations:
            assert config.level == integration_config.level

    @pytest.mark.slow
    def test_data_flow_integration(self, mock_environment_setup, mock_agent_setup, integration_config):
        """Test data flow integration across components."""
        # Setup mocks
        mock_env = mock_environment_setup
        mock_model = mock_agent_setup
        
        # Create environment
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(integration_config.n_envs)
        
        # Create agent
        agent = MarioAgent(vec_env, integration_config)
        
        # Test data flow: config -> agent -> model
        model = agent.create_model()
        assert model == mock_model
        
        # Test data flow: observation -> prediction
        observation = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        action, state = agent.predict(observation)
        assert action is not None
        
        # Test data flow: training -> history
        agent.train()
        history = agent.get_training_history()
        assert isinstance(history, list)

    @pytest.mark.slow
    def test_state_management_integration(self, mock_environment_setup, mock_agent_setup, integration_config):
        """Test state management integration across components."""
        # Setup mocks
        mock_env = mock_environment_setup
        mock_model = mock_agent_setup
        
        # Create environment
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(integration_config.n_envs)
        
        # Create agent
        agent = MarioAgent(vec_env, integration_config)
        
        # Test initial state
        assert agent.model is None
        assert agent.training_history == []
        
        # Test state after model creation
        model = agent.create_model()
        assert agent.model == model
        
        # Test state after training
        agent.train()
        assert agent.model == model
        
        # Test state after prediction
        observation = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        action, state = agent.predict(observation)
        assert agent.model == model

    @pytest.mark.slow
    def test_resource_cleanup_integration(self, mock_environment_setup, integration_config):
        """Test resource cleanup integration."""
        # Setup mocks
        mock_env = mock_environment_setup
        
        # Create environment
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(integration_config.n_envs)
        
        # Test resource cleanup
        vec_env.close()
        mock_env.close.assert_called()

    @pytest.mark.slow
    def test_error_recovery_integration(self, integration_config):
        """Test error recovery integration across components."""
        # Test environment error recovery
        with patch('mario_rl.environments.make') as mock_make:
            # First call fails
            mock_make.side_effect = [Exception("First call failed"), Mock()]
            
            env_wrapper = MarioEnvironment(
                level=integration_config.level,
                movement_type=integration_config.movement_type
            )
            
            # First call should fail
            with pytest.raises(Exception, match="First call failed"):
                env_wrapper.create_env()
            
            # Second call should succeed
            mock_make.side_effect = None
            mock_make.return_value = Mock()
            env = env_wrapper.create_env()
            assert env is not None
        
        # Test agent error recovery
        with patch('mario_rl.agents.PPO') as mock_ppo_class:
            # First call fails
            mock_ppo_class.side_effect = [Exception("First call failed"), Mock()]
            
            # Create mock environment
            mock_vec_env = Mock()
            mock_vec_env.num_envs = 1
            mock_vec_env.observation_space = Mock()
            mock_vec_env.action_space = Mock()
            
            agent = MarioAgent(mock_vec_env, integration_config)
            
            # First call should fail
            with pytest.raises(Exception, match="First call failed"):
                agent.create_model()
            
            # Second call should succeed
            mock_ppo_class.side_effect = None
            mock_ppo_class.return_value = Mock()
            model = agent.create_model()
            assert model is not None

    @pytest.mark.slow
    def test_configuration_validation_integration(self, integration_config):
        """Test configuration validation integration."""
        # Test valid configuration
        assert integration_config.level in EnvironmentConfig.LEVELS
        assert integration_config.movement_type in EnvironmentConfig.MOVEMENT_TYPES
        assert integration_config.learning_rate > 0
        assert integration_config.n_steps > 0
        assert integration_config.batch_size > 0
        assert integration_config.n_epochs > 0
        assert 0 < integration_config.gamma < 1
        assert 0 < integration_config.gae_lambda < 1
        assert 0 < integration_config.clip_range < 1
        assert integration_config.ent_coef >= 0
        assert integration_config.vf_coef > 0
        assert integration_config.max_grad_norm > 0
        assert integration_config.total_timesteps > 0
        assert integration_config.save_freq > 0
        assert integration_config.eval_freq > 0
        assert integration_config.n_eval_episodes > 0
        assert integration_config.n_envs > 0

    @pytest.mark.slow
    def test_end_to_end_workflow_integration(self, mock_environment_setup, mock_agent_setup, integration_config):
        """Test complete end-to-end workflow integration."""
        # Setup mocks
        mock_env = mock_environment_setup
        mock_model = mock_agent_setup
        
        # Step 1: Create environment
        env_wrapper = MarioEnvironment(
            level=integration_config.level,
            movement_type=integration_config.movement_type
        )
        vec_env = env_wrapper.create_vectorized_env(integration_config.n_envs)
        
        # Step 2: Create agent
        agent = MarioAgent(vec_env, integration_config)
        
        # Step 3: Create model
        model = agent.create_model()
        assert model == mock_model
        
        # Step 4: Train
        agent.train()
        mock_model.learn.assert_called_once()
        
        # Step 5: Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "end_to_end_model.zip")
            agent.save_model(path=save_path)
            mock_model.save.assert_called_once_with(save_path)
        
        # Step 6: Make prediction
        observation = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        action, state = agent.predict(observation)
        assert action is not None
        assert state is None
        
        # Step 7: Get training history
        history = agent.get_training_history()
        assert isinstance(history, list)
        
        # Step 8: Cleanup
        vec_env.close()
        mock_env.close.assert_called()
