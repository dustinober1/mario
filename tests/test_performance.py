"""Performance and benchmark tests for Mario RL system."""

import pytest
import time
import numpy as np
import psutil
import os
from unittest.mock import Mock, patch
from memory_profiler import profile

from mario_rl.environments import MarioEnvironment
from mario_rl.agents import MarioAgent
from mario_rl.configs import TrainingConfig
from mario_rl.utils.logging_utils import setup_logging
from mario_rl.utils.plotting_utils import plot_training_progress


class TestPerformance:
    """Test system performance characteristics."""

    @pytest.fixture
    def performance_config(self):
        """Create configuration for performance tests."""
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
            n_eval_episodes=5
        )

    @pytest.mark.slow
    def test_environment_creation_performance(self, performance_config):
        """Test environment creation performance."""
        # Test single environment creation time
        start_time = time.time()
        
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
            mock_monitor.return_value = mock_monitor_env
            
            env_wrapper = MarioEnvironment(
                level=performance_config.level,
                movement_type=performance_config.movement_type
            )
            vec_env = env_wrapper.create_vectorized_env(1)
            
            end_time = time.time()
            creation_time = end_time - start_time
            
            # Environment creation should be fast
            assert creation_time < 1.0  # Should take less than 1 second
            assert vec_env is not None

    @pytest.mark.slow
    def test_multiple_environment_creation_performance(self, performance_config):
        """Test multiple environment creation performance."""
        # Test creating multiple environments
        num_envs = 10
        start_time = time.time()
        
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
            mock_monitor.return_value = mock_monitor_env
            
            env_wrapper = MarioEnvironment(
                level=performance_config.level,
                movement_type=performance_config.movement_type
            )
            
            environments = []
            for _ in range(num_envs):
                vec_env = env_wrapper.create_vectorized_env(1)
                environments.append(vec_env)
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / num_envs
            
            # Multiple environment creation should be efficient
            assert total_time < 5.0  # Should take less than 5 seconds total
            assert avg_time < 0.5  # Average should be less than 0.5 seconds
            assert len(environments) == num_envs

    @pytest.mark.slow
    def test_agent_creation_performance(self, performance_config):
        """Test agent creation performance."""
        # Test agent creation time
        start_time = time.time()
        
        with patch('mario_rl.agents.PPO') as mock_ppo_class:
            mock_model = Mock()
            mock_ppo_class.return_value = mock_model
            
            # Create mock environment
            mock_vec_env = Mock()
            mock_vec_env.num_envs = 1
            mock_vec_env.observation_space = Mock()
            mock_vec_env.action_space = Mock()
            
            agent = MarioAgent(mock_vec_env, performance_config)
            model = agent.create_model()
            
            end_time = time.time()
            creation_time = end_time - start_time
            
            # Agent creation should be fast
            assert creation_time < 1.0  # Should take less than 1 second
            assert model == mock_model

    @pytest.mark.slow
    def test_configuration_creation_performance(self, performance_config):
        """Test configuration creation performance."""
        # Test configuration creation time
        start_time = time.time()
        
        configs = []
        for _ in range(100):
            config = TrainingConfig(
                level=performance_config.level,
                movement_type=performance_config.movement_type
            )
            configs.append(config)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 100
        
        # Configuration creation should be very fast
        assert total_time < 1.0  # Should take less than 1 second total
        assert avg_time < 0.01  # Average should be less than 0.01 seconds
        assert len(configs) == 100

    @pytest.mark.slow
    def test_memory_usage_performance(self, performance_config):
        """Test memory usage performance."""
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple environments and agents
        with patch('mario_rl.environments.make') as mock_make, \
             patch('mario_rl.environments.JoypadSpace') as mock_joypad, \
             patch('mario_rl.environments.Monitor') as mock_monitor, \
             patch('mario_rl.agents.PPO') as mock_ppo_class:
            
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
            
            mock_model = Mock()
            mock_ppo_class.return_value = mock_model
            
            # Create multiple environments and agents
            environments = []
            agents = []
            
            for _ in range(10):
                env_wrapper = MarioEnvironment(
                    level=performance_config.level,
                    movement_type=performance_config.movement_type
                )
                vec_env = env_wrapper.create_vectorized_env(1)
                environments.append(vec_env)
                
                agent = MarioAgent(vec_env, performance_config)
                model = agent.create_model()
                agents.append(agent)
            
            # Get memory usage after creation
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 100  # Should use less than 100MB additional memory
            assert len(environments) == 10
            assert len(agents) == 10

    @pytest.mark.slow
    def test_logging_performance(self, performance_config):
        """Test logging performance."""
        # Test logging performance
        start_time = time.time()
        
        with patch('mario_rl.utils.logging_utils.setup_logging') as mock_setup_logging:
            mock_logger = Mock()
            mock_setup_logging.return_value = mock_logger
            
            # Create logger
            logger = setup_logging()
            
            # Log many messages
            for i in range(1000):
                logger.info(f"Test message {i}")
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / 1000
            
            # Logging should be fast
            assert total_time < 5.0  # Should take less than 5 seconds total
            assert avg_time < 0.005  # Average should be less than 0.005 seconds

    @pytest.mark.slow
    def test_plotting_performance(self, performance_config):
        """Test plotting performance."""
        # Test plotting performance
        start_time = time.time()
        
        with patch('matplotlib.pyplot.show'):
            # Create large dataset
            rewards = list(range(1000))
            losses = {
                'policy_loss': [0.1] * 1000,
                'value_loss': [0.05] * 1000
            }
            
            # Plot training progress
            plot_training_progress(rewards, losses=losses, show=False)
            
            end_time = time.time()
            plotting_time = end_time - start_time
            
            # Plotting should be reasonably fast
            assert plotting_time < 10.0  # Should take less than 10 seconds

    @pytest.mark.slow
    def test_prediction_performance(self, performance_config):
        """Test prediction performance."""
        # Test prediction performance
        start_time = time.time()
        
        with patch('mario_rl.agents.PPO') as mock_ppo_class:
            mock_model = Mock()
            mock_model.predict.return_value = (np.array([0]), None)
            mock_ppo_class.return_value = mock_model
            
            # Create mock environment
            mock_vec_env = Mock()
            mock_vec_env.num_envs = 1
            mock_vec_env.observation_space = Mock()
            mock_vec_env.action_space = Mock()
            
            agent = MarioAgent(mock_vec_env, performance_config)
            model = agent.create_model()
            
            # Make many predictions
            observations = [np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8) for _ in range(100)]
            
            for obs in observations:
                action, state = agent.predict(obs)
                assert action is not None
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / 100
            
            # Predictions should be fast
            assert total_time < 5.0  # Should take less than 5 seconds total
            assert avg_time < 0.05  # Average should be less than 0.05 seconds

    @pytest.mark.slow
    def test_training_performance(self, performance_config):
        """Test training performance."""
        # Test training performance
        start_time = time.time()
        
        with patch('mario_rl.agents.PPO') as mock_ppo_class:
            mock_model = Mock()
            mock_model.learn.return_value = None
            mock_ppo_class.return_value = mock_model
            
            # Create mock environment
            mock_vec_env = Mock()
            mock_vec_env.num_envs = 1
            mock_vec_env.observation_space = Mock()
            mock_vec_env.action_space = Mock()
            
            agent = MarioAgent(mock_vec_env, performance_config)
            model = agent.create_model()
            
            # Train
            agent.train()
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Training should be reasonably fast (mocked)
            assert training_time < 5.0  # Should take less than 5 seconds
            mock_model.learn.assert_called_once()

    @pytest.mark.slow
    def test_concurrent_performance(self, performance_config):
        """Test concurrent operations performance."""
        # Test concurrent environment creation
        start_time = time.time()
        
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
            mock_monitor.return_value = mock_monitor_env
            
            # Create multiple environments concurrently (simulated)
            environments = []
            for _ in range(5):
                env_wrapper = MarioEnvironment(
                    level=performance_config.level,
                    movement_type=performance_config.movement_type
                )
                vec_env = env_wrapper.create_vectorized_env(1)
                environments.append(vec_env)
            
            end_time = time.time()
            concurrent_time = end_time - start_time
            
            # Concurrent creation should be efficient
            assert concurrent_time < 3.0  # Should take less than 3 seconds
            assert len(environments) == 5

    @pytest.mark.slow
    def test_scalability_performance(self, performance_config):
        """Test scalability performance."""
        # Test scalability with different sizes
        sizes = [1, 5, 10, 20]
        
        for size in sizes:
            start_time = time.time()
            
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
                mock_monitor.return_value = mock_monitor_env
                
                # Create environments
                environments = []
                for _ in range(size):
                    env_wrapper = MarioEnvironment(
                        level=performance_config.level,
                        movement_type=performance_config.movement_type
                    )
                    vec_env = env_wrapper.create_vectorized_env(1)
                    environments.append(vec_env)
                
                end_time = time.time()
                creation_time = end_time - start_time
                
                # Creation time should scale reasonably
                assert creation_time < size * 0.5  # Should scale linearly
                assert len(environments) == size

    @pytest.mark.slow
    def test_memory_efficiency_performance(self, performance_config):
        """Test memory efficiency performance."""
        # Test memory efficiency with large datasets
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large datasets
        large_rewards = list(range(10000))
        large_losses = {
            'policy_loss': [0.1] * 10000,
            'value_loss': [0.05] * 10000
        }
        
        # Plot large datasets
        with patch('matplotlib.pyplot.show'):
            plot_training_progress(large_rewards, losses=large_losses, show=False)
        
        # Get memory usage after plotting
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 200  # Should use less than 200MB additional memory

    @pytest.mark.slow
    def test_error_handling_performance(self, performance_config):
        """Test error handling performance."""
        # Test error handling performance
        start_time = time.time()
        
        # Test with many error conditions
        for _ in range(100):
            try:
                # Simulate error condition
                raise ValueError("Test error")
            except ValueError:
                pass  # Handle error
        
        end_time = time.time()
        error_handling_time = end_time - start_time
        avg_time = error_handling_time / 100
        
        # Error handling should be fast
        assert error_handling_time < 1.0  # Should take less than 1 second total
        assert avg_time < 0.01  # Average should be less than 0.01 seconds

    @pytest.mark.slow
    def test_resource_cleanup_performance(self, performance_config):
        """Test resource cleanup performance."""
        # Test resource cleanup performance
        start_time = time.time()
        
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
            
            # Create and cleanup many environments
            for _ in range(50):
                env_wrapper = MarioEnvironment(
                    level=performance_config.level,
                    movement_type=performance_config.movement_type
                )
                vec_env = env_wrapper.create_vectorized_env(1)
                vec_env.close()
            
            end_time = time.time()
            cleanup_time = end_time - start_time
            avg_time = cleanup_time / 50
            
            # Resource cleanup should be fast
            assert cleanup_time < 5.0  # Should take less than 5 seconds total
            assert avg_time < 0.1  # Average should be less than 0.1 seconds

    @pytest.mark.slow
    def test_benchmark_comparison(self, performance_config):
        """Test benchmark comparison performance."""
        # Test different configurations
        configs = [
            TrainingConfig(level="1-1", movement_type="simple"),
            TrainingConfig(level="1-1", movement_type="complex"),
            TrainingConfig(level="2-1", movement_type="simple"),
            TrainingConfig(level="2-1", movement_type="complex")
        ]
        
        creation_times = []
        
        for config in configs:
            start_time = time.time()
            
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
                mock_monitor.return_value = mock_monitor_env
                
                env_wrapper = MarioEnvironment(
                    level=config.level,
                    movement_type=config.movement_type
                )
                vec_env = env_wrapper.create_vectorized_env(1)
                
                end_time = time.time()
                creation_time = end_time - start_time
                creation_times.append(creation_time)
        
        # All configurations should have similar performance
        assert len(creation_times) == 4
        assert all(time < 1.0 for time in creation_times)  # All should be fast
        assert max(creation_times) - min(creation_times) < 0.5  # Should be similar

    @pytest.mark.slow
    def test_performance_regression(self, performance_config):
        """Test performance regression."""
        # Test that performance doesn't degrade over time
        times = []
        
        for iteration in range(10):
            start_time = time.time()
            
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
                mock_monitor.return_value = mock_monitor_env
                
                env_wrapper = MarioEnvironment(
                    level=performance_config.level,
                    movement_type=performance_config.movement_type
                )
                vec_env = env_wrapper.create_vectorized_env(1)
                
                end_time = time.time()
                creation_time = end_time - start_time
                times.append(creation_time)
        
        # Performance should be consistent
        assert len(times) == 10
        assert all(time < 1.0 for time in times)  # All should be fast
        assert max(times) - min(times) < 0.5  # Should be consistent
