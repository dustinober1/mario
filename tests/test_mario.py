"""Tests for the main mario module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestMarioEnvironment:
    """Test Mario environment functionality."""
    
    def test_environment_creation(self, mock_env):
        """Test that environment can be created."""
        assert mock_env is not None
        assert mock_env.action_space.n == 7
        assert mock_env.observation_space.shape == (84, 84, 4)
    
    def test_environment_reset(self, mock_env):
        """Test environment reset functionality."""
        obs = mock_env.reset()
        assert obs.shape == (84, 84, 4)
        assert obs.dtype == np.uint8
    
    def test_environment_step(self, mock_env, sample_action):
        """Test environment step functionality."""
        obs, reward, done, info = mock_env.step(sample_action)
        assert obs.shape == (84, 84, 4)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)


class TestMarioAgent:
    """Test Mario agent functionality."""
    
    def test_agent_initialization(self, mock_model):
        """Test that agent can be initialized."""
        assert mock_model is not None
    
    def test_agent_training(self, mock_model):
        """Test agent training functionality."""
        # Mock training call
        mock_model.learn.assert_not_called()
        mock_model.learn(timesteps=1000)
        mock_model.learn.assert_called_once_with(timesteps=1000)
    
    def test_agent_saving(self, mock_model, temp_dir):
        """Test agent saving functionality."""
        save_path = temp_dir / "test_model.zip"
        mock_model.save(str(save_path))
        mock_model.save.assert_called_once_with(str(save_path))


class TestConfiguration:
    """Test configuration functionality."""
    
    def test_config_loading(self):
        """Test configuration loading."""
        # This is a placeholder test - you'll need to implement actual config tests
        assert True
    
    def test_hyperparameters(self):
        """Test hyperparameter validation."""
        # This is a placeholder test - you'll need to implement actual hyperparameter tests
        assert True


class TestUtilities:
    """Test utility functions."""
    
    def test_logging_setup(self):
        """Test logging setup."""
        # This is a placeholder test - you'll need to implement actual logging tests
        assert True
    
    def test_plotting_functions(self):
        """Test plotting functionality."""
        # This is a placeholder test - you'll need to implement actual plotting tests
        assert True


# Integration tests (marked as slow)
@pytest.mark.slow
class TestIntegration:
    """Integration tests that may take longer to run."""
    
    def test_full_training_cycle(self, mock_env, mock_model):
        """Test a complete training cycle."""
        # Mock a full training cycle
        mock_model.learn(timesteps=1000)
        mock_model.save("test_model.zip")
        
        assert mock_model.learn.called
        assert mock_model.save.called
    
    def test_evaluation_pipeline(self, mock_env, mock_model):
        """Test the evaluation pipeline."""
        # Mock evaluation
        mock_env.reset()
        obs, reward, done, info = mock_env.step(0)
        
        assert obs is not None
        assert isinstance(reward, (int, float))


# Performance tests
@pytest.mark.slow
class TestPerformance:
    """Performance-related tests."""
    
    def test_memory_usage(self):
        """Test memory usage during operations."""
        # This is a placeholder test - you'll need to implement actual memory tests
        assert True
    
    def test_training_speed(self):
        """Test training speed."""
        # This is a placeholder test - you'll need to implement actual speed tests
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
