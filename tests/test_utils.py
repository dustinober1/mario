"""Comprehensive tests for utility functions."""

import pytest
import os
import tempfile
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mario_rl.utils.logging_utils import (
    setup_logging,
    get_logger,
    log_training_start,
    log_training_progress,
    log_evaluation_results,
    log_model_save,
    log_model_load
)
from mario_rl.utils.plotting_utils import (
    plot_training_progress,
    plot_evaluation_results,
    plot_model_comparison,
    plot_learning_curves,
    create_training_summary_plot
)


class TestLoggingUtils:
    """Test logging utility functions."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        logger = setup_logging()
        
        # Test logger properties
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'mario_rl'
        assert logger.level == logging.INFO
        
        # Test handlers
        assert len(logger.handlers) == 1  # Console handler only
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = setup_logging(log_file=log_file)
            
            # Test logger properties
            assert isinstance(logger, logging.Logger)
            assert logger.name == 'mario_rl'
            assert logger.level == logging.INFO
            
            # Test handlers
            assert len(logger.handlers) == 2  # Console and file handlers
            assert isinstance(logger.handlers[0], logging.StreamHandler)
            assert isinstance(logger.handlers[1], logging.FileHandler)
            
            # Test file creation
            assert os.path.exists(log_file)

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        logger = setup_logging(log_level=logging.DEBUG)
        
        # Test logger level
        assert logger.level == logging.DEBUG
        
        # Test handler levels
        for handler in logger.handlers:
            assert handler.level == logging.DEBUG

    def test_setup_logging_custom_format(self):
        """Test logging setup with custom format."""
        custom_format = '%(levelname)s - %(message)s'
        logger = setup_logging(log_format=custom_format)
        
        # Test formatter
        for handler in logger.handlers:
            assert handler.formatter._fmt == custom_format

    def test_setup_logging_file_directory_creation(self):
        """Test that log file directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "nested", "directory")
            log_file = os.path.join(log_dir, "test.log")
            
            # Directory shouldn't exist initially
            assert not os.path.exists(log_dir)
            
            # Setup logging should create directory
            logger = setup_logging(log_file=log_file)
            
            # Directory should now exist
            assert os.path.exists(log_dir)
            assert os.path.exists(log_file)

    def test_setup_logging_removes_existing_handlers(self):
        """Test that setup_logging removes existing handlers."""
        logger = setup_logging()
        initial_handler_count = len(logger.handlers)
        
        # Setup logging again
        logger = setup_logging()
        
        # Should have same number of handlers (not duplicated)
        assert len(logger.handlers) == initial_handler_count

    def test_get_logger(self):
        """Test getting logger with specific name."""
        logger = get_logger('test_module')
        
        # Test logger name
        assert logger.name == 'mario_rl.test_module'
        assert isinstance(logger, logging.Logger)

    def test_log_training_start(self):
        """Test logging training start."""
        config = {'learning_rate': 1e-3, 'n_steps': 256}
        
        # Test that function doesn't raise exception
        log_training_start(config)

    def test_log_training_progress(self):
        """Test logging training progress."""
        # Test that function doesn't raise exception
        log_training_progress(episode=1, reward=10.5, info={'x_pos': 100})

    def test_log_evaluation_results(self):
        """Test logging evaluation results."""
        results = {'mean_reward': 10.5, 'std_reward': 2.1}
        
        # Test that function doesn't raise exception
        log_evaluation_results(results)

    def test_log_model_save(self):
        """Test logging model save."""
        path = "/path/to/model.zip"
        
        # Test that function doesn't raise exception
        log_model_save(path)

    def test_log_model_load(self):
        """Test logging model load."""
        path = "/path/to/model.zip"
        
        # Test that function doesn't raise exception
        log_model_load(path)

    def test_logging_integration(self):
        """Test logging integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "integration_test.log")
            logger = setup_logging(log_file=log_file)
            
            # Test logging messages
            logger.info("Test message")
            logger.warning("Test warning")
            logger.error("Test error")
            
            # Test that messages are written to file
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
                assert "Test warning" in content
                assert "Test error" in content


class TestPlottingUtils:
    """Test plotting utility functions."""

    def test_plot_training_progress_basic(self):
        """Test basic training progress plotting."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test that function doesn't raise exception
        with patch('matplotlib.pyplot.show'):
            plot_training_progress(rewards, show=False)

    def test_plot_training_progress_with_losses(self):
        """Test training progress plotting with losses."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        losses = {
            'policy_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'value_loss': [0.3, 0.25, 0.2, 0.15, 0.1]
        }
        
        # Test that function doesn't raise exception
        with patch('matplotlib.pyplot.show'):
            plot_training_progress(rewards, losses=losses, show=False)

    def test_plot_training_progress_save(self):
        """Test training progress plotting with save."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "training_progress.png")
            
            # Test that function doesn't raise exception
            with patch('matplotlib.pyplot.show'):
                plot_training_progress(rewards, save_path=save_path, show=False)
            
            # Test that file is created
            assert os.path.exists(save_path)

    def test_plot_evaluation_results(self):
        """Test evaluation results plotting."""
        results = {
            'rewards': [10.0, 12.0, 8.0, 15.0, 11.0],
            'episode_lengths': [100, 120, 80, 150, 110]
        }
        
        # Test that function doesn't raise exception
        with patch('matplotlib.pyplot.show'):
            plot_evaluation_results(results, show=False)

    def test_plot_evaluation_results_save(self):
        """Test evaluation results plotting with save."""
        results = {
            'rewards': [10.0, 12.0, 8.0, 15.0, 11.0]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "evaluation_results.png")
            
            # Test that function doesn't raise exception
            with patch('matplotlib.pyplot.show'):
                plot_evaluation_results(results, save_path=save_path, show=False)
            
            # Test that file is created
            assert os.path.exists(save_path)

    def test_plot_model_comparison(self):
        """Test model comparison plotting."""
        models = {
            'Model A': {'accuracy': 0.85, 'f1_score': 0.82},
            'Model B': {'accuracy': 0.90, 'f1_score': 0.88}
        }
        metrics = ['accuracy', 'f1_score']
        
        # Test that function doesn't raise exception
        with patch('matplotlib.pyplot.show'):
            plot_model_comparison(models, metrics, show=False)

    def test_plot_model_comparison_save(self):
        """Test model comparison plotting with save."""
        models = {
            'Model A': {'accuracy': 0.85, 'f1_score': 0.82},
            'Model B': {'accuracy': 0.90, 'f1_score': 0.88}
        }
        metrics = ['accuracy', 'f1_score']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "model_comparison.png")
            
            # Test that function doesn't raise exception
            with patch('matplotlib.pyplot.show'):
                plot_model_comparison(models, metrics, save_path=save_path, show=False)
            
            # Test that file is created
            assert os.path.exists(save_path)

    def test_plot_learning_curves(self):
        """Test learning curves plotting."""
        training_data = {
            'rewards': [1.0, 2.0, 3.0, 4.0, 5.0],
            'losses': [0.5, 0.4, 0.3, 0.2, 0.1]
        }
        
        # Test that function doesn't raise exception
        with patch('matplotlib.pyplot.show'):
            plot_learning_curves(training_data, show=False)

    def test_plot_learning_curves_save(self):
        """Test learning curves plotting with save."""
        training_data = {
            'rewards': [1.0, 2.0, 3.0, 4.0, 5.0],
            'losses': [0.5, 0.4, 0.3, 0.2, 0.1]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "learning_curves.png")
            
            # Test that function doesn't raise exception
            with patch('matplotlib.pyplot.show'):
                plot_learning_curves(training_data, save_path=save_path, show=False)
            
            # Test that file is created
            assert os.path.exists(save_path)

    def test_create_training_summary_plot(self):
        """Test training summary plot creation."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        losses = {
            'policy_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'value_loss': [0.3, 0.25, 0.2, 0.15, 0.1]
        }
        
        # Test that function doesn't raise exception
        with patch('matplotlib.pyplot.show'):
            create_training_summary_plot(rewards, losses=losses, show=False)

    def test_create_training_summary_plot_without_losses(self):
        """Test training summary plot creation without losses."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test that function doesn't raise exception
        with patch('matplotlib.pyplot.show'):
            create_training_summary_plot(rewards, show=False)

    def test_create_training_summary_plot_save(self):
        """Test training summary plot creation with save."""
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        losses = {
            'policy_loss': [0.5, 0.4, 0.3, 0.2, 0.1]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "training_summary.png")
            
            # Test that function doesn't raise exception
            with patch('matplotlib.pyplot.show'):
                create_training_summary_plot(rewards, losses=losses, save_path=save_path, show=False)
            
            # Test that file is created
            assert os.path.exists(save_path)

    def test_plotting_with_empty_data(self):
        """Test plotting with empty data."""
        # Test with empty rewards
        with patch('matplotlib.pyplot.show'):
            plot_training_progress([], show=False)
        
        # Test with empty results
        with patch('matplotlib.pyplot.show'):
            plot_evaluation_results({}, show=False)
        
        # Test with empty training data
        with patch('matplotlib.pyplot.show'):
            plot_learning_curves({}, show=False)

    def test_plotting_with_single_data_point(self):
        """Test plotting with single data point."""
        # Test with single reward
        with patch('matplotlib.pyplot.show'):
            plot_training_progress([1.0], show=False)
        
        # Test with single result
        with patch('matplotlib.pyplot.show'):
            plot_evaluation_results({'rewards': [10.0]}, show=False)

    def test_plotting_with_large_data(self):
        """Test plotting with large datasets."""
        # Test with large reward list
        large_rewards = list(range(1000))
        with patch('matplotlib.pyplot.show'):
            plot_training_progress(large_rewards, show=False)
        
        # Test with large results
        large_results = {
            'rewards': list(range(1000)),
            'episode_lengths': list(range(1000))
        }
        with patch('matplotlib.pyplot.show'):
            plot_evaluation_results(large_results, show=False)

    def test_plotting_error_handling(self):
        """Test plotting error handling."""
        # Test with invalid data types
        with patch('matplotlib.pyplot.show'):
            try:
                plot_training_progress("invalid_data", show=False)
            except (TypeError, ValueError):
                pass  # Expected to fail
        
        # Test with None values
        with patch('matplotlib.pyplot.show'):
            try:
                plot_training_progress([None, None], show=False)
            except (TypeError, ValueError):
                pass  # Expected to fail

    def test_plotting_file_creation(self):
        """Test that plotting functions create files correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test training progress plot
            save_path = os.path.join(temp_dir, "test_plot.png")
            with patch('matplotlib.pyplot.show'):
                plot_training_progress([1.0, 2.0, 3.0], save_path=save_path, show=False)
            assert os.path.exists(save_path)
            
            # Test evaluation results plot
            save_path = os.path.join(temp_dir, "test_eval.png")
            with patch('matplotlib.pyplot.show'):
                plot_evaluation_results({'rewards': [1.0, 2.0, 3.0]}, save_path=save_path, show=False)
            assert os.path.exists(save_path)

    def test_plotting_directory_creation(self):
        """Test that plotting functions create directories if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nested", "directory")
            save_path = os.path.join(nested_dir, "test_plot.png")
            
            # Directory shouldn't exist initially
            assert not os.path.exists(nested_dir)
            
            # Plotting should create directory
            with patch('matplotlib.pyplot.show'):
                plot_training_progress([1.0, 2.0, 3.0], save_path=save_path, show=False)
            
            # Directory should now exist
            assert os.path.exists(nested_dir)
            assert os.path.exists(save_path)


class TestUtilityIntegration:
    """Integration tests for utility functions."""

    def test_logging_and_plotting_integration(self):
        """Test integration between logging and plotting utilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup logging
            log_file = os.path.join(temp_dir, "integration.log")
            logger = setup_logging(log_file=log_file)
            
            # Log some data
            logger.info("Starting training")
            log_training_start({'learning_rate': 1e-3})
            
            # Create some training data
            rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
            losses = {'policy_loss': [0.5, 0.4, 0.3, 0.2, 0.1]}
            
            # Plot training progress
            plot_path = os.path.join(temp_dir, "training_plot.png")
            with patch('matplotlib.pyplot.show'):
                plot_training_progress(rewards, losses=losses, save_path=plot_path, show=False)
            
            # Log completion
            logger.info("Training completed")
            log_evaluation_results({'mean_reward': 5.0})
            
            # Verify files were created
            assert os.path.exists(log_file)
            assert os.path.exists(plot_path)
            
            # Verify log content
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Starting training" in content
                assert "Training completed" in content

    def test_utility_functions_with_real_data(self):
        """Test utility functions with realistic data."""
        # Create realistic training data
        np.random.seed(42)
        rewards = np.random.normal(10, 2, 100).tolist()
        losses = {
            'policy_loss': np.random.exponential(0.1, 100).tolist(),
            'value_loss': np.random.exponential(0.05, 100).tolist()
        }
        
        # Test plotting with realistic data
        with patch('matplotlib.pyplot.show'):
            plot_training_progress(rewards, losses=losses, show=False)
        
        # Create realistic evaluation data
        eval_results = {
            'rewards': np.random.normal(15, 3, 50).tolist(),
            'episode_lengths': np.random.normal(200, 50, 50).tolist()
        }
        
        # Test evaluation plotting
        with patch('matplotlib.pyplot.show'):
            plot_evaluation_results(eval_results, show=False)
        
        # Test model comparison
        models = {
            'PPO': {'mean_reward': 15.2, 'std_reward': 2.8, 'success_rate': 0.85},
            'A2C': {'mean_reward': 12.1, 'std_reward': 3.2, 'success_rate': 0.72}
        }
        metrics = ['mean_reward', 'std_reward', 'success_rate']
        
        with patch('matplotlib.pyplot.show'):
            plot_model_comparison(models, metrics, show=False)

    def test_utility_functions_error_recovery(self):
        """Test utility functions error recovery."""
        # Test logging error recovery
        try:
            setup_logging(log_file="/invalid/path/test.log")
        except (OSError, PermissionError):
            # Should fall back to console logging
            logger = setup_logging()
            assert logger is not None
        
        # Test plotting error recovery
        try:
            with patch('matplotlib.pyplot.show'):
                plot_training_progress([1.0, 2.0, 3.0], save_path="/invalid/path/plot.png", show=False)
        except (OSError, PermissionError):
            # Should still work without saving
            with patch('matplotlib.pyplot.show'):
                plot_training_progress([1.0, 2.0, 3.0], show=False)

    def test_utility_functions_performance(self):
        """Test utility functions performance with large datasets."""
        # Test logging performance
        logger = setup_logging()
        
        # Log many messages
        for i in range(1000):
            logger.info(f"Test message {i}")
        
        # Test plotting performance with large datasets
        large_rewards = list(range(10000))
        large_losses = {
            'policy_loss': [0.1] * 10000,
            'value_loss': [0.05] * 10000
        }
        
        with patch('matplotlib.pyplot.show'):
            plot_training_progress(large_rewards, losses=large_losses, show=False)
        
        # Test evaluation plotting with large datasets
        large_eval_results = {
            'rewards': list(range(1000)),
            'episode_lengths': list(range(1000))
        }
        
        with patch('matplotlib.pyplot.show'):
            plot_evaluation_results(large_eval_results, show=False)
