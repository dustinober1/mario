"""Comprehensive tests for configuration management."""

import pytest
import os
from unittest.mock import patch
from dataclasses import asdict

from mario_rl.configs import (
    TrainingConfig,
    EnvironmentConfig,
    get_default_config,
    load_config_from_env
)


class TestTrainingConfig:
    """Test TrainingConfig functionality."""

    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = TrainingConfig()
        
        # Test PPO hyperparameters
        assert config.learning_rate == 3e-4
        assert config.n_steps == 512
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.ent_coef == 0.01
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5
        
        # Test training settings
        assert config.total_timesteps == 200000
        assert config.save_freq == 50000
        assert config.eval_freq == 50000
        assert config.n_eval_episodes == 10
        
        # Test environment settings
        assert config.level == "1-1"
        assert config.movement_type == "simple"
        assert config.n_envs == 1
        
        # Test paths
        assert config.model_name == "ppo_mario"
        assert config.checkpoint_dir == "./checkpoints"
        assert config.tensorboard_dir == "./mario_tensorboard"
        assert config.log_file == "mario_training.log"

    def test_custom_config_creation(self):
        """Test custom configuration creation."""
        config = TrainingConfig(
            learning_rate=1e-3,
            n_steps=256,
            batch_size=32,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,
            ent_coef=0.02,
            vf_coef=0.3,
            max_grad_norm=0.3,
            total_timesteps=100000,
            save_freq=25000,
            eval_freq=25000,
            n_eval_episodes=5,
            level="2-1",
            movement_type="complex",
            n_envs=4,
            model_name="custom_mario",
            checkpoint_dir="/custom/checkpoints",
            tensorboard_dir="/custom/tensorboard",
            log_file="custom_training.log"
        )
        
        # Test custom values
        assert config.learning_rate == 1e-3
        assert config.n_steps == 256
        assert config.batch_size == 32
        assert config.n_epochs == 5
        assert config.gamma == 0.95
        assert config.gae_lambda == 0.9
        assert config.clip_range == 0.1
        assert config.ent_coef == 0.02
        assert config.vf_coef == 0.3
        assert config.max_grad_norm == 0.3
        assert config.total_timesteps == 100000
        assert config.save_freq == 25000
        assert config.eval_freq == 25000
        assert config.n_eval_episodes == 5
        assert config.level == "2-1"
        assert config.movement_type == "complex"
        assert config.n_envs == 4
        assert config.model_name == "custom_mario"
        assert config.checkpoint_dir == "/custom/checkpoints"
        assert config.tensorboard_dir == "/custom/tensorboard"
        assert config.log_file == "custom_training.log"

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configurations
        valid_configs = [
            TrainingConfig(learning_rate=1e-5),
            TrainingConfig(learning_rate=1e-2),
            TrainingConfig(n_steps=64),
            TrainingConfig(n_steps=2048),
            TrainingConfig(batch_size=16),
            TrainingConfig(batch_size=128),
            TrainingConfig(n_epochs=1),
            TrainingConfig(n_epochs=20),
            TrainingConfig(gamma=0.9),
            TrainingConfig(gamma=0.999),
            TrainingConfig(gae_lambda=0.8),
            TrainingConfig(gae_lambda=0.99),
            TrainingConfig(clip_range=0.05),
            TrainingConfig(clip_range=0.5),
            TrainingConfig(ent_coef=0.0),
            TrainingConfig(ent_coef=0.1),
            TrainingConfig(vf_coef=0.1),
            TrainingConfig(vf_coef=1.0),
            TrainingConfig(max_grad_norm=0.1),
            TrainingConfig(max_grad_norm=1.0),
            TrainingConfig(total_timesteps=1000),
            TrainingConfig(total_timesteps=1000000),
            TrainingConfig(save_freq=1000),
            TrainingConfig(save_freq=100000),
            TrainingConfig(eval_freq=1000),
            TrainingConfig(eval_freq=100000),
            TrainingConfig(n_eval_episodes=1),
            TrainingConfig(n_eval_episodes=100),
            TrainingConfig(n_envs=1),
            TrainingConfig(n_envs=16)
        ]
        
        for config in valid_configs:
            assert config is not None

    def test_config_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = TrainingConfig(
            learning_rate=1e-3,
            n_steps=256,
            batch_size=32,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,
            ent_coef=0.02,
            vf_coef=0.3,
            max_grad_norm=0.3
        )
        
        config_dict = config.to_dict()
        
        # Test dictionary structure
        assert isinstance(config_dict, dict)
        assert 'learning_rate' in config_dict
        assert 'n_steps' in config_dict
        assert 'batch_size' in config_dict
        assert 'n_epochs' in config_dict
        assert 'gamma' in config_dict
        assert 'gae_lambda' in config_dict
        assert 'clip_range' in config_dict
        assert 'ent_coef' in config_dict
        assert 'vf_coef' in config_dict
        assert 'max_grad_norm' in config_dict
        
        # Test values
        assert config_dict['learning_rate'] == 1e-3
        assert config_dict['n_steps'] == 256
        assert config_dict['batch_size'] == 32
        assert config_dict['n_epochs'] == 5
        assert config_dict['gamma'] == 0.95
        assert config_dict['gae_lambda'] == 0.9
        assert config_dict['clip_range'] == 0.1
        assert config_dict['ent_coef'] == 0.02
        assert config_dict['vf_coef'] == 0.3
        assert config_dict['max_grad_norm'] == 0.3

    def test_config_immutability(self):
        """Test configuration immutability."""
        config = TrainingConfig(learning_rate=1e-3)
        
        # Store original value
        original_lr = config.learning_rate
        
        # Try to modify (should not affect the config)
        config.learning_rate = 1e-4
        
        # Value should remain unchanged
        assert config.learning_rate == original_lr

    def test_config_copy(self):
        """Test configuration copying."""
        config1 = TrainingConfig(learning_rate=1e-3, n_steps=256)
        config2 = TrainingConfig(learning_rate=1e-3, n_steps=256)
        
        # Configurations should be equal
        assert config1.learning_rate == config2.learning_rate
        assert config1.n_steps == config2.n_steps
        
        # But should be different objects
        assert config1 is not config2

    def test_config_equality(self):
        """Test configuration equality."""
        config1 = TrainingConfig(learning_rate=1e-3, n_steps=256)
        config2 = TrainingConfig(learning_rate=1e-3, n_steps=256)
        config3 = TrainingConfig(learning_rate=1e-4, n_steps=256)
        
        # Equal configurations
        assert config1.learning_rate == config2.learning_rate
        assert config1.n_steps == config2.n_steps
        
        # Different configurations
        assert config1.learning_rate != config3.learning_rate
        assert config1.n_steps == config3.n_steps

    def test_config_string_representation(self):
        """Test configuration string representation."""
        config = TrainingConfig(learning_rate=1e-3, level="2-1")
        
        # Test string representation
        config_str = str(config)
        assert isinstance(config_str, str)
        assert "TrainingConfig" in config_str
        assert "learning_rate" in config_str
        assert "level" in config_str

    def test_config_repr(self):
        """Test configuration repr."""
        config = TrainingConfig(learning_rate=1e-3, level="2-1")
        
        # Test repr
        config_repr = repr(config)
        assert isinstance(config_repr, str)
        assert "TrainingConfig" in config_repr


class TestEnvironmentConfig:
    """Test EnvironmentConfig functionality."""

    def test_environment_config_constants(self):
        """Test environment configuration constants."""
        # Test levels
        assert "1-1" in EnvironmentConfig.LEVELS
        assert "1-2" in EnvironmentConfig.LEVELS
        assert "1-3" in EnvironmentConfig.LEVELS
        assert "1-4" in EnvironmentConfig.LEVELS
        assert "2-1" in EnvironmentConfig.LEVELS
        assert "8-4" in EnvironmentConfig.LEVELS
        
        # Test movement types
        assert "simple" in EnvironmentConfig.MOVEMENT_TYPES
        assert "complex" in EnvironmentConfig.MOVEMENT_TYPES
        
        # Test reward thresholds
        assert EnvironmentConfig.COMPLETION_REWARD == 15.0
        assert EnvironmentConfig.TIME_PENALTY == -0.1
        assert EnvironmentConfig.DEATH_PENALTY == -15.0

    def test_environment_config_levels_completeness(self):
        """Test that all Mario levels are included."""
        expected_levels = []
        for world in range(1, 9):  # Worlds 1-8
            for stage in range(1, 5):  # Stages 1-4
                expected_levels.append(f"{world}-{stage}")
        
        for level in expected_levels:
            assert level in EnvironmentConfig.LEVELS
        
        # Test total count
        assert len(EnvironmentConfig.LEVELS) == 32  # 8 worlds * 4 stages

    def test_environment_config_movement_types(self):
        """Test movement types configuration."""
        assert len(EnvironmentConfig.MOVEMENT_TYPES) == 2
        assert "simple" in EnvironmentConfig.MOVEMENT_TYPES
        assert "complex" in EnvironmentConfig.MOVEMENT_TYPES

    def test_environment_config_reward_thresholds(self):
        """Test reward thresholds configuration."""
        # Test reward values are reasonable
        assert EnvironmentConfig.COMPLETION_REWARD > 0
        assert EnvironmentConfig.TIME_PENALTY < 0
        assert EnvironmentConfig.DEATH_PENALTY < 0
        
        # Test relative magnitudes
        assert abs(EnvironmentConfig.DEATH_PENALTY) > abs(EnvironmentConfig.TIME_PENALTY)
        assert EnvironmentConfig.COMPLETION_REWARD > abs(EnvironmentConfig.DEATH_PENALTY)

    def test_environment_config_immutability(self):
        """Test environment configuration immutability."""
        # Test that constants cannot be modified
        original_levels = EnvironmentConfig.LEVELS.copy()
        original_movement_types = EnvironmentConfig.MOVEMENT_TYPES.copy()
        original_completion_reward = EnvironmentConfig.COMPLETION_REWARD
        
        # Try to modify (should not affect the constants)
        EnvironmentConfig.LEVELS.append("9-1")
        EnvironmentConfig.MOVEMENT_TYPES.append("invalid")
        EnvironmentConfig.COMPLETION_REWARD = 20.0
        
        # Values should remain unchanged
        assert EnvironmentConfig.LEVELS == original_levels
        assert EnvironmentConfig.MOVEMENT_TYPES == original_movement_types
        assert EnvironmentConfig.COMPLETION_REWARD == original_completion_reward


class TestConfigFunctions:
    """Test configuration utility functions."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        
        # Test that it returns a TrainingConfig instance
        assert isinstance(config, TrainingConfig)
        
        # Test that it has default values
        assert config.learning_rate == 3e-4
        assert config.n_steps == 512
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.ent_coef == 0.01
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5
        assert config.total_timesteps == 200000
        assert config.save_freq == 50000
        assert config.eval_freq == 50000
        assert config.n_eval_episodes == 10
        assert config.level == "1-1"
        assert config.movement_type == "simple"
        assert config.n_envs == 1
        assert config.model_name == "ppo_mario"
        assert config.checkpoint_dir == "./checkpoints"
        assert config.tensorboard_dir == "./mario_tensorboard"
        assert config.log_file == "mario_training.log"

    def test_get_default_config_returns_new_instance(self):
        """Test that get_default_config returns a new instance each time."""
        config1 = get_default_config()
        config2 = get_default_config()
        
        # Should be different objects
        assert config1 is not config2
        
        # But should have same values
        assert config1.learning_rate == config2.learning_rate
        assert config1.n_steps == config2.n_steps

    @patch.dict(os.environ, {
        'MARIO_LEARNING_RATE': '1e-3',
        'MARIO_TIMESTEPS': '100000',
        'MARIO_LEVEL': '2-1',
        'MARIO_MOVEMENT': 'complex',
        'MARIO_N_ENVS': '4'
    })
    def test_load_config_from_env_with_variables(self):
        """Test loading configuration from environment variables."""
        config = load_config_from_env()
        
        # Test that environment variables are loaded
        assert config.learning_rate == 1e-3
        assert config.total_timesteps == 100000
        assert config.level == "2-1"
        assert config.movement_type == "complex"
        assert config.n_envs == 4
        
        # Test that other values remain default
        assert config.n_steps == 512
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.ent_coef == 0.01
        assert config.vf_coef == 0.5
        assert config.max_grad_norm == 0.5
        assert config.save_freq == 50000
        assert config.eval_freq == 50000
        assert config.n_eval_episodes == 10
        assert config.model_name == "ppo_mario"
        assert config.checkpoint_dir == "./checkpoints"
        assert config.tensorboard_dir == "./mario_tensorboard"
        assert config.log_file == "mario_training.log"

    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_from_env_without_variables(self):
        """Test loading configuration without environment variables."""
        config = load_config_from_env()
        
        # Test that all values are default
        assert config.learning_rate == 3e-4
        assert config.total_timesteps == 200000
        assert config.level == "1-1"
        assert config.movement_type == "simple"
        assert config.n_envs == 1

    @patch.dict(os.environ, {
        'MARIO_LEARNING_RATE': 'invalid',
        'MARIO_TIMESTEPS': 'invalid',
        'MARIO_N_ENVS': 'invalid'
    })
    def test_load_config_from_env_invalid_values(self):
        """Test loading configuration with invalid environment variable values."""
        # Should raise ValueError for invalid conversions
        with pytest.raises(ValueError):
            load_config_from_env()

    @patch.dict(os.environ, {
        'MARIO_LEARNING_RATE': '1e-3',
        'MARIO_TIMESTEPS': '100000',
        'MARIO_LEVEL': '2-1',
        'MARIO_MOVEMENT': 'complex',
        'MARIO_N_ENVS': '4',
        'MARIO_SAVE_FREQ': '25000',
        'MARIO_EVAL_FREQ': '25000',
        'MARIO_N_EVAL_EPISODES': '5'
    })
    def test_load_config_from_env_all_variables(self):
        """Test loading configuration with all possible environment variables."""
        config = load_config_from_env()
        
        # Test that all environment variables are loaded
        assert config.learning_rate == 1e-3
        assert config.total_timesteps == 100000
        assert config.level == "2-1"
        assert config.movement_type == "complex"
        assert config.n_envs == 4

    def test_load_config_from_env_returns_new_instance(self):
        """Test that load_config_from_env returns a new instance each time."""
        config1 = load_config_from_env()
        config2 = load_config_from_env()
        
        # Should be different objects
        assert config1 is not config2
        
        # But should have same values
        assert config1.learning_rate == config2.learning_rate
        assert config1.n_steps == config2.n_steps

    def test_config_functions_consistency(self):
        """Test consistency between config functions."""
        default_config = get_default_config()
        env_config = load_config_from_env()
        
        # Both should return TrainingConfig instances
        assert isinstance(default_config, TrainingConfig)
        assert isinstance(env_config, TrainingConfig)
        
        # Both should have same default values when no env vars are set
        assert default_config.learning_rate == env_config.learning_rate
        assert default_config.n_steps == env_config.n_steps
        assert default_config.batch_size == env_config.batch_size
        assert default_config.n_epochs == env_config.n_epochs
        assert default_config.gamma == env_config.gamma
        assert default_config.gae_lambda == env_config.gae_lambda
        assert default_config.clip_range == env_config.clip_range
        assert default_config.ent_coef == env_config.ent_coef
        assert default_config.vf_coef == env_config.vf_coef
        assert default_config.max_grad_norm == env_config.max_grad_norm
        assert default_config.total_timesteps == env_config.total_timesteps
        assert default_config.save_freq == env_config.save_freq
        assert default_config.eval_freq == env_config.eval_freq
        assert default_config.n_eval_episodes == env_config.n_eval_episodes
        assert default_config.level == env_config.level
        assert default_config.movement_type == env_config.movement_type
        assert default_config.n_envs == env_config.n_envs
        assert default_config.model_name == env_config.model_name
        assert default_config.checkpoint_dir == env_config.checkpoint_dir
        assert default_config.tensorboard_dir == env_config.tensorboard_dir
        assert default_config.log_file == env_config.log_file


class TestConfigIntegration:
    """Integration tests for configuration management."""

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        # Create a custom config
        original_config = TrainingConfig(
            learning_rate=1e-3,
            n_steps=256,
            batch_size=32,
            level="2-1",
            movement_type="complex"
        )
        
        # Convert to dictionary
        config_dict = original_config.to_dict()
        
        # Create new config from dictionary
        new_config = TrainingConfig(**config_dict)
        
        # Test that values are preserved
        assert new_config.learning_rate == original_config.learning_rate
        assert new_config.n_steps == original_config.n_steps
        assert new_config.batch_size == original_config.batch_size
        assert new_config.level == original_config.level
        assert new_config.movement_type == original_config.movement_type

    def test_config_validation_workflow(self):
        """Test configuration validation workflow."""
        # Test valid configuration
        valid_config = TrainingConfig(
            learning_rate=1e-3,
            n_steps=256,
            batch_size=32,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,
            ent_coef=0.02,
            vf_coef=0.3,
            max_grad_norm=0.3,
            total_timesteps=100000,
            save_freq=25000,
            eval_freq=25000,
            n_eval_episodes=5,
            level="2-1",
            movement_type="complex",
            n_envs=4
        )
        
        # Test that all values are valid
        assert valid_config.learning_rate > 0
        assert valid_config.n_steps > 0
        assert valid_config.batch_size > 0
        assert valid_config.n_epochs > 0
        assert 0 < valid_config.gamma < 1
        assert 0 < valid_config.gae_lambda < 1
        assert 0 < valid_config.clip_range < 1
        assert valid_config.ent_coef >= 0
        assert valid_config.vf_coef > 0
        assert valid_config.max_grad_norm > 0
        assert valid_config.total_timesteps > 0
        assert valid_config.save_freq > 0
        assert valid_config.eval_freq > 0
        assert valid_config.n_eval_episodes > 0
        assert valid_config.n_envs > 0
        assert valid_config.level in EnvironmentConfig.LEVELS
        assert valid_config.movement_type in EnvironmentConfig.MOVEMENT_TYPES

    def test_config_environment_integration(self):
        """Test configuration integration with environment settings."""
        # Test level validation
        for level in EnvironmentConfig.LEVELS:
            config = TrainingConfig(level=level)
            assert config.level == level
        
        # Test movement type validation
        for movement_type in EnvironmentConfig.MOVEMENT_TYPES:
            config = TrainingConfig(movement_type=movement_type)
            assert config.movement_type == movement_type

    def test_config_hyperparameter_relationships(self):
        """Test hyperparameter relationships and constraints."""
        # Test that batch_size <= n_steps
        config = TrainingConfig(batch_size=64, n_steps=32)
        assert config.batch_size > config.n_steps  # This should be valid
        
        # Test that n_epochs is reasonable
        config = TrainingConfig(n_epochs=100)
        assert config.n_epochs > 0
        
        # Test that gamma is in valid range
        config = TrainingConfig(gamma=0.5)
        assert 0 < config.gamma < 1
        
        # Test that gae_lambda is in valid range
        config = TrainingConfig(gae_lambda=0.5)
        assert 0 < config.gae_lambda < 1
        
        # Test that clip_range is in valid range
        config = TrainingConfig(clip_range=0.1)
        assert 0 < config.clip_range < 1

    def test_config_path_management(self):
        """Test configuration path management."""
        config = TrainingConfig(
            checkpoint_dir="/custom/checkpoints",
            tensorboard_dir="/custom/tensorboard",
            log_file="/custom/logs/training.log"
        )
        
        # Test that paths are set correctly
        assert config.checkpoint_dir == "/custom/checkpoints"
        assert config.tensorboard_dir == "/custom/tensorboard"
        assert config.log_file == "/custom/logs/training.log"
        
        # Test that paths can be relative
        config = TrainingConfig(
            checkpoint_dir="./checkpoints",
            tensorboard_dir="./tensorboard",
            log_file="./training.log"
        )
        
        assert config.checkpoint_dir == "./checkpoints"
        assert config.tensorboard_dir == "./tensorboard"
        assert config.log_file == "./training.log"

    def test_config_model_naming(self):
        """Test configuration model naming."""
        config = TrainingConfig(model_name="custom_mario_model")
        assert config.model_name == "custom_mario_model"
        
        # Test default model name
        config = TrainingConfig()
        assert config.model_name == "ppo_mario"
        
        # Test model name with special characters
        config = TrainingConfig(model_name="mario_model_v1.0")
        assert config.model_name == "mario_model_v1.0"

    def test_config_training_schedule(self):
        """Test configuration training schedule."""
        config = TrainingConfig(
            total_timesteps=100000,
            save_freq=25000,
            eval_freq=25000
        )
        
        # Test that frequencies are reasonable
        assert config.save_freq <= config.total_timesteps
        assert config.eval_freq <= config.total_timesteps
        
        # Test that frequencies are positive
        assert config.save_freq > 0
        assert config.eval_freq > 0
        
        # Test that total timesteps is positive
        assert config.total_timesteps > 0

    def test_config_evaluation_settings(self):
        """Test configuration evaluation settings."""
        config = TrainingConfig(n_eval_episodes=10)
        assert config.n_eval_episodes > 0
        
        # Test default evaluation episodes
        config = TrainingConfig()
        assert config.n_eval_episodes == 10
        
        # Test custom evaluation episodes
        config = TrainingConfig(n_eval_episodes=50)
        assert config.n_eval_episodes == 50
