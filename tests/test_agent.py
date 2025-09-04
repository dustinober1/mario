"""Comprehensive tests for Mario agent and model functionality."""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from mario_rl.agents import MarioAgent
from mario_rl.models import PPOModel
from mario_rl.configs import TrainingConfig


class TestMarioAgent:
    """Test Mario agent functionality."""

    @pytest.fixture
    def mock_vec_env(self):
        """Create a mock vectorized environment."""
        env = Mock(spec=DummyVecEnv)
        env.num_envs = 1
        env.observation_space = Mock()
        env.action_space = Mock()
        env.reset.return_value = np.random.randint(0, 256, (1, 84, 84, 4), dtype=np.uint8)
        env.step.return_value = (
            np.random.randint(0, 256, (1, 84, 84, 4), dtype=np.uint8),
            np.array([1.0]),
            np.array([False]),
            [{}]
        )
        return env

    @pytest.fixture
    def training_config(self):
        """Create a training configuration."""
        return TrainingConfig(
            level="1-1",
            movement_type="simple",
            total_timesteps=1000,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            save_freq=500,
            eval_freq=500,
            n_eval_episodes=5,
            checkpoint_dir="./test_checkpoints",
            tensorboard_dir="./test_tensorboard"
        )

    def test_agent_initialization(self, mock_vec_env, training_config):
        """Test agent initialization."""
        agent = MarioAgent(mock_vec_env, training_config)
        
        assert agent.env == mock_vec_env
        assert agent.config == training_config
        assert agent.model is None
        assert agent.training_history == []

    def test_agent_initialization_invalid_env(self, training_config):
        """Test agent initialization with invalid environment."""
        with pytest.raises(TypeError):
            MarioAgent(None, training_config)

    def test_agent_initialization_invalid_config(self, mock_vec_env):
        """Test agent initialization with invalid config."""
        with pytest.raises(TypeError):
            MarioAgent(mock_vec_env, None)

    @patch('mario_rl.agents.PPO')
    def test_create_model_success(self, mock_ppo_class, mock_vec_env, training_config):
        """Test successful model creation."""
        # Setup mock
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        model = agent.create_model()
        
        # Verify model creation
        assert model == mock_model
        assert agent.model == mock_model
        mock_ppo_class.assert_called_once()
        
        # Verify PPO parameters
        call_args = mock_ppo_class.call_args
        assert call_args[1]['policy'] == 'CnnPolicy'
        assert call_args[1]['env'] == mock_vec_env
        assert call_args[1]['learning_rate'] == training_config.learning_rate
        assert call_args[1]['n_steps'] == training_config.n_steps
        assert call_args[1]['batch_size'] == training_config.batch_size
        assert call_args[1]['n_epochs'] == training_config.n_epochs
        assert call_args[1]['gamma'] == training_config.gamma
        assert call_args[1]['gae_lambda'] == training_config.gae_lambda
        assert call_args[1]['clip_range'] == training_config.clip_range
        assert call_args[1]['ent_coef'] == training_config.ent_coef
        assert call_args[1]['vf_coef'] == training_config.vf_coef
        assert call_args[1]['max_grad_norm'] == training_config.max_grad_norm
        assert call_args[1]['tensorboard_log'] == training_config.tensorboard_dir

    @patch('mario_rl.agents.PPO')
    def test_create_model_with_kwargs(self, mock_ppo_class, mock_vec_env, training_config):
        """Test model creation with additional kwargs."""
        # Setup mock
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model with additional kwargs
        agent = MarioAgent(mock_vec_env, training_config)
        model = agent.create_model(verbose=0, device='cpu')
        
        # Verify additional parameters
        call_args = mock_ppo_class.call_args
        assert call_args[1]['verbose'] == 0
        assert call_args[1]['device'] == 'cpu'

    @patch('mario_rl.agents.PPO')
    def test_create_model_failure(self, mock_ppo_class, mock_vec_env, training_config):
        """Test model creation failure."""
        # Setup mock to raise exception
        mock_ppo_class.side_effect = Exception("Model creation failed")
        
        agent = MarioAgent(mock_vec_env, training_config)
        
        with pytest.raises(Exception, match="Model creation failed"):
            agent.create_model()

    def test_train_without_model(self, mock_vec_env, training_config):
        """Test training without creating model first."""
        agent = MarioAgent(mock_vec_env, training_config)
        
        with pytest.raises(ValueError, match="Model not created"):
            agent.train()

    @patch('mario_rl.agents.PPO')
    def test_train_success(self, mock_ppo_class, mock_vec_env, training_config):
        """Test successful training."""
        # Setup mock
        mock_model = Mock()
        mock_model.learn.return_value = None
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        # Train
        agent.train()
        
        # Verify training call
        mock_model.learn.assert_called_once()
        call_args = mock_model.learn.call_args
        assert call_args[1]['total_timesteps'] == training_config.total_timesteps
        assert call_args[1]['progress_bar'] == True
        assert 'callback' in call_args[1]

    @patch('mario_rl.agents.PPO')
    def test_train_with_custom_timesteps(self, mock_ppo_class, mock_vec_env, training_config):
        """Test training with custom timesteps."""
        # Setup mock
        mock_model = Mock()
        mock_model.learn.return_value = None
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        # Train with custom timesteps
        custom_timesteps = 2000
        agent.train(total_timesteps=custom_timesteps)
        
        # Verify training call
        call_args = mock_model.learn.call_args
        assert call_args[1]['total_timesteps'] == custom_timesteps

    @patch('mario_rl.agents.PPO')
    def test_train_failure(self, mock_ppo_class, mock_vec_env, training_config):
        """Test training failure."""
        # Setup mock
        mock_model = Mock()
        mock_model.learn.side_effect = Exception("Training failed")
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        with pytest.raises(Exception, match="Training failed"):
            agent.train()

    @patch('mario_rl.agents.PPO')
    @patch('mario_rl.agents.CheckpointCallback')
    @patch('mario_rl.agents.EvalCallback')
    @patch('os.makedirs')
    def test_create_callbacks(self, mock_makedirs, mock_eval_callback, mock_checkpoint_callback, mock_ppo_class, mock_vec_env, training_config):
        """Test callback creation."""
        # Setup mocks
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        mock_checkpoint_callback_instance = Mock()
        mock_checkpoint_callback.return_value = mock_checkpoint_callback_instance
        
        mock_eval_callback_instance = Mock()
        mock_eval_callback.return_value = mock_eval_callback_instance
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        # Test callback creation
        callbacks = agent._create_callbacks()
        
        # Verify callbacks
        assert len(callbacks) == 2
        assert mock_checkpoint_callback_instance in callbacks
        assert mock_eval_callback_instance in callbacks
        
        # Verify checkpoint callback parameters
        mock_checkpoint_callback.assert_called_once()
        checkpoint_args = mock_checkpoint_callback.call_args
        assert checkpoint_args[1]['save_freq'] == training_config.save_freq
        assert checkpoint_args[1]['save_path'] == training_config.checkpoint_dir
        assert checkpoint_args[1]['name_prefix'] == training_config.model_name
        
        # Verify eval callback parameters
        mock_eval_callback.assert_called_once()
        eval_args = mock_eval_callback.call_args
        assert eval_args[0][0] == mock_vec_env
        assert eval_args[1]['best_model_save_path'] == training_config.checkpoint_dir
        assert eval_args[1]['log_path'] == training_config.checkpoint_dir
        assert eval_args[1]['eval_freq'] == training_config.eval_freq
        assert eval_args[1]['n_eval_episodes'] == training_config.n_eval_episodes
        assert eval_args[1]['deterministic'] == True
        assert eval_args[1]['render'] == False

    def test_evaluate_without_model(self, mock_vec_env, training_config):
        """Test evaluation without creating model first."""
        agent = MarioAgent(mock_vec_env, training_config)
        
        with pytest.raises(ValueError, match="Model not created"):
            agent.evaluate()

    @patch('mario_rl.agents.PPO')
    @patch('mario_rl.agents.evaluate_policy')
    def test_evaluate_success(self, mock_evaluate_policy, mock_ppo_class, mock_vec_env, training_config):
        """Test successful evaluation."""
        # Setup mocks
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        mock_evaluate_policy.return_value = (10.5, 2.1)
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        # Evaluate
        results = agent.evaluate(n_eval_episodes=10)
        
        # Verify evaluation
        assert results['mean_reward'] == 10.5
        assert results['std_reward'] == 2.1
        assert results['n_episodes'] == 10
        
        # Verify evaluate_policy call
        mock_evaluate_policy.assert_called_once()
        eval_args = mock_evaluate_policy.call_args
        assert eval_args[0][0] == mock_model
        assert eval_args[0][1] == mock_vec_env
        assert eval_args[1]['n_eval_episodes'] == 10
        assert eval_args[1]['deterministic'] == True

    @patch('mario_rl.agents.PPO')
    @patch('mario_rl.agents.evaluate_policy')
    def test_evaluate_failure(self, mock_evaluate_policy, mock_ppo_class, mock_vec_env, training_config):
        """Test evaluation failure."""
        # Setup mocks
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        mock_evaluate_policy.side_effect = Exception("Evaluation failed")
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        with pytest.raises(Exception, match="Evaluation failed"):
            agent.evaluate()

    def test_save_model_without_model(self, mock_vec_env, training_config):
        """Test saving model without creating model first."""
        agent = MarioAgent(mock_vec_env, training_config)
        
        with pytest.raises(ValueError, match="Model not created"):
            agent.save_model()

    @patch('mario_rl.agents.PPO')
    def test_save_model_success(self, mock_ppo_class, mock_vec_env, training_config):
        """Test successful model saving."""
        # Setup mock
        mock_model = Mock()
        mock_model.save.return_value = None
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        # Save model
        agent.save_model()
        
        # Verify save call
        mock_model.save.assert_called_once()
        save_path = mock_model.save.call_args[0][0]
        assert save_path.startswith(training_config.checkpoint_dir)
        assert training_config.model_name in save_path

    @patch('mario_rl.agents.PPO')
    def test_save_model_custom_path(self, mock_ppo_class, mock_vec_env, training_config):
        """Test saving model with custom path."""
        # Setup mock
        mock_model = Mock()
        mock_model.save.return_value = None
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        # Save model with custom path
        custom_path = "/custom/path/model.zip"
        agent.save_model(path=custom_path)
        
        # Verify save call
        mock_model.save.assert_called_once_with(custom_path)

    @patch('mario_rl.agents.PPO')
    def test_save_model_failure(self, mock_ppo_class, mock_vec_env, training_config):
        """Test model saving failure."""
        # Setup mock
        mock_model = Mock()
        mock_model.save.side_effect = Exception("Save failed")
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        with pytest.raises(Exception, match="Save failed"):
            agent.save_model()

    @patch('mario_rl.agents.PPO')
    def test_load_model_success(self, mock_ppo_class, mock_vec_env, training_config):
        """Test successful model loading."""
        # Setup mock
        mock_model = Mock()
        mock_ppo_class.load.return_value = mock_model
        
        # Create agent
        agent = MarioAgent(mock_vec_env, training_config)
        
        # Load model
        model_path = "/path/to/model.zip"
        agent.load_model(model_path)
        
        # Verify load call
        mock_ppo_class.load.assert_called_once_with(model_path, env=mock_vec_env)
        assert agent.model == mock_model

    @patch('mario_rl.agents.PPO')
    def test_load_model_failure(self, mock_ppo_class, mock_vec_env, training_config):
        """Test model loading failure."""
        # Setup mock
        mock_ppo_class.load.side_effect = Exception("Load failed")
        
        # Create agent
        agent = MarioAgent(mock_vec_env, training_config)
        
        with pytest.raises(Exception, match="Load failed"):
            agent.load_model("/path/to/model.zip")

    def test_predict_without_model(self, mock_vec_env, training_config):
        """Test prediction without creating model first."""
        agent = MarioAgent(mock_vec_env, training_config)
        
        with pytest.raises(ValueError, match="Model not created"):
            agent.predict(np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8))

    @patch('mario_rl.agents.PPO')
    def test_predict_success(self, mock_ppo_class, mock_vec_env, training_config):
        """Test successful prediction."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        # Make prediction
        observation = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        action, state = agent.predict(observation, deterministic=True)
        
        # Verify prediction
        assert action is not None
        assert state is None
        mock_model.predict.assert_called_once_with(observation, deterministic=True)

    @patch('mario_rl.agents.PPO')
    def test_predict_deterministic_false(self, mock_ppo_class, mock_vec_env, training_config):
        """Test prediction with deterministic=False."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)
        mock_ppo_class.return_value = mock_model
        
        # Create agent and model
        agent = MarioAgent(mock_vec_env, training_config)
        agent.create_model()
        
        # Make prediction
        observation = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        agent.predict(observation, deterministic=False)
        
        # Verify prediction call
        mock_model.predict.assert_called_once_with(observation, deterministic=False)

    def test_get_training_history(self, mock_vec_env, training_config):
        """Test getting training history."""
        agent = MarioAgent(mock_vec_env, training_config)
        
        # Add some training history
        agent.training_history = [
            {'episode': 1, 'reward': 10.0},
            {'episode': 2, 'reward': 15.0}
        ]
        
        # Get training history
        history = agent.get_training_history()
        
        # Verify history
        assert len(history) == 2
        assert history[0]['episode'] == 1
        assert history[0]['reward'] == 10.0
        assert history[1]['episode'] == 2
        assert history[1]['reward'] == 15.0
        
        # Verify it's a copy
        assert history is not agent.training_history

    def test_agent_state_management(self, mock_vec_env, training_config):
        """Test agent state management."""
        agent = MarioAgent(mock_vec_env, training_config)
        
        # Initial state
        assert agent.model is None
        assert agent.training_history == []
        
        # Add training history
        agent.training_history.append({'episode': 1, 'reward': 10.0})
        assert len(agent.training_history) == 1
        
        # Get history (should be a copy)
        history = agent.get_training_history()
        history.append({'episode': 2, 'reward': 15.0})
        assert len(agent.training_history) == 1  # Original unchanged
        assert len(history) == 2  # Copy modified


class TestPPOModel:
    """Test PPO model functionality."""

    @pytest.fixture
    def mock_vec_env(self):
        """Create a mock vectorized environment."""
        env = Mock(spec=DummyVecEnv)
        env.num_envs = 1
        env.observation_space = Mock()
        env.action_space = Mock()
        return env

    @patch('mario_rl.models.PPO')
    def test_ppo_model_initialization(self, mock_ppo_class, mock_vec_env):
        """Test PPO model initialization."""
        # Setup mock
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        # Create PPO model
        kwargs = {'learning_rate': 3e-4, 'n_steps': 64}
        ppo_model = PPOModel(mock_vec_env, **kwargs)
        
        # Verify initialization
        assert ppo_model.env == mock_vec_env
        assert ppo_model.model == mock_model
        assert ppo_model.training_config == kwargs
        
        # Verify PPO creation
        mock_ppo_class.assert_called_once_with("CnnPolicy", mock_vec_env, **kwargs)

    @patch('mario_rl.models.PPO')
    def test_ppo_model_train(self, mock_ppo_class, mock_vec_env):
        """Test PPO model training."""
        # Setup mock
        mock_model = Mock()
        mock_model.learn.return_value = None
        mock_ppo_class.return_value = mock_model
        
        # Create PPO model
        ppo_model = PPOModel(mock_vec_env, learning_rate=3e-4)
        
        # Train
        ppo_model.train(total_timesteps=1000)
        
        # Verify training call
        mock_model.learn.assert_called_once_with(total_timesteps=1000)

    @patch('mario_rl.models.PPO')
    def test_ppo_model_train_with_kwargs(self, mock_ppo_class, mock_vec_env):
        """Test PPO model training with additional kwargs."""
        # Setup mock
        mock_model = Mock()
        mock_model.learn.return_value = None
        mock_ppo_class.return_value = mock_model
        
        # Create PPO model
        ppo_model = PPOModel(mock_vec_env, learning_rate=3e-4)
        
        # Train with additional kwargs
        ppo_model.train(total_timesteps=1000, progress_bar=True)
        
        # Verify training call
        mock_model.learn.assert_called_once_with(total_timesteps=1000, progress_bar=True)

    @patch('mario_rl.models.PPO')
    def test_ppo_model_train_failure(self, mock_ppo_class, mock_vec_env):
        """Test PPO model training failure."""
        # Setup mock
        mock_model = Mock()
        mock_model.learn.side_effect = Exception("Training failed")
        mock_ppo_class.return_value = mock_model
        
        # Create PPO model
        ppo_model = PPOModel(mock_vec_env, learning_rate=3e-4)
        
        with pytest.raises(Exception, match="Training failed"):
            ppo_model.train(total_timesteps=1000)

    @patch('mario_rl.models.PPO')
    def test_ppo_model_predict(self, mock_ppo_class, mock_vec_env):
        """Test PPO model prediction."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)
        mock_ppo_class.return_value = mock_model
        
        # Create PPO model
        ppo_model = PPOModel(mock_vec_env, learning_rate=3e-4)
        
        # Make prediction
        observation = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        action, state = ppo_model.predict(observation, deterministic=True)
        
        # Verify prediction
        assert action is not None
        assert state is None
        mock_model.predict.assert_called_once_with(observation, deterministic=True)

    @patch('mario_rl.models.PPO')
    def test_ppo_model_save(self, mock_ppo_class, mock_vec_env):
        """Test PPO model saving."""
        # Setup mock
        mock_model = Mock()
        mock_model.save.return_value = None
        mock_ppo_class.return_value = mock_model
        
        # Create PPO model
        ppo_model = PPOModel(mock_vec_env, learning_rate=3e-4)
        
        # Save model
        save_path = "/path/to/model.zip"
        ppo_model.save(save_path)
        
        # Verify save call
        mock_model.save.assert_called_once_with(save_path)

    @patch('mario_rl.models.PPO')
    def test_ppo_model_load(self, mock_ppo_class, mock_vec_env):
        """Test PPO model loading."""
        # Setup mock
        mock_model = Mock()
        mock_ppo_class.load.return_value = mock_model
        
        # Create PPO model
        ppo_model = PPOModel(mock_vec_env, learning_rate=3e-4)
        
        # Load model
        load_path = "/path/to/model.zip"
        ppo_model.load(load_path)
        
        # Verify load call
        mock_ppo_class.load.assert_called_once_with(load_path, env=mock_vec_env)
        assert ppo_model.model == mock_model

    @patch('mario_rl.models.PPO')
    def test_ppo_model_get_parameters(self, mock_ppo_class, mock_vec_env):
        """Test getting PPO model parameters."""
        # Setup mock
        mock_model = Mock()
        mock_ppo_class.return_value = mock_model
        
        # Create PPO model
        kwargs = {'learning_rate': 3e-4, 'n_steps': 64, 'batch_size': 32}
        ppo_model = PPOModel(mock_vec_env, **kwargs)
        
        # Get parameters
        parameters = ppo_model.get_parameters()
        
        # Verify parameters
        assert parameters == kwargs
        assert parameters is not ppo_model.training_config  # Should be a copy

    def test_ppo_model_parameter_immutability(self, mock_vec_env):
        """Test that PPO model parameters are immutable."""
        with patch('mario_rl.models.PPO') as mock_ppo_class:
            # Setup mock
            mock_model = Mock()
            mock_ppo_class.return_value = mock_model
            
            # Create PPO model
            kwargs = {'learning_rate': 3e-4, 'n_steps': 64}
            ppo_model = PPOModel(mock_vec_env, **kwargs)
            
            # Get parameters and modify
            parameters = ppo_model.get_parameters()
            parameters['learning_rate'] = 1e-3
            
            # Original should be unchanged
            assert ppo_model.training_config['learning_rate'] == 3e-4
            assert parameters['learning_rate'] == 1e-3


class TestAgentIntegration:
    """Integration tests for agent functionality."""

    @pytest.mark.slow
    @patch('mario_rl.agents.PPO')
    @patch('mario_rl.agents.CheckpointCallback')
    @patch('mario_rl.agents.EvalCallback')
    @patch('mario_rl.agents.evaluate_policy')
    @patch('os.makedirs')
    def test_full_agent_workflow(self, mock_makedirs, mock_evaluate_policy, mock_eval_callback, mock_checkpoint_callback, mock_ppo_class):
        """Test complete agent workflow."""
        # Setup mocks
        mock_vec_env = Mock(spec=DummyVecEnv)
        mock_vec_env.num_envs = 1
        mock_vec_env.observation_space = Mock()
        mock_vec_env.action_space = Mock()
        
        mock_model = Mock()
        mock_model.learn.return_value = None
        mock_model.save.return_value = None
        mock_model.predict.return_value = (np.array([0]), None)
        mock_ppo_class.return_value = mock_model
        
        mock_evaluate_policy.return_value = (10.5, 2.1)
        
        mock_checkpoint_callback_instance = Mock()
        mock_checkpoint_callback.return_value = mock_checkpoint_callback_instance
        
        mock_eval_callback_instance = Mock()
        mock_eval_callback.return_value = mock_eval_callback_instance
        
        # Create configuration
        config = TrainingConfig(
            level="1-1",
            movement_type="simple",
            total_timesteps=1000,
            learning_rate=3e-4,
            checkpoint_dir="./test_checkpoints",
            tensorboard_dir="./test_tensorboard"
        )
        
        # Create agent
        agent = MarioAgent(mock_vec_env, config)
        
        # Test workflow
        # 1. Create model
        model = agent.create_model()
        assert model == mock_model
        
        # 2. Train
        agent.train()
        mock_model.learn.assert_called_once()
        
        # 3. Evaluate
        results = agent.evaluate(n_eval_episodes=5)
        assert results['mean_reward'] == 10.5
        assert results['std_reward'] == 2.1
        assert results['n_episodes'] == 5
        
        # 4. Save model
        agent.save_model()
        mock_model.save.assert_called_once()
        
        # 5. Make prediction
        observation = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        action, state = agent.predict(observation)
        assert action is not None
        assert state is None
        
        # 6. Get training history
        history = agent.get_training_history()
        assert isinstance(history, list)

    @pytest.mark.slow
    def test_agent_error_recovery(self):
        """Test agent error recovery and resilience."""
        with patch('mario_rl.agents.PPO') as mock_ppo_class:
            # Setup mocks
            mock_vec_env = Mock(spec=DummyVecEnv)
            mock_vec_env.num_envs = 1
            mock_vec_env.observation_space = Mock()
            mock_vec_env.action_space = Mock()
            
            config = TrainingConfig(
                level="1-1",
                movement_type="simple",
                total_timesteps=1000
            )
            
            # Test model creation failure
            mock_ppo_class.side_effect = Exception("Model creation failed")
            agent = MarioAgent(mock_vec_env, config)
            
            with pytest.raises(Exception, match="Model creation failed"):
                agent.create_model()
            
            # Test recovery - create model successfully
            mock_model = Mock()
            mock_model.learn.return_value = None
            mock_ppo_class.side_effect = None
            mock_ppo_class.return_value = mock_model
            
            model = agent.create_model()
            assert model == mock_model
            
            # Test training failure
            mock_model.learn.side_effect = Exception("Training failed")
            
            with pytest.raises(Exception, match="Training failed"):
                agent.train()
            
            # Test recovery - train successfully
            mock_model.learn.side_effect = None
            
            agent.train()
            mock_model.learn.assert_called()
