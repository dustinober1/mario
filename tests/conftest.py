"""Pytest configuration and common fixtures."""

import pytest
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture
def mock_env():
    """Create a mock Mario environment for testing."""
    env = Mock()
    env.action_space = Mock()
    env.action_space.n = 7
    env.observation_space = Mock()
    env.observation_space.shape = (84, 84, 4)
    env.observation_space.dtype = np.uint8
    
    # Mock reset method
    env.reset.return_value = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    
    # Mock step method
    def mock_step(action):
        obs = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
        reward = np.random.uniform(-1, 1)
        done = np.random.choice([True, False], p=[0.1, 0.9])
        info = {}
        return obs, reward, done, info
    
    env.step.side_effect = mock_step
    
    return env


@pytest.fixture
def sample_observation():
    """Create a sample observation for testing."""
    return np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)


@pytest.fixture
def sample_action():
    """Create a sample action for testing."""
    return np.random.randint(0, 7)


@pytest.fixture
def mock_model():
    """Create a mock PPO model for testing."""
    model = Mock()
    model.learn.return_value = None
    model.save.return_value = None
    model.load.return_value = None
    
    return model


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Mock matplotlib to avoid display issues in CI
    with patch('matplotlib.pyplot.show'):
        yield
