"""Pytest configuration and common fixtures."""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

from mario_rl.configs import TrainingConfig, EnvironmentConfig


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
def mock_vec_env():
    """Create a mock vectorized environment for testing."""
    env = Mock()
    env.num_envs = 1
    env.observation_space = Mock()
    env.observation_space.shape = (84, 84, 4)
    env.observation_space.dtype = np.uint8
    env.action_space = Mock()
    env.action_space.n = 7
    
    # Mock vectorized methods
    env.reset.return_value = np.random.randint(0, 256, (1, 84, 84, 4), dtype=np.uint8)
    env.step.return_value = (
        np.random.randint(0, 256, (1, 84, 84, 4), dtype=np.uint8),
        np.array([1.0]),
        np.array([False]),
        [{}]
    )
    env.close = Mock()
    
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
    model.predict.return_value = (np.array([0]), None)
    
    return model


@pytest.fixture
def training_config():
    """Create a training configuration for testing."""
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


@pytest.fixture
def environment_config():
    """Create an environment configuration for testing."""
    return EnvironmentConfig()


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file for testing."""
    log_file = tmp_path / "test.log"
    return str(log_file)


@pytest.fixture
def temp_plot_file(tmp_path):
    """Create a temporary plot file for testing."""
    plot_file = tmp_path / "test_plot.png"
    return str(plot_file)


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    return {
        'rewards': [1.0, 2.0, 3.0, 4.0, 5.0],
        'losses': {
            'policy_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
            'value_loss': [0.3, 0.25, 0.2, 0.15, 0.1]
        }
    }


@pytest.fixture
def sample_evaluation_data():
    """Create sample evaluation data for testing."""
    return {
        'rewards': [10.0, 12.0, 8.0, 15.0, 11.0],
        'episode_lengths': [100, 120, 80, 150, 110]
    }


@pytest.fixture
def sample_model_comparison_data():
    """Create sample model comparison data for testing."""
    return {
        'models': {
            'Model A': {'accuracy': 0.85, 'f1_score': 0.82},
            'Model B': {'accuracy': 0.90, 'f1_score': 0.88}
        },
        'metrics': ['accuracy', 'f1_score']
    }


@pytest.fixture
def mock_gym_env():
    """Create a mock gym environment for testing."""
    env = Mock()
    env.observation_space = Mock()
    env.observation_space.shape = (84, 84, 4)
    env.observation_space.dtype = np.uint8
    env.action_space = Mock()
    env.action_space.n = 7
    env.reset.return_value = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    env.step.return_value = (
        np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8),
        1.0,
        False,
        {}
    )
    env.close = Mock()
    return env


@pytest.fixture
def mock_joypad_env():
    """Create a mock joypad environment for testing."""
    env = Mock()
    env.observation_space = Mock()
    env.observation_space.shape = (84, 84, 4)
    env.observation_space.dtype = np.uint8
    env.action_space = Mock()
    env.action_space.n = 7
    env.reset.return_value = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    env.step.return_value = (
        np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8),
        1.0,
        False,
        {}
    )
    env.close = Mock()
    return env


@pytest.fixture
def mock_monitor_env():
    """Create a mock monitor environment for testing."""
    env = Mock()
    env.observation_space = Mock()
    env.observation_space.shape = (84, 84, 4)
    env.observation_space.dtype = np.uint8
    env.action_space = Mock()
    env.action_space.n = 7
    env.reset.return_value = np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8)
    env.step.return_value = (
        np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8),
        1.0,
        False,
        {}
    )
    env.close = Mock()
    return env


@pytest.fixture
def mock_ppo_model():
    """Create a mock PPO model for testing."""
    model = Mock()
    model.learn.return_value = None
    model.save.return_value = None
    model.load.return_value = None
    model.predict.return_value = (np.array([0]), None)
    model.get_parameters.return_value = {
        'learning_rate': 3e-4,
        'n_steps': 64,
        'batch_size': 32
    }
    return model


@pytest.fixture
def mock_checkpoint_callback():
    """Create a mock checkpoint callback for testing."""
    callback = Mock()
    callback.save_freq = 500
    callback.save_path = "./test_checkpoints"
    callback.name_prefix = "test_model"
    return callback


@pytest.fixture
def mock_eval_callback():
    """Create a mock evaluation callback for testing."""
    callback = Mock()
    callback.eval_freq = 500
    callback.n_eval_episodes = 5
    callback.deterministic = True
    callback.render = False
    return callback


@pytest.fixture
def mock_evaluate_policy():
    """Create a mock evaluate_policy function for testing."""
    def mock_evaluate(model, env, n_eval_episodes=10, deterministic=True):
        return 10.5, 2.1
    return mock_evaluate


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.setLevel = Mock()
    logger.addHandler = Mock()
    logger.removeHandler = Mock()
    return logger


@pytest.fixture
def mock_file_handler():
    """Create a mock file handler for testing."""
    handler = Mock()
    handler.setLevel = Mock()
    handler.setFormatter = Mock()
    return handler


@pytest.fixture
def mock_console_handler():
    """Create a mock console handler for testing."""
    handler = Mock()
    handler.setLevel = Mock()
    handler.setFormatter = Mock()
    return handler


@pytest.fixture
def mock_formatter():
    """Create a mock formatter for testing."""
    formatter = Mock()
    formatter._fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    return formatter


@pytest.fixture
def mock_matplotlib():
    """Create a mock matplotlib for testing."""
    with patch('matplotlib.pyplot.show') as mock_show, \
         patch('matplotlib.pyplot.figure') as mock_figure, \
         patch('matplotlib.pyplot.subplots') as mock_subplots, \
         patch('matplotlib.pyplot.plot') as mock_plot, \
         patch('matplotlib.pyplot.hist') as mock_hist, \
         patch('matplotlib.pyplot.barplot') as mock_barplot, \
         patch('matplotlib.pyplot.title') as mock_title, \
         patch('matplotlib.pyplot.xlabel') as mock_xlabel, \
         patch('matplotlib.pyplot.ylabel') as mock_ylabel, \
         patch('matplotlib.pyplot.legend') as mock_legend, \
         patch('matplotlib.pyplot.grid') as mock_grid, \
         patch('matplotlib.pyplot.tight_layout') as mock_tight_layout, \
         patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.close') as mock_close:
        
        yield {
            'show': mock_show,
            'figure': mock_figure,
            'subplots': mock_subplots,
            'plot': mock_plot,
            'hist': mock_hist,
            'barplot': mock_barplot,
            'title': mock_title,
            'xlabel': mock_xlabel,
            'ylabel': mock_ylabel,
            'legend': mock_legend,
            'grid': mock_grid,
            'tight_layout': mock_tight_layout,
            'savefig': mock_savefig,
            'close': mock_close
        }


@pytest.fixture
def mock_seaborn():
    """Create a mock seaborn for testing."""
    with patch('seaborn.barplot') as mock_barplot:
        yield {'barplot': mock_barplot}


@pytest.fixture
def mock_pandas():
    """Create a mock pandas for testing."""
    with patch('pandas.DataFrame') as mock_dataframe, \
         patch('pandas.Series') as mock_series:
        yield {
            'DataFrame': mock_dataframe,
            'Series': mock_series
        }


@pytest.fixture
def mock_numpy():
    """Create a mock numpy for testing."""
    with patch('numpy.mean') as mock_mean, \
         patch('numpy.std') as mock_std, \
         patch('numpy.random') as mock_random:
        yield {
            'mean': mock_mean,
            'std': mock_std,
            'random': mock_random
        }


@pytest.fixture
def mock_os():
    """Create a mock os for testing."""
    with patch('os.path.exists') as mock_exists, \
         patch('os.makedirs') as mock_makedirs, \
         patch('os.path.dirname') as mock_dirname:
        yield {
            'exists': mock_exists,
            'makedirs': mock_makedirs,
            'dirname': mock_dirname
        }


@pytest.fixture
def mock_pathlib():
    """Create a mock pathlib for testing."""
    with patch('pathlib.Path') as mock_path:
        yield {'Path': mock_path}


@pytest.fixture
def mock_tempfile():
    """Create a mock tempfile for testing."""
    with patch('tempfile.TemporaryDirectory') as mock_tempdir, \
         patch('tempfile.mkdtemp') as mock_mkdtemp:
        yield {
            'TemporaryDirectory': mock_tempdir,
            'mkdtemp': mock_mkdtemp
        }


@pytest.fixture
def mock_psutil():
    """Create a mock psutil for testing."""
    with patch('psutil.Process') as mock_process:
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value = Mock(rss=1000000)  # 1MB
        mock_process.return_value = mock_process_instance
        yield {'Process': mock_process}


@pytest.fixture
def mock_time():
    """Create a mock time for testing."""
    with patch('time.time') as mock_time_func:
        mock_time_func.side_effect = [0.0, 1.0]  # Start and end times
        yield {'time': mock_time_func}


@pytest.fixture
def mock_memory_profiler():
    """Create a mock memory_profiler for testing."""
    with patch('memory_profiler.profile') as mock_profile:
        yield {'profile': mock_profile}


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Mock matplotlib to avoid display issues in CI
    with patch('matplotlib.pyplot.show'):
        yield


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test files after each test."""
    yield
    
    # Cleanup any test files that might have been created
    test_files = [
        "./test_checkpoints",
        "./test_tensorboard",
        "./test.log",
        "./test_plot.png"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)


@pytest.fixture
def skip_slow_tests():
    """Skip slow tests if requested."""
    if os.getenv('SKIP_SLOW_TESTS', 'false').lower() == 'true':
        pytest.skip("Slow tests skipped by environment variable")


@pytest.fixture
def skip_gpu_tests():
    """Skip GPU tests if requested."""
    if os.getenv('SKIP_GPU_TESTS', 'false').lower() == 'true':
        pytest.skip("GPU tests skipped by environment variable")


@pytest.fixture
def skip_integration_tests():
    """Skip integration tests if requested."""
    if os.getenv('SKIP_INTEGRATION_TESTS', 'false').lower() == 'true':
        pytest.skip("Integration tests skipped by environment variable")


@pytest.fixture
def skip_performance_tests():
    """Skip performance tests if requested."""
    if os.getenv('SKIP_PERFORMANCE_TESTS', 'false').lower() == 'true':
        pytest.skip("Performance tests skipped by environment variable")


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "mario: marks tests specific to Mario environment")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")


# Test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add slow marker to performance and integration tests
        if "performance" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add unit marker to unit tests
        if "test_" in item.nodeid and "integration" not in item.nodeid and "performance" not in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Add mario marker to mario-specific tests
        if "mario" in item.nodeid.lower():
            item.add_marker(pytest.mark.mario)


# Test reporting
def pytest_html_report_title(report):
    """Set the title of the HTML report."""
    report.title = "Mario RL Test Report"


def pytest_html_results_summary(prefix, summary, postfix):
    """Customize the HTML report summary."""
    prefix.extend([
        "<h2>Test Summary</h2>",
        "<p>This report contains test results for the Mario RL project.</p>"
    ])
