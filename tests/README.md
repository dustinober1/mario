# Mario RL Test Suite

This directory contains comprehensive tests for the Mario RL project, covering unit tests, integration tests, performance tests, and benchmarks.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and fixtures
├── test_mario.py              # Original basic tests
├── test_environment.py        # Environment tests
├── test_agent.py              # Agent and model tests
├── test_config.py             # Configuration tests
├── test_utils.py              # Utility function tests
├── test_integration.py        # Integration tests
├── test_performance.py        # Performance and benchmark tests
└── README.md                  # This file
```

## Test Categories

### Unit Tests
- **test_environment.py**: Tests for Mario environment functionality
- **test_agent.py**: Tests for Mario agent and PPO model
- **test_config.py**: Tests for configuration management
- **test_utils.py**: Tests for utility functions (logging, plotting)

### Integration Tests
- **test_integration.py**: End-to-end system integration tests
- **test_mario.py**: Basic integration tests

### Performance Tests
- **test_performance.py**: Performance benchmarks and memory usage tests

## Running Tests

### Basic Test Commands

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test categories
make test-unit
make test-integration
make test-performance

# Run fast tests only (skip slow tests)
make test-fast

# Run slow tests only
make test-slow
```

### Advanced Test Commands

```bash
# Run tests in parallel
make test-parallel

# Generate HTML test report
make test-html

# Run performance benchmarks
make test-benchmark

# Run memory profiling
make test-memory

# Run all test categories
make test-all
```

### Pytest Direct Commands

```bash
# Run specific test file
pytest tests/test_environment.py -v

# Run tests with specific markers
pytest -m "unit" -v
pytest -m "integration" -v
pytest -m "performance" -v
pytest -m "slow" -v

# Run tests with coverage
pytest --cov=mario_rl --cov-report=html

# Run tests in parallel
pytest -n auto

# Run tests with HTML report
pytest --html=reports/test_report.html --self-contained-html
```

## Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.mario`: Mario-specific tests

## Test Fixtures

The `conftest.py` file provides comprehensive fixtures for testing:

### Environment Fixtures
- `mock_env`: Mock Mario environment
- `mock_vec_env`: Mock vectorized environment
- `mock_gym_env`: Mock gym environment
- `mock_joypad_env`: Mock joypad environment
- `mock_monitor_env`: Mock monitor environment

### Agent Fixtures
- `mock_model`: Mock PPO model
- `mock_ppo_model`: Mock PPO model with parameters
- `mock_checkpoint_callback`: Mock checkpoint callback
- `mock_eval_callback`: Mock evaluation callback

### Configuration Fixtures
- `training_config`: Training configuration
- `environment_config`: Environment configuration

### Utility Fixtures
- `temp_dir`: Temporary directory
- `temp_log_file`: Temporary log file
- `temp_plot_file`: Temporary plot file
- `sample_training_data`: Sample training data
- `sample_evaluation_data`: Sample evaluation data

### Mock Fixtures
- `mock_logger`: Mock logger
- `mock_matplotlib`: Mock matplotlib
- `mock_seaborn`: Mock seaborn
- `mock_pandas`: Mock pandas
- `mock_numpy`: Mock numpy
- `mock_os`: Mock os module
- `mock_psutil`: Mock psutil
- `mock_time`: Mock time module

## Test Coverage

The test suite aims for comprehensive coverage:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test system performance and memory usage
- **Error Handling**: Test error conditions and recovery
- **Edge Cases**: Test boundary conditions and edge cases

## Test Data

Tests use realistic but controlled data:

- **Observations**: 84x84x4 numpy arrays (Mario screen frames)
- **Actions**: Integer actions (0-6 for simple, 0-11 for complex)
- **Rewards**: Float rewards (-1 to 1 range)
- **Configurations**: Valid training configurations
- **Training Data**: Realistic training progress data

## Performance Benchmarks

Performance tests measure:

- **Environment Creation**: Time to create environments
- **Agent Training**: Training performance (mocked)
- **Memory Usage**: Memory consumption patterns
- **Prediction Speed**: Inference performance
- **Scalability**: Performance with different scales

## Continuous Integration

Tests are designed to run in CI environments:

- **Mocked Dependencies**: External dependencies are mocked
- **Deterministic**: Tests use fixed random seeds
- **Fast Execution**: Most tests run quickly
- **Isolated**: Tests don't depend on external resources
- **Parallelizable**: Tests can run in parallel

## Test Reports

The test suite generates various reports:

- **Coverage Report**: HTML coverage report
- **Test Report**: HTML test results
- **Performance Report**: Benchmark results
- **Memory Report**: Memory usage analysis

## Environment Variables

Control test behavior with environment variables:

- `SKIP_SLOW_TESTS=true`: Skip slow tests
- `SKIP_GPU_TESTS=true`: Skip GPU tests
- `SKIP_INTEGRATION_TESTS=true`: Skip integration tests
- `SKIP_PERFORMANCE_TESTS=true`: Skip performance tests

## Best Practices

### Writing Tests

1. **Use Descriptive Names**: Test names should clearly describe what is being tested
2. **Test One Thing**: Each test should test one specific behavior
3. **Use Fixtures**: Leverage pytest fixtures for common setup
4. **Mock External Dependencies**: Mock external libraries and services
5. **Test Edge Cases**: Include tests for boundary conditions
6. **Test Error Conditions**: Test error handling and recovery

### Test Organization

1. **Group Related Tests**: Use test classes to group related tests
2. **Use Markers**: Mark tests with appropriate pytest markers
3. **Keep Tests Fast**: Most tests should run quickly
4. **Make Tests Deterministic**: Use fixed random seeds
5. **Clean Up**: Ensure tests clean up after themselves

### Performance Testing

1. **Measure What Matters**: Focus on performance-critical operations
2. **Use Realistic Data**: Test with realistic data sizes
3. **Benchmark Consistently**: Use consistent benchmarking methodology
4. **Monitor Memory**: Track memory usage patterns
5. **Test Scalability**: Test performance at different scales

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Issues**: Check that mocks are properly configured
3. **File Permissions**: Ensure test files can be created/deleted
4. **Memory Issues**: Monitor memory usage during tests
5. **Timeout Issues**: Increase timeout for slow tests

### Debugging Tests

```bash
# Run tests with verbose output
pytest -v -s

# Run specific test with debugging
pytest tests/test_environment.py::TestMarioEnvironment::test_environment_creation -v -s

# Run tests with pdb debugging
pytest --pdb

# Run tests with coverage debugging
pytest --cov=mario_rl --cov-report=term-missing
```

## Contributing

When adding new tests:

1. **Follow Naming Conventions**: Use descriptive test names
2. **Add Appropriate Markers**: Mark tests with relevant pytest markers
3. **Update Fixtures**: Add new fixtures to `conftest.py` if needed
4. **Document Tests**: Add docstrings to test functions
5. **Test Edge Cases**: Include tests for edge cases and error conditions
6. **Update Documentation**: Update this README if needed

## Test Metrics

The test suite provides various metrics:

- **Coverage**: Code coverage percentage
- **Performance**: Execution time benchmarks
- **Memory**: Memory usage patterns
- **Reliability**: Test stability and consistency
- **Maintainability**: Test code quality and maintainability
