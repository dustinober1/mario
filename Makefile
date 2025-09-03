.PHONY: help install install-dev test test-cov lint format type-check clean build docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run type checking with mypy"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Generate documentation"
	@echo "  pre-commit   - Install pre-commit hooks"
	@echo "  ci           - Run all CI checks locally"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=mario --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:
	black .
	isort .

type-check:
	mypy . --ignore-missing-imports

# Pre-commit
pre-commit:
	pre-commit run --all-files

# CI checks (run locally)
ci: format lint type-check test

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Building
build:
	python -m build

# Documentation
docs:
	# Placeholder for documentation generation
	@echo "Documentation generation not yet implemented"

# Development helpers
run-mario:
	python mario.py

run-evaluate:
	python evaluate.py

run-visualize:
	python visualize.py

# Docker (if needed later)
docker-build:
	docker build -t mario-rl .

docker-run:
	docker run -it mario-rl

# Environment setup
setup-env:
	python -m venv mario_env
	@echo "Virtual environment created. Activate with:"
	@echo "source mario_env/bin/activate  # On Unix/macOS"
	@echo "mario_env\\Scripts\\activate     # On Windows"

# Quick start
quick-start: setup-env install-dev
	@echo "Development environment setup complete!"
	@echo "Activate your virtual environment and run 'make test' to verify installation."
