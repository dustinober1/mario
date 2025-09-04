"""Setup script for Mario RL project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mario-rl",
    version="1.0.0",
    author="Reinforcement Learning Engineer",
    author_email="developer@example.com",
    description="Deep Reinforcement Learning agent for Super Mario Bros",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dustinober1/mario",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Arcade",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.8",
            "pre-commit>=2.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "mario-rl=mario_rl.cli.main:main",
        ],
    },
)