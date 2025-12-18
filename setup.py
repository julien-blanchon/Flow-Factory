#!/usr/bin/env python
# setup.py
"""
Flow-Factory: Unified RL Fine-tuning Framework for Diffusion Models
"""
from setuptools import setup, find_packages

# Read requirements
def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="flow-factory",
    version="0.1.0",
    description="Unified RL Fine-tuning Framework for Diffusion/Flow-Matching Models",
    author="Flow-Factory Team",
    author_email="",
    url="https://github.com/your-org/flow-factory",
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.28.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "datasets>=2.14.0",
        "pyyaml>=6.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "reward": [
            "ImageReward>=1.0.0",
        ],
    },
    
    # Entry points - command line scripts
    entry_points={
        "console_scripts": [
            "flow-factory-train=flow_factory.cli:train_cli",
            "flow-factory-eval=flow_factory.cli:eval_cli",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Include package data
    include_package_data=True,
    zip_safe=False,
)