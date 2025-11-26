#!/bin/bash
#
# Author: s Bostan
# Created on: Nov, 2025
#
# Setup script for Python environment

echo "Setting up Python environment for AdaptiveMultimodalRAG..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Python environment setup complete!"
echo "To activate the environment, run: source venv/bin/activate"

