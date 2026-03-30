#!/usr/bin/env bash

set -e

VENV_DIR="venv"

echo "Setting up virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created new virtual environment in ./$VENV_DIR"
else
    echo "Virtual environment already exists in ./$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

echo "Upgrading pip & setuptools..."
pip install --upgrade pip setuptools

echo "Installing project via setup tools (editable mode)..."
# This will install according to what's defined in setup.py / pyproject.toml
pip install -e .

echo "=========================================================="
echo "Installation complete!"
echo "Your virtual environment is ready."
echo "To activate it in your shell, run: source $VENV_DIR/bin/activate"
echo "=========================================================="
