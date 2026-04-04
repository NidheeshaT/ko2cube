#!/usr/bin/env bash

set -e

VENV_DIR="venv"

echo "Setting up virtual environment..."

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "Mac/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "Windows: powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    uv venv "$VENV_DIR"
    echo "Created new virtual environment in ./$VENV_DIR"
else
    echo "Virtual environment already exists in ./$VENV_DIR"
fi

# Activate the virtual environment depending on OS
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    # Windows (Git Bash / MSYS2)
    source "$VENV_DIR/Scripts/activate"
    ACTIVATE_CMD="source $VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    # Mac / Linux
    source "$VENV_DIR/bin/activate"
    ACTIVATE_CMD="source $VENV_DIR/bin/activate"
else
    echo "Error: Could not find activation script."
    exit 1
fi

echo "Installing project via uv (editable mode)..."
# This will install according to what's defined in pyproject.toml
uv pip install -e .

echo "=========================================================="
echo "Installation complete!"
echo "Your virtual environment is ready."
echo "To activate it in your shell, run: $ACTIVATE_CMD"
echo "=========================================================="
