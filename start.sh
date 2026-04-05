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

# Optional KWOK cluster setup
# Set KWOK_ENABLED=true before running to enable Kubernetes simulation
if [ "${KWOK_ENABLED:-false}" = "true" ]; then
    echo ""
    echo "[KWOK] Setting up virtual cluster..."

    KWOK_CLUSTER="${KWOK_CLUSTER_NAME:-kwok}"
    KWOK_KUBECTL="$HOME/.kwok/clusters/${KWOK_CLUSTER}/bin/kubectl"
    KWOK_KUBECONFIG="$HOME/.kwok/clusters/${KWOK_CLUSTER}/kubeconfig.yaml"

    if [ ! -f "$KWOK_KUBECTL" ]; then
        echo "Error: kubectl not found at $KWOK_KUBECTL"
        echo "Make sure you have run: kwokctl create cluster"
        exit 1
    fi

    # Export for the Python adapter to pick up
    export KWOK_KUBECTL="$KWOK_KUBECTL"
    export KWOK_KUBECONFIG="$KWOK_KUBECONFIG"

    # KWOK setup is non-fatal — simulation runs even if KWOK setup fails
    set +e

    # Apply namespace config
    "$KWOK_KUBECTL" --kubeconfig "$KWOK_KUBECONFIG" apply -f kwok/cluster.yaml

    # Create fake nodes (one per region x instance type)
    python -m server.kwok.node_setup

    set -e

    echo "[KWOK] Virtual cluster ready."
    echo ""
fi

echo "Starting Ko2cube environment server..."
# Set PYTHONPATH so bare imports in server/ (rewards, data.scenarios, etc.) resolve correctly
export PYTHONPATH="$(pwd)/server:$(pwd):${PYTHONPATH:-}"
uvicorn server.app:app --host 0.0.0.0 --port 8000

