#!/bin/bash
# Setup script for Flower FL nodes
#
# This script sets up a Python virtual environment and installs all dependencies.
# Works on Linux (x86), Raspberry Pi (ARM), and macOS (Apple Silicon).
#
# Usage:
#   ./scripts/setup_node.sh
#
# After setup, activate the environment with:
#   source .venv/bin/activate

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "============================================================"
echo "Flower FL Node Setup"
echo "============================================================"

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)

echo "Detected: $OS on $ARCH"

# Check Python version
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.11+ is required but not found."
    echo "Please install Python 3.11 or later."
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"
echo ""

# Create virtual environment
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, delete it first: rm -rf $VENV_DIR"
else
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install project dependencies
echo ""
echo "Installing project dependencies..."
echo "This may take a few minutes, especially on Raspberry Pi..."
echo ""

# Install the project in editable mode
pip install -e .

# Platform-specific notes
echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    if [ "$OS" = "Linux" ]; then
        echo "Raspberry Pi detected!"
        echo ""
        echo "Tips for Raspberry Pi:"
        echo "  - Use smaller batch sizes (e.g., --run-config batch-size=16)"
        echo "  - Training will be slower than on x86/Apple Silicon"
        echo "  - Monitor memory usage with: htop"
        echo ""
    fi
fi

echo "To start as a SuperNode:"
echo "  ./scripts/start_supernode.sh <partition-id> <num-partitions> [superlink-address]"
echo ""
echo "Example (connecting to server at 192.168.1.100):"
echo "  ./scripts/start_supernode.sh 0 2 192.168.1.100:9092"
echo ""
