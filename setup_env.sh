#!/bin/bash
# Setup script for CUBIC cluster
# Usage: bash setup_env.sh

echo "=========================================="
echo "Setting up RA_ChexZeroVariant environment"
echo "=========================================="

# Unload any existing modules
echo "Clearing existing modules..."
module purge

# Load required modules
echo "Loading Python 3.11 and CUDA 11.8..."
module load python/3.11
module load cuda/11.8

# Verify modules loaded
echo "Loaded modules:"
module list

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python3 << 'EOF'
import torch
import h5py
import pandas
import numpy
import cv2

print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"✓ h5py version: {h5py.__version__}")
print(f"✓ pandas version: {pandas.__version__}")
print(f"✓ numpy version: {numpy.__version__}")
print(f"✓ opencv version: {cv2.__version__}")
print("\n✓ All imports successful!")
EOF

# Create logs directory
echo ""
echo "Creating logs directory..."
mkdir -p logs

# Test data access
echo ""
echo "=========================================="
echo "Testing data access..."
echo "=========================================="
if [ -d "/cbica/projects/CXR/data/CheXpert/chexpertplus/" ]; then
    echo "✓ CheXpert-Plus data accessible"
else
    echo "✗ Cannot access CheXpert-Plus data"
fi

if [ -d "/cbica/projects/CXR/data/RexGradient/data/" ]; then
    echo "✓ ReXGradient data accessible"
else
    echo "✗ Cannot access ReXGradient data"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future:"
echo "  module load python/3.11 cuda/11.8"
echo "  source venv/bin/activate"
echo ""
echo "To submit the preprocessing job:"
echo "  sbatch run_preprocessing_job.sh"
echo ""
