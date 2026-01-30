#!/bin/bash

# install.sh
# Sets up a conda environment 'pointnet-seg' with all required libraries for the resegmentation project.
# Based on the 'supercluster' environment used during development.

set -e

ENV_NAME="pointnet-seg"
PYTHON_VERSION="3.10"

echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Initialize conda for the current shell session
# This allows 'conda activate' to work within the script
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

echo "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

echo "Installing PyTorch and CUDA toolkit (defaulting to CUDA 11.8 for compatibility)..."

conda install -y torch torchvision torchaudio -c pytorch -c nvidia

echo "Installing PyTorch Geometric (PyG)..."
conda install -y pyg -c pyg

echo "Installing additional libraries via pip..."
# laspy[lazrs] provides LAZ support. 
# plyfile is required for PLY I/O.
# optuna is used for HPO.
pip install \
    tqdm \
    numpy \
    scipy \
    scikit-learn \
    laspy \
    laspy[lazrs] \
    plyfile \
    optuna

echo "--------------------------------------------------"
echo "Setup Complete!"
echo "To activate the environment, use:"
echo "conda activate $ENV_NAME"
echo "--------------------------------------------------"
