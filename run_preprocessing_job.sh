#!/bin/bash
#SBATCH --partition=all
#SBATCH --job-name=preprocess_cxr
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/preprocess-%j.out
#SBATCH --error=logs/preprocess-%j.err

# Load modules
module purge
module load python/3.11
module load cuda/11.8

# Activate virtual environment
source ~/RA_ChexZeroVariant/venv/bin/activate

# Navigate to project directory
cd ~/RA_ChexZeroVariant

# Create logs directory
mkdir -p logs

# Run preprocessing
echo "Starting preprocessing at $(date)"
python3 run_preprocess_combined.py

echo "Finished preprocessing at $(date)"
