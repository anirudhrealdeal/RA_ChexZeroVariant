#!/bin/bash
#SBATCH --partition=all
#SBATCH --job-name=preprocess_resume
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/preprocess_resume-%j.out
#SBATCH --error=logs/preprocess_resume-%j.err

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

# Run resume preprocessing
echo "Starting RESUME preprocessing at $(date)"
echo "Skipping CheXpert (already done), processing ReXGradient + merge"
python3 run_preprocess_resume.py

echo "Finished resume preprocessing at $(date)"
