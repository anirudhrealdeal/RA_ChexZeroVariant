#!/bin/bash
#SBATCH --job-name=postprocess
#SBATCH --partition=all
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/postprocess-%j.out
#SBATCH --error=logs/postprocess-%j.err

# Load modules
module purge
module load python/3.11

# Navigate to project directory
cd ~/RA_ChexZeroVariant/final

# Prioritize user packages over system packages
export PYTHONPATH=~/.local/lib/python3.11/site-packages:$PYTHONPATH

echo "Starting post-processing at $(date)"
echo "================================================"

# Step 1: Compute bootstrap confidence intervals (CPU only)
echo ""
echo "Computing bootstrap 95% confidence intervals..."
echo "------------------------------------------------"
python3 compute_bootstrap_ci.py \
    --results_path results/checkpoint_auroc_results.csv \
    --output_path results/bootstrap_ci.csv \
    --n_bootstrap 1000

if [ $? -ne 0 ]; then
    echo "ERROR: Bootstrap CI computation failed!"
    exit 1
fi

echo ""
echo "✓ Bootstrap CI complete!"
echo ""

# Step 2: Generate plots (CPU only)
echo "Generating result plots..."
echo "------------------------------------------------"
python3 plot_results.py \
    --results_path results/checkpoint_auroc_results.csv \
    --bootstrap_path results/bootstrap_ci.csv \
    --output_dir results/plots

if [ $? -ne 0 ]; then
    echo "ERROR: Plot generation failed!"
    exit 1
fi

echo ""
echo "✓ Plots generated!"
echo ""
echo "================================================"
echo "All post-processing completed at $(date)"
echo ""
echo "Results:"
echo "  - results/checkpoint_auroc_results.csv"
echo "  - results/bootstrap_ci.csv"
echo "  - results/plots/"
