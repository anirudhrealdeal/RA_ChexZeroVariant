#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --partition=all
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=logs/plot-%j.out
#SBATCH --error=logs/plot-%j.err

# Load modules
module purge
module load python/3.11

# Navigate to project directory
cd ~/RA_ChexZeroVariant/final

# Install seaborn if not present
pip install --user -q seaborn

# Prioritize user packages over system packages
export PYTHONPATH=~/.local/lib/python3.11/site-packages:$PYTHONPATH

echo "Starting plotting at $(date)"
echo "================================================"

# Create symlink for file plot script expects in current directory
ln -sf results/checkpoint_auroc_results.csv checkpoint_auroc_results.csv 2>/dev/null || true

# Generate plots from training and evaluation results (using simple version with debug output)
python3 plot_results_simple.py \
    --checkpoint_dir checkpoints \
    --output_dir results/plots

if [ $? -ne 0 ]; then
    echo "ERROR: Plot generation failed!"
    exit 1
fi

echo ""
echo "✓ Plots generated!"
echo ""
echo "================================================"
echo "Plotting completed at $(date)"
echo ""
echo "Plots saved to: results/plots/"
echo ""
echo "Generated plots:"
ls -lh results/plots/
