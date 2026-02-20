#!/bin/bash
#SBATCH --job-name=eval_plip
#SBATCH --partition=ai
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64GB
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

# Load modules
module purge
module load python/3.11
module load cuda/12.2

# Navigate to project directory
cd ~/RA_ChexZeroVariant/final

# Prioritize user packages over system packages
export PYTHONPATH=~/.local/lib/python3.11/site-packages:$PYTHONPATH

# Create results directory
mkdir -p results

echo "Starting evaluation at $(date)"
echo "================================================"

# Step 1: Evaluate all checkpoints on validation set (selects best by AUROC)
echo ""
echo "Step 1: Evaluating checkpoints on validation set..."
echo "------------------------------------------------"
python3 evaluate_checkpoints_fixed.py \
    --checkpoint_dir checkpoints \
    --data_dir ../metadata \
    --output_csv results/checkpoint_auroc_results.csv

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 2: Compute bootstrap confidence intervals
echo "Step 2: Computing bootstrap 95% confidence intervals..."
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

# Step 3: Generate plots
echo "Step 3: Generating result plots..."
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
echo "All evaluation steps completed at $(date)"
echo ""
echo "Results saved to:"
echo "  - results/checkpoint_auroc_results.csv (AUROC per checkpoint per pathology)"
echo "  - results/bootstrap_ci.csv (95% confidence intervals)"
echo "  - results/plots/ (visualizations)"
echo ""
echo "To view results:"
echo "  cat results/checkpoint_auroc_results.csv"
echo "  cat results/bootstrap_ci.csv"
echo ""
echo "Best checkpoint will be identified by highest Mean AUROC"
