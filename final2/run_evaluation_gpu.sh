#!/bin/bash
#SBATCH --job-name=eval_gpu
#SBATCH --partition=ai
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64GB
#SBATCH --output=logs/eval-gpu-%j.out
#SBATCH --error=logs/eval-gpu-%j.err

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

# Evaluate all checkpoints on validation set (NEEDS GPU)
echo ""
echo "Evaluating checkpoints on validation set..."
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
echo "Results saved to: results/checkpoint_auroc_results.csv"
echo ""
echo "Next steps (run on CPU):"
echo "  sbatch run_postprocess_cpu.sh"
