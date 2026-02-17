#!/bin/bash
#SBATCH --job-name=train_plip
#SBATCH --partition=ai
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=160GB
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

# Load modules
module purge
module load python/3.11
module load cuda/12.2

# Navigate to project directory
cd ~/RA_ChexZeroVariant/final

# Install dependencies with stable precompiled versions (no compilation needed)
pip install --user -q --only-binary :all: numpy h5py torch torchvision pandas scikit-learn matplotlib tqdm timm transformers accelerate ftfy regex Pillow 2>/dev/null || \
pip install --user -q 'numpy<2.0' 'h5py>=3.10,<4.0' 'pandas<3.0' torch torchvision scikit-learn matplotlib tqdm timm transformers accelerate ftfy regex Pillow

# Prioritize user packages over system packages to avoid binary incompatibility
export PYTHONPATH=~/.local/lib/python3.11/site-packages:$PYTHONPATH

# Create logs and checkpoints directories
mkdir -p logs checkpoints

# Run training
echo "Starting training at $(date)"
echo "Using training_strategy.md specifications:"
echo "  - Vision: DINOv3 ViT-B/16 (pretrained)"
echo "  - Text: CLIP ViT-B/32 (pretrained)"
echo "  - Shared embedding: 512-dim (CLIP latent space)"
echo "  - Batch size: 64 (CheXzero paper best model)"
echo "  - Total steps: 25,000 (PLIP strategy)"
echo "  - Validation/Save: Every 500 steps"

python3 train_plip.py \
    --data_dir ../metadata \
    --checkpoint_dir checkpoints \
    --batch_size 64 \
    --max_steps 25000 \
    --lr 1e-4 \
    --momentum 0.9 \
    --optimizer sgd \
    --embed_dim 512 \
    --num_workers 8 \
    --save_steps 500 \
    --val_steps 500

echo "Finished training at $(date)"
