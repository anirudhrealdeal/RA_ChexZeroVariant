#!/bin/bash
#SBATCH --job-name=train_plip
#SBATCH --partition=ai
#SBATCH --time=72:00:00
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
cd ~/RA_ChexZeroVariant/final3

# Install exact versions from requirements.txt (known to work with precompiled wheels)
pip install --user -q \
    numpy==1.24.3 \
    h5py==3.8.0 \
    pandas==1.5.3 \
    Pillow==9.5.0 \
    scikit-learn==1.2.2 \
    matplotlib==3.7.1 \
    tqdm==4.65.0 \
    ftfy==6.1.1 \
    regex==2023.3.23 \
    torch torchvision \
    transformers accelerate

# Prioritize user packages over system packages
export PYTHONPATH=~/.local/lib/python3.11/site-packages:$PYTHONPATH

# Create logs and checkpoints directories
mkdir -p logs checkpoints

# Run training
echo "Starting training at $(date)"
echo "final3: CheXzero-style CLIP ViT-B/32 fine-tuned on 462K pairs"
echo "  - Vision: CLIP ViT-B/32 (jointly pretrained, no projection needed)"
echo "  - Text:   CLIP ViT-B/32 (jointly pretrained)"
echo "  - Both encoders fine-tuned end-to-end"
echo "  - Dataset: 462K pairs (CheXpert-Plus + ReXGradient) vs CheXzero's ~200K"
echo "  - Batch size: 64 | Steps: 25,000 | Val/Save: every 500"

python3 train_plip.py \
    --data_dir ../metadata \
    --checkpoint_dir checkpoints \
    --batch_size 64 \
    --num_epochs 4 \
    --lr 1e-4 \
    --momentum 0.9 \
    --optimizer sgd \
    --embed_dim 512 \
    --num_workers 8 \
    --save_interval 100 \
    --log_interval 10 \
    --val_interval 500

echo "Finished training at $(date)"
