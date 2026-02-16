# PLIP-Style Training on CheXpert-Plus + ReXGradient

This directory contains the final, clean versions of preprocessing and training scripts for PLIP-style contrastive learning on chest X-ray data.

## Overview

**Model Architecture:**
- **Vision Encoder**: DINOv2 ViT-B/14 (pretrained on natural images)
- **Text Encoder**: CLIP text encoder (from OpenAI CLIP ViT-B/32)
- **Training Objective**: Contrastive learning (CLIP-style symmetric cross-entropy)
- **Embedding Dimension**: 512 (configurable)

**Datasets:**
- **Training**: CheXpert-Plus (223K images) + ReXGradient (239K images) = 462K total
- **Validation**: CheXpert-Plus validation set

## Files

### Preprocessing
- `preprocess.py` - Complete preprocessing pipeline (CheXpert-Plus + ReXGradient → HDF5 + CSV)
  - Handles file extension mismatch (.jpg → .png)
  - Memory-efficient chunked HDF5 merging
  - Creates aligned HDF5/CSV pairs

### Training
- `train_plip.py` - PLIP-style training script
  - DINOv2 vision encoder + CLIP text encoder
  - Contrastive learning with symmetric cross-entropy loss
  - Mixed precision training (AMP)
  - Automatic checkpointing and best model saving

### Job Scripts
- `run_training_job.sh` - SLURM script for GPU training
  - Requests 1 GPU, 8 CPUs, 64GB RAM
  - 48-hour time limit
  - Batch size 128, learning rate 1e-4

## Quick Start

### On CUBIC Cluster

```bash
# 1. Navigate to final directory
cd ~/RA_ChexZeroVariant/final

# 2. Ensure preprocessing is complete
ls -lh ../metadata/combined_train.h5
ls -lh ../metadata/combined_train.csv
ls -lh ../metadata/chexpert_plus_valid.h5
ls -lh ../metadata/chexpert_plus_valid.csv

# 3. Submit training job
sbatch run_training_job.sh

# 4. Monitor training
squeue -u $USER
tail -f logs/train-*.out
```

## Training Configuration

### Default Hyperparameters

```python
--batch_size 128          # Batch size per GPU
--num_epochs 50           # Total training epochs
--lr 1e-4                 # Learning rate (AdamW)
--weight_decay 0.01       # Weight decay
--warmup_epochs 1         # LR warmup epochs
--embed_dim 512           # Embedding dimension
--temperature 0.07        # Initial contrastive temperature
--save_freq 5             # Save checkpoint every N epochs
```

### Custom Training

```bash
# Example: Larger model with more epochs
python3 train_plip.py \
    --data_dir ../metadata \
    --embed_dim 768 \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 5e-5
```

## Model Architecture Details

### Vision Path
```
Input Image (3, 320, 320)
  ↓
DINOv2 ViT-B/14 (frozen or finetuned)
  ↓
Vision Features (768-dim)
  ↓
Projection Head (768 → 512)
  ↓
L2 Normalize
  ↓
Image Embeddings (512-dim)
```

### Text Path
```
Input Text (impression)
  ↓
CLIP Tokenizer (max 77 tokens)
  ↓
CLIP Text Encoder
  ↓
Text Features (512-dim)
  ↓
Projection Head (if embed_dim != 512)
  ↓
L2 Normalize
  ↓
Text Embeddings (512-dim)
```

### Contrastive Loss
```
Similarity Matrix = logit_scale * (image_emb @ text_emb.T)
Loss = 0.5 * (CE_i2t + CE_t2i)

Where:
- CE_i2t = CrossEntropy(logits_per_image, diagonal_labels)
- CE_t2i = CrossEntropy(logits_per_text, diagonal_labels)
- logit_scale is learnable (initialized to 1/0.07)
```

## Output Structure

After training, you'll have:

```
checkpoints/
├── checkpoint_epoch5.pt
├── checkpoint_epoch10.pt
├── checkpoint_epoch15.pt
├── ...
├── checkpoint_epoch50.pt
└── best_model.pt          # Best validation loss model

logs/
├── train-<job_id>.out     # Training progress
└── train-<job_id>.err     # Errors and warnings
```

## Checkpoint Format

Each checkpoint contains:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'train_loss': float,
    'val_loss': float,
    'args': Namespace  # All training arguments
}
```

## Loading Pretrained Model

```python
import torch
from train_plip import PLIPModel

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Initialize model
model = PLIPModel(embed_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Encode image
with torch.no_grad():
    image_embedding = model.encode_image(image_tensor)

# Encode text
with torch.no_grad():
    text_embedding = model.encode_text(text_tokens)

# Compute similarity
similarity = (image_embedding @ text_embedding.T).item()
```

## Performance Tips

1. **Batch Size**: Larger batch size = better contrastive learning
   - Try batch_size=256 or 512 if you have enough GPU memory
   - Reduce if you get OOM errors

2. **Learning Rate**: Scale with batch size
   - Default: 1e-4 for batch_size=128
   - Use 2e-4 for batch_size=256
   - Use 5e-5 for batch_size=64

3. **Mixed Precision**: Already enabled by default
   - Uses `torch.cuda.amp` for faster training
   - Reduces memory usage

4. **Data Loading**: Adjust `num_workers` based on CPU count
   - Default: 8 workers
   - More workers = faster data loading (up to a point)

## Monitoring Training

### Live Monitoring
```bash
# Watch job status
watch -n 2 'squeue -u $USER'

# Follow training output
tail -f logs/train-*.out

# Check for errors
tail -f logs/train-*.err
```

### After Training
```bash
# View final results
tail -n 100 logs/train-*.out

# Check all saved checkpoints
ls -lh checkpoints/

# Find best epoch
grep "Saved best model" logs/train-*.out
```

## Troubleshooting

### OOM (Out of Memory)
- Reduce `--batch_size` (try 64 or 32)
- Reduce `--embed_dim` (try 256)
- Reduce `--num_workers`

### Slow Training
- Increase `--num_workers` (up to # of CPUs)
- Check GPU utilization: `nvidia-smi`
- Ensure data is on fast storage (not network mount)

### Poor Convergence
- Increase `--warmup_epochs` (try 2 or 3)
- Reduce learning rate (try 5e-5 or 3e-5)
- Increase `--batch_size` for more stable gradients

## Next Steps

After training:
1. **Zero-shot evaluation** on downstream tasks
2. **Linear probing** on classification tasks
3. **Fine-tuning** on specific medical imaging tasks
4. **Visualization** of learned embeddings (t-SNE, UMAP)

## Citation

If you use this code, please cite:

- **DINOv2**: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
- **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
- **CheXpert**: Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels"
- **PLIP**: Huang et al., "PLIP: A Vision-Language Foundation Model for Pathology Image Analysis"
