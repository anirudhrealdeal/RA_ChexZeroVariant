# Pre-Training Checklist

## ✅ Prerequisites

### 1. Environment Setup
- [ ] Python 3.11 loaded: `module load python/3.11`
- [ ] CUDA 11.8 loaded: `module load cuda/11.8`
- [ ] Virtual environment activated: `source ~/RA_ChexZeroVariant/venv/bin/activate`
- [ ] PyTorch installed with CUDA support
- [ ] All dependencies installed from requirements.txt

**Verify:**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Preprocessing Complete
- [ ] Combined training HDF5: `../metadata/combined_train.h5` exists
- [ ] Combined training CSV: `../metadata/combined_train.csv` exists
- [ ] Validation HDF5: `../metadata/chexpert_plus_valid.h5` exists
- [ ] Validation CSV: `../metadata/chexpert_plus_valid.csv` exists

**Verify:**
```bash
cd ~/RA_ChexZeroVariant/final
ls -lh ../metadata/combined_train.h5
ls -lh ../metadata/combined_train.csv
ls -lh ../metadata/chexpert_plus_valid.h5
ls -lh ../metadata/chexpert_plus_valid.csv
```

**Expected sizes:**
- combined_train.h5: ~180GB (223K + 239K images)
- combined_train.csv: ~20MB (462K rows)
- chexpert_plus_valid.h5: ~2GB (234 images)
- chexpert_plus_valid.csv: ~10KB (234 rows)

### 3. Data Alignment Verified
- [ ] HDF5 and CSV row counts match

**Verify:**
```bash
python3 << 'EOF'
import h5py
import pandas as pd

# Check training data
with h5py.File('../metadata/combined_train.h5', 'r') as f:
    h5_train = len(f['cxr'])
df_train = pd.read_csv('../metadata/combined_train.csv')
print(f"Training - HDF5: {h5_train}, CSV: {len(df_train)}, Match: {h5_train == len(df_train)}")

# Check validation data
with h5py.File('../metadata/chexpert_plus_valid.h5', 'r') as f:
    h5_val = len(f['cxr'])
df_val = pd.read_csv('../metadata/chexpert_plus_valid.csv')
print(f"Validation - HDF5: {h5_val}, CSV: {len(df_val)}, Match: {h5_val == len(df_val)}")
EOF
```

### 4. Required Files Present
- [ ] `train_plip.py` in final directory
- [ ] `run_training_job.sh` in final directory
- [ ] `simple_tokenizer.py` in parent directory (required import)
- [ ] `clip.py` in parent directory (required import)
- [ ] `bpe_simple_vocab_16e6.txt.gz` in parent directory (tokenizer vocab)

**Verify:**
```bash
cd ~/RA_ChexZeroVariant/final
ls -la train_plip.py run_training_job.sh
ls -la ../simple_tokenizer.py ../clip.py ../bpe_simple_vocab_16e6.txt.gz
```

### 5. Directory Structure
```
~/RA_ChexZeroVariant/
├── final/
│   ├── train_plip.py           ← Training script
│   ├── run_training_job.sh     ← SLURM job script
│   ├── README.md
│   ├── PLIP_STRATEGY.md
│   └── CHECKLIST.md
├── metadata/
│   ├── combined_train.h5       ← Training images
│   ├── combined_train.csv      ← Training metadata
│   ├── chexpert_plus_valid.h5  ← Validation images
│   └── chexpert_plus_valid.csv ← Validation metadata
├── clip.py                     ← CLIP implementation
├── simple_tokenizer.py         ← Text tokenizer
├── bpe_simple_vocab_16e6.txt.gz ← Tokenizer vocab
└── venv/                       ← Virtual environment
```

## 🚀 Ready to Train

### Option 1: Submit SLURM Job (Recommended)

```bash
cd ~/RA_ChexZeroVariant/final
sbatch run_training_job.sh
```

Monitor:
```bash
# Check job status
squeue -u $USER

# Watch output
tail -f logs/train-*.out

# Check for errors
tail -f logs/train-*.err
```

### Option 2: Interactive Testing (Debug)

```bash
# Request interactive GPU session
srun --partition=gpu --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=2:00:00 --pty bash

# Activate environment
module load python/3.11 cuda/11.8
source ~/RA_ChexZeroVariant/venv/bin/activate

# Run small test
cd ~/RA_ChexZeroVariant/final
python3 train_plip.py \
    --batch_size 32 \
    --num_epochs 2 \
    --save_freq 1
```

## 🔍 Common Issues

### Issue: ModuleNotFoundError: No module named 'clip'

**Solution:**
```bash
# Ensure you're in the right directory
cd ~/RA_ChexZeroVariant/final

# clip.py should be in parent directory
ls -la ../clip.py
```

### Issue: FileNotFoundError: bpe_simple_vocab_16e6.txt.gz

**Solution:**
```bash
# Download CLIP vocab file
cd ~/RA_ChexZeroVariant
wget https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz
```

### Issue: CUDA out of memory

**Solutions:**
1. Reduce batch size in `run_training_job.sh`:
   ```bash
   --batch_size 64  # or 32
   ```

2. Reduce number of workers:
   ```bash
   --num_workers 4  # instead of 8
   ```

3. Reduce embedding dimension:
   ```bash
   --embed_dim 256  # instead of 512
   ```

### Issue: Training very slow

**Check:**
1. Are you using GPU? `nvidia-smi` should show Python process
2. Are workers loading data? Increase `--num_workers` if CPU usage is low
3. Is HDF5 on fast storage? Network mounts can be slow

**Solution:**
```bash
# Copy data to local scratch if available
cp ../metadata/*.h5 /scratch/$USER/
python3 train_plip.py --data_dir /scratch/$USER/
```

### Issue: Job killed (OOM)

If job is killed due to memory (not GPU memory, but system RAM):

**Solution:** Request more memory in SLURM script:
```bash
#SBATCH --mem=128G  # instead of 64G
```

## 📊 Expected Timeline

**With 1 GPU (V100/A100):**
- Epoch time: ~30-45 minutes (batch_size=128)
- 50 epochs: ~25-40 hours total
- First checkpoint saved at epoch 5 (~2-3 hours)

**Checkpoints saved every 5 epochs:**
- Epoch 5: ~2.5 hours
- Epoch 10: ~5 hours
- Epoch 15: ~7.5 hours
- ...
- Epoch 50: ~40 hours

## ✨ Post-Training

After training completes:

1. **Check final results:**
   ```bash
   tail -n 100 logs/train-*.out
   ```

2. **Find best model:**
   ```bash
   ls -lh checkpoints/best_model.pt
   ```

3. **Verify checkpoints:**
   ```bash
   ls -lh checkpoints/
   # Should see: checkpoint_epoch5.pt, epoch10.pt, ..., epoch50.pt, best_model.pt
   ```

4. **Copy to safe location:**
   ```bash
   cp -r checkpoints ~/results/plip_training_$(date +%Y%m%d)
   ```

5. **Test loading:**
   ```python
   import torch
   checkpoint = torch.load('checkpoints/best_model.pt')
   print(f"Best epoch: {checkpoint['epoch']}")
   print(f"Best val loss: {checkpoint['val_loss']:.4f}")
   ```

## 🎯 Success Criteria

Training is successful if:
- ✅ All 50 epochs complete without errors
- ✅ Validation loss decreases and stabilizes
- ✅ Best model checkpoint exists
- ✅ Final validation loss < 2.0 (ideally < 1.5)
- ✅ Training loss and validation loss don't diverge significantly

Good luck! 🚀
