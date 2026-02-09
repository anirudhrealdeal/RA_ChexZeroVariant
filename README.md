# RA CheXZero Variant

Training CLIP model on CheXpert-Plus + ReXGradient datasets with DINOv2 vision encoder.

## Setup on CUBIC Cluster

### 1. Clone Repository

```bash
cd ~/
git clone <your-repo-url> RA_ChexZeroVariant
cd RA_ChexZeroVariant
```

### 2. Load Modules

```bash
module load python/3.9
module load cuda/11.8
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install h5py pandas numpy tqdm opencv-python Pillow
```

### 5. Verify Setup

```bash
# Test imports
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import h5py, pandas, numpy, cv2; print('✓ All imports successful')"

# Test data access
ls /cbica/projects/CXR/data/CheXpert/chexpertplus/
ls /cbica/projects/CXR/data/RexGradient/data/
```

### 6. Create Logs Directory

```bash
mkdir -p logs
```

## Running Preprocessing

### Option 1: Full Preprocessing (First Time)

```bash
sbatch run_preprocessing_job.sh
```

### Option 2: Resume Preprocessing (After Timeout)

If your job timed out and `metadata/chexpert_plus_train.h5` exists:

```bash
# This skips CheXpert processing (saves ~12 hours)
# and only processes ReXGradient + merge
sbatch run_preprocessing_resume_job.sh
```

### Monitor Job

```bash
# Check job status
squeue -u $USER

# View logs in real-time
tail -f logs/preprocess-*.out
```

### Check Results

```bash
# After completion
ls -lh metadata/

# Verify outputs
python3 << 'EOF'
import h5py
import pandas as pd

with h5py.File('metadata/combined_train.h5', 'r') as f:
    print(f"Training HDF5: {len(f['cxr'])} images")

df = pd.read_csv('metadata/combined_train.csv')
print(f"Training CSV: {len(df)} rows")
EOF
```

## Project Structure

```
RA_ChexZeroVariant/
├── run_preprocess_combined.py    # Dataset preprocessing script
├── run_preprocessing_job.sh      # SLURM job script for preprocessing
├── data_process.py                # Data processing utilities
├── train.py                       # Training utilities
├── model.py                       # CLIP model definition
├── clip.py                        # CLIP implementation
├── simple_tokenizer.py            # Text tokenizer
├── logs/                          # Job logs (created at runtime)
└── metadata/                      # Preprocessed data (created at runtime)
    ├── combined_train.h5
    ├── combined_train.csv
    ├── chexpert_plus_valid.h5
    └── chexpert_plus_valid.csv
```

## Data Paths

- **CheXpert-Plus**: `/cbica/projects/CXR/data/CheXpert/chexpertplus/`
- **ReXGradient**: `/cbica/projects/CXR/data/RexGradient/data/`
- **Output**: `~/RA_ChexZeroVariant/metadata/`

## Notes

- Preprocessing takes ~4-8 hours depending on dataset size
- Requires 64GB RAM and 16 CPU cores
- Output HDF5 files will be several GB in size
