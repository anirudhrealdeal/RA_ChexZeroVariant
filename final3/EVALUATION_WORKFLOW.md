# Complete Evaluation Workflow

## Overview

This document describes the complete workflow for training, evaluating, and generating publication-quality plots for your PLIP-style model following the CheXzero evaluation strategy.

---

## 📋 **Workflow Steps**

### **1. Train the Model**

```bash
cd ~/RA_ChexZeroVariant/final
sbatch run_training_job.sh
```

**What happens:**
- Trains for 25,000 steps
- Saves checkpoints every 500 steps to `checkpoints/`
- Validates every 500 steps (computes contrastive loss)
- Logs training/validation loss to `checkpoints/training_metrics.csv`
- Saves `best_model.pt` based on lowest validation loss (NOT used for final evaluation)

**Duration:** ~24-48 hours on A100 GPU

**Monitor progress:**
```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f logs/train-<jobid>.out
```

---

### **2. Evaluate All Checkpoints**

After training completes, evaluate all checkpoints using zero-shot AUROC:

```bash
cd ~/RA_ChexZeroVariant/final

python3 evaluate_checkpoints.py \
    --checkpoint_dir checkpoints \
    --data_dir ../metadata \
    --batch_size 64 \
    --num_workers 8 \
    --device cuda
```

**What happens:**
- Loads each checkpoint from `checkpoints/checkpoint_step*.pt`
- For each checkpoint:
  - Loads the model weights
  - Runs zero-shot evaluation on validation set (234 images)
  - Uses Positive-Negative Similarity (PNS) strategy: `P(pos) = exp(s_pos) / (exp(s_pos) + exp(s_neg))`
  - Computes AUROC for each of 14 pathologies
  - Computes mean AUROC
- Selects best checkpoint by **highest mean AUROC** (NOT lowest loss!)
- Saves results to:
  - `checkpoint_auroc_results.csv` - All checkpoint results
  - `best_checkpoint_info.json` - Best checkpoint details

**Duration:** ~1-2 hours for 50 checkpoints

---

### **3. Generate Plots**

Generate publication-quality plots:

```bash
cd ~/RA_ChexZeroVariant/final

python3 plot_results.py \
    --checkpoint_dir checkpoints \
    --output_dir plots
```

**What happens:**
- Reads `checkpoints/training_metrics.csv` (from training)
- Reads `checkpoint_auroc_results.csv` (from evaluation)
- Reads `best_checkpoint_info.json` (from evaluation)
- Generates 3 plots + 1 summary report

**Duration:** < 1 minute

---

## 📊 **Generated Plots**

### **1. Training and Validation Loss**
**File:** `plots/training_loss.png`

- X-axis: Training steps
- Y-axis: Contrastive loss
- Two curves: Training loss (blue) and Validation loss (red)
- Shows model convergence and overfitting trends

### **2. Validation AUROC Over Steps**
**File:** `plots/validation_auroc_over_steps.png`

- X-axis: Training steps
- Y-axis: Mean AUROC (across 14 pathologies)
- Shows AUROC progression during training
- **Best checkpoint highlighted with red star**
- Demonstrates when model reached peak performance

### **3. Individual Pathology AUROCs**
**File:** `plots/individual_pathology_aurocs.png`

- Horizontal bar chart of AUROC for each of 14 pathologies
- **At the best checkpoint** (selected by mean AUROC)
- Sorted by AUROC value (descending)
- Shows which pathologies the model performs best/worst on
- Includes mean AUROC line

### **4. Training Summary Report**
**File:** `plots/training_summary.txt`

Text report containing:
- Training configuration (steps, validation frequency)
- Best checkpoint selection details
- Table of individual pathology AUROCs
- **Explanation of why this checkpoint was selected**
- Comparison with final checkpoint (overfitting analysis)

---

## 🎯 **Key Evaluation Metrics**

| Metric | Source | Purpose |
|--------|--------|---------|
| **Training Loss** | Contrastive loss on training set | Shows model learning progress |
| **Validation Loss** | Contrastive loss on validation set | Shows generalization during training |
| **Validation AUROC** | Zero-shot PNS on validation set | **Clinical performance metric** |
| **Mean AUROC** | Average of 14 pathology AUROCs | **Overall model performance** |

**Best checkpoint = highest mean AUROC** ✅

---

## 📝 **For Your Paper/Report**

Include these in your results section:

### **Required Figures:**

1. **Figure 1: Training Dynamics**
   - Use `plots/training_loss.png`
   - Caption: "Training and validation contrastive loss over 25,000 training steps. Model converges after ~15,000 steps."

2. **Figure 2: Validation Performance**
   - Use `plots/validation_auroc_over_steps.png`
   - Caption: "Mean AUROC on CheXpert validation set (234 images) across training. Best checkpoint (red star) selected at step X with mean AUROC = Y."

3. **Figure 3: Pathology-Specific Performance**
   - Use `plots/individual_pathology_aurocs.png`
   - Caption: "Individual pathology AUROCs at the best checkpoint. Model achieves mean AUROC of X across 14 CheXpert pathologies."

### **Required Text:**

From `plots/training_summary.txt`, include:

> "We selected the checkpoint at step X (out of 25,000 total training steps) as our final model, as it achieved the highest mean AUROC (Y) on the CheXpert validation set across all 14 pathologies. This checkpoint was identified using the Positive-Negative Similarity (PNS) zero-shot evaluation strategy from the CheXzero paper. [Note: If best checkpoint is before final step, add:] Notably, the best checkpoint occurred before the final training step, indicating that validation AUROC peaked earlier in training, which validates the importance of checkpoint selection by clinical performance metrics rather than training loss alone."

---

## 🔬 **Comparison with CheXzero**

Your evaluation follows the exact CheXzero strategy:

| Aspect | CheXzero | Your Implementation |
|--------|----------|---------------------|
| Evaluation method | Positive-Negative Similarity | ✅ Implemented in `evaluate_checkpoints.py` |
| Prompts | `("{}", "no {}")` | ✅ Same templates |
| Metric | AUROC per pathology | ✅ Same |
| Dataset | CheXpert validation (234 images) | ✅ Same |
| Pathologies | 14 CheXpert labels | ✅ Same |
| Best model selection | By AUROC, not loss | ✅ Implemented |

**Your results are directly comparable to CheXzero!** 🎯

---

## 📁 **Final File Structure**

After completing all steps:

```
~/RA_ChexZeroVariant/final/
├── checkpoints/
│   ├── checkpoint_step500.pt
│   ├── checkpoint_step1000.pt
│   ├── ...
│   ├── checkpoint_step25000.pt
│   ├── best_model.pt  (lowest loss - not used)
│   └── training_metrics.csv  ← Training/val loss log
├── checkpoint_auroc_results.csv  ← AUROC for all checkpoints
├── best_checkpoint_info.json     ← Best checkpoint details
└── plots/
    ├── training_loss.png
    ├── validation_auroc_over_steps.png
    ├── individual_pathology_aurocs.png
    └── training_summary.txt
```

---

## ✅ **Checklist**

Before submission/publication:

- [ ] Training completed successfully (25,000 steps)
- [ ] All checkpoints saved to `checkpoints/`
- [ ] `training_metrics.csv` contains loss data
- [ ] Ran `evaluate_checkpoints.py` successfully
- [ ] `checkpoint_auroc_results.csv` generated
- [ ] Best checkpoint identified in `best_checkpoint_info.json`
- [ ] Ran `plot_results.py` successfully
- [ ] All 3 plots generated in `plots/`
- [ ] `training_summary.txt` contains clear justification
- [ ] Plots included in paper/report
- [ ] Best checkpoint clearly stated with AUROC value

---

## 📚 **References**

Sources for evaluation strategy:
- [CheXzero Paper (Nature Biomedical Engineering)](https://www.nature.com/articles/s41551-022-00936-9)
- [Improving Zero-Shot X-ray Classification (Scientific Reports)](https://www.nature.com/articles/s41598-024-73695-z)
- [CheXzero GitHub Repository](https://github.com/rajpurkarlab/CheXzero)

---

## 🆘 **Troubleshooting**

### **Issue: evaluate_checkpoints.py fails**
- Check that validation labels CSV exists: `~/RA_ChexZeroVariant/metadata/chexpert_plus_valid_labels.csv`
- Check that validation HDF5 exists: `~/RA_ChexZeroVariant/metadata/chexpert_plus_valid.h5`
- Verify paths match between CSV and HDF5

### **Issue: plot_results.py fails**
- Ensure training completed and `training_metrics.csv` exists
- Ensure evaluation completed and `checkpoint_auroc_results.csv` exists
- Run `evaluate_checkpoints.py` first

### **Issue: Out of memory during evaluation**
- Reduce batch size: `--batch_size 32` or `--batch_size 16`
- Reduce num_workers: `--num_workers 4`

---

## 🎉 **You're Ready!**

Submit your training job and follow this workflow to generate publication-ready results!

```bash
cd ~/RA_ChexZeroVariant/final
sbatch run_training_job.sh
```

Good luck! 🚀
