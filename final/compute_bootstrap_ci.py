#!/usr/bin/env python3
"""
Compute bootstrap confidence intervals for the best checkpoint.

This follows the CheXzero methodology:
- 1,000 bootstrap iterations
- 95% confidence intervals (2.5th and 97.5th percentiles)
- Reported as: Mean AUROC ± CI

Usage:
    python compute_bootstrap_ci.py --checkpoint <path> --n_bootstrap 1000
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from tqdm import tqdm
import h5py
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_plip import PLIPModel
from simple_tokenizer import SimpleTokenizer
from evaluate_checkpoints_fixed import (
    CXREvalDataset,
    precompute_text_embeddings,
    compute_probs_from_precomputed,
    CHEXPERT_LABELS
)


def compute_aurocs_for_sample(y_pred_sample, y_true_sample):
    """
    Compute AUROC for each pathology for a bootstrap sample.

    Args:
        y_pred_sample: (N, 14) predictions
        y_true_sample: (N, 14) ground truth

    Returns:
        aucs: List of 14 AUROCs (or NaN if can't compute)
    """
    aucs = []

    for i in range(len(CHEXPERT_LABELS)):
        n_pos = y_true_sample[:, i].sum()
        n_neg = len(y_true_sample) - n_pos

        if n_pos == 0 or n_neg == 0:
            aucs.append(np.nan)
        else:
            try:
                auc = roc_auc_score(y_true_sample[:, i], y_pred_sample[:, i])
                aucs.append(auc)
            except:
                aucs.append(np.nan)

    return aucs


def bootstrap_confidence_intervals(y_pred, y_true, n_bootstrap=1000):
    """
    Compute bootstrap confidence intervals for AUROCs.

    Following CheXzero methodology:
    - Sample with replacement n_bootstrap times
    - Compute AUROC for each sample
    - Take 2.5th and 97.5th percentiles for 95% CI

    Args:
        y_pred: (N, 14) predictions on full validation set
        y_true: (N, 14) ground truth labels
        n_bootstrap: Number of bootstrap iterations (default 1000)

    Returns:
        results_df: DataFrame with mean, lower CI, upper CI for each pathology
    """
    print(f"\nRunning {n_bootstrap} bootstrap iterations...")
    print("This may take a few minutes...\n")

    n_samples = len(y_pred)
    indices = np.arange(n_samples)

    # Store AUROCs for each bootstrap sample
    bootstrap_aucs = []

    np.random.seed(42)  # For reproducibility

    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Sample with replacement
        sample_indices = resample(indices, replace=True, n_samples=n_samples, random_state=i)

        y_pred_sample = y_pred[sample_indices]
        y_true_sample = y_true[sample_indices]

        # Compute AUROCs for this sample
        aucs = compute_aurocs_for_sample(y_pred_sample, y_true_sample)
        bootstrap_aucs.append(aucs)

    # Convert to numpy array: (n_bootstrap, 14)
    bootstrap_aucs = np.array(bootstrap_aucs)

    # Compute statistics for each pathology
    results = []

    for i, label in enumerate(CHEXPERT_LABELS):
        aucs_for_label = bootstrap_aucs[:, i]

        # Remove NaN values
        aucs_for_label = aucs_for_label[~np.isnan(aucs_for_label)]

        if len(aucs_for_label) == 0:
            results.append({
                'pathology': label,
                'mean_auroc': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'std': np.nan
            })
        else:
            mean_auc = np.mean(aucs_for_label)
            ci_lower = np.percentile(aucs_for_label, 2.5)
            ci_upper = np.percentile(aucs_for_label, 97.5)
            std = np.std(aucs_for_label)

            results.append({
                'pathology': label,
                'mean_auroc': mean_auc,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': std
            })

    results_df = pd.DataFrame(results)

    # Compute mean across pathologies (excluding NaN)
    valid_aucs = results_df[~results_df['mean_auroc'].isna()]['mean_auroc'].values
    mean_auroc = np.mean(valid_aucs) if len(valid_aucs) > 0 else np.nan

    # Bootstrap for mean AUROC
    bootstrap_means = []
    for i in range(n_bootstrap):
        aucs = bootstrap_aucs[i, :]
        valid = aucs[~np.isnan(aucs)]
        if len(valid) > 0:
            bootstrap_means.append(np.mean(valid))

    mean_ci_lower = np.percentile(bootstrap_means, 2.5)
    mean_ci_upper = np.percentile(bootstrap_means, 97.5)
    mean_std = np.std(bootstrap_means)

    # Add overall mean row
    overall_row = pd.DataFrame([{
        'pathology': 'Mean (14 pathologies)',
        'mean_auroc': mean_auroc,
        'ci_lower': mean_ci_lower,
        'ci_upper': mean_ci_upper,
        'std': mean_std
    }])

    results_df = pd.concat([results_df, overall_row], ignore_index=True)

    return results_df


def run_inference_and_bootstrap(checkpoint_path, val_loader, tokenizer, device, n_bootstrap=1000):
    """
    Load checkpoint, run inference, and compute bootstrap CIs.

    Args:
        checkpoint_path: Path to checkpoint
        val_loader: Validation data loader
        tokenizer: Text tokenizer
        device: torch device
        n_bootstrap: Number of bootstrap iterations

    Returns:
        results_df: DataFrame with bootstrap CIs
    """
    print("="*70)
    print("BOOTSTRAP CONFIDENCE INTERVAL COMPUTATION")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    model = PLIPModel(embed_dim=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Pre-compute text embeddings
    pos_embeds, neg_embeds = precompute_text_embeddings(model, tokenizer, device)
    logit_scale = model.logit_scale

    # Run inference on validation set
    print("\nRunning inference on validation set...")
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            img_embeds = model.encode_image(images)
            probs = compute_probs_from_precomputed(img_embeds, pos_embeds, neg_embeds, logit_scale)
            y_pred_list.append(probs.cpu().numpy())
            y_true_list.append(labels.numpy())

    y_pred = np.concatenate(y_pred_list, axis=0)  # (N, 14)
    y_true = np.concatenate(y_true_list, axis=0)  # (N, 14)

    print(f"✓ Collected predictions for {len(y_pred)} samples")

    # Compute bootstrap confidence intervals
    results_df = bootstrap_confidence_intervals(y_pred, y_true, n_bootstrap=n_bootstrap)

    return results_df


def main():
    parser = argparse.ArgumentParser(description='Compute bootstrap CIs for best checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (if None, uses best from best_checkpoint_info.json)')
    parser.add_argument('--data_dir', type=str, default='../metadata',
                        help='Directory containing validation data')
    parser.add_argument('--val_h5', type=str, default=None,
                        help='Validation HDF5 file')
    parser.add_argument('--val_labels_csv', type=str, default=None,
                        help='Validation labels CSV')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap iterations')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_csv', type=str, default='bootstrap_confidence_intervals.csv',
                        help='Output CSV file')

    args = parser.parse_args()

    # Get checkpoint path
    if args.checkpoint is None:
        # Load best checkpoint from evaluation
        if not os.path.exists('best_checkpoint_info.json'):
            print("ERROR: best_checkpoint_info.json not found!")
            print("Run evaluate_checkpoints_fixed.py first to find the best checkpoint.")
            sys.exit(1)

        with open('best_checkpoint_info.json', 'r') as f:
            best_info = json.load(f)

        args.checkpoint = best_info['best_checkpoint']
        print(f"Using best checkpoint from evaluation: {args.checkpoint}")
        print(f"Mean AUROC (from evaluation): {best_info['mean_auroc']:.4f}\n")

    # Set up paths
    if args.val_h5 is None:
        args.val_h5 = os.path.join(args.data_dir, 'chexpert_plus_valid.h5')
    if args.val_labels_csv is None:
        args.val_labels_csv = os.path.join(args.data_dir, 'chexpert_plus_valid_labels.csv')

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    # Create validation dataset
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    transform = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        Normalize(mean=clip_mean, std=clip_std)
    ])

    print("Loading validation dataset...")
    val_dataset = CXREvalDataset(args.val_h5, args.val_labels_csv, transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Run inference and bootstrap
    results_df = run_inference_and_bootstrap(
        args.checkpoint, val_loader, tokenizer, device, args.n_bootstrap
    )

    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Results saved to: {args.output_csv}")

    # Print results
    print("\n" + "="*70)
    print("BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    print("="*70)
    print()
    print(f"{'Pathology':<35} {'Mean AUROC':>12} {'95% CI':>20} {'Std':>8}")
    print("-"*80)

    for _, row in results_df.iterrows():
        pathology = row['pathology']
        mean_auc = row['mean_auroc']
        ci_lower = row['ci_lower']
        ci_upper = row['ci_upper']
        std = row['std']

        if not np.isnan(mean_auc):
            ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
            print(f"{pathology:<35} {mean_auc:>12.4f} {ci_str:>20} {std:>8.4f}")

    print("="*70)
    print("\nFor your paper, report as:")
    mean_row = results_df[results_df['pathology'] == 'Mean (14 pathologies)'].iloc[0]
    mean_auc = mean_row['mean_auroc']
    ci_lower = mean_row['ci_lower']
    ci_upper = mean_row['ci_upper']
    margin = (ci_upper - ci_lower) / 2
    print(f"  Mean AUROC: {mean_auc:.3f} ± {margin:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print("="*70)


if __name__ == '__main__':
    main()
