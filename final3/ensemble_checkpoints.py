#!/usr/bin/env python3
"""
Ensemble predictions from top-K checkpoints for improved AUROC.

Following CheXzero's ensemble.py strategy:
- Take top 3-5 checkpoints by AUROC
- Average their probability predictions
- Usually gives +0.01 to +0.02 AUROC boost

Usage:
    python ensemble_checkpoints.py --top_k 3 --auroc_csv checkpoint_auroc_results.csv
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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


def get_predictions_for_checkpoint(checkpoint_path, val_loader, tokenizer, device):
    """
    Get probability predictions for a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        val_loader: Validation data loader
        tokenizer: Text tokenizer
        device: torch device

    Returns:
        y_pred: (N, 14) probability predictions
        y_true: (N, 14) ground truth labels
    """
    print(f"\nLoading checkpoint: {os.path.basename(checkpoint_path)}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    model = PLIPModel(embed_dim=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Pre-compute text embeddings
    pos_embeds, neg_embeds = precompute_text_embeddings(model, tokenizer, device)
    logit_scale = model.logit_scale

    # Run inference
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Inference", leave=False):
            images = images.to(device)
            img_embeds = model.encode_image(images)
            probs = compute_probs_from_precomputed(img_embeds, pos_embeds, neg_embeds, logit_scale)
            y_pred_list.append(probs.cpu().numpy())
            y_true_list.append(labels.numpy())

    y_pred = np.concatenate(y_pred_list, axis=0)
    y_true = np.concatenate(y_true_list, axis=0)

    print(f"  ✓ Got predictions for {len(y_pred)} samples")

    return y_pred, y_true


def ensemble_predictions(predictions_list):
    """
    Average predictions from multiple checkpoints.

    Args:
        predictions_list: List of (N, 14) numpy arrays

    Returns:
        averaged_preds: (N, 14) averaged predictions
    """
    # Stack and average
    stacked = np.stack(predictions_list, axis=0)  # (K, N, 14)
    averaged = np.mean(stacked, axis=0)  # (N, 14)

    return averaged


def compute_aurocs(y_pred, y_true):
    """
    Compute AUROC for each pathology.

    Args:
        y_pred: (N, 14) predictions
        y_true: (N, 14) ground truth

    Returns:
        aucs: List of AUROCs
        mean_auc: Mean AUROC
    """
    aucs = []
    valid_aucs = []

    for i, label in enumerate(CHEXPERT_LABELS):
        n_pos = y_true[:, i].sum()
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            aucs.append(np.nan)
        else:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
                valid_aucs.append(auc)
            except:
                aucs.append(np.nan)

    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

    return aucs, mean_auc


def main():
    parser = argparse.ArgumentParser(description='Ensemble top-K checkpoints')
    parser.add_argument('--auroc_csv', type=str, default='checkpoint_auroc_results.csv',
                        help='CSV with checkpoint AUROCs')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of top checkpoints to ensemble (3-5 recommended)')
    parser.add_argument('--data_dir', type=str, default='../metadata',
                        help='Directory containing validation data')
    parser.add_argument('--val_h5', type=str, default=None,
                        help='Validation HDF5 file')
    parser.add_argument('--val_labels_csv', type=str, default=None,
                        help='Validation labels CSV')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_json', type=str, default='ensemble_results.json',
                        help='Output JSON file')

    args = parser.parse_args()

    print("="*70)
    print("CHECKPOINT ENSEMBLE")
    print("="*70)
    print(f"Top-K checkpoints: {args.top_k}")
    print()

    # Load AUROC results
    if not os.path.exists(args.auroc_csv):
        print(f"ERROR: {args.auroc_csv} not found!")
        print("Run evaluate_checkpoints_fixed.py first.")
        sys.exit(1)

    auroc_df = pd.read_csv(args.auroc_csv)
    auroc_df = auroc_df.sort_values('mean_auroc', ascending=False)

    # Select top-K checkpoints
    top_k_df = auroc_df.head(args.top_k)

    print(f"Top {args.top_k} checkpoints by AUROC:")
    for _, row in top_k_df.iterrows():
        print(f"  Step {int(row['step']):5d}: AUROC = {row['mean_auroc']:.4f}")
    print()

    checkpoint_paths = top_k_df['checkpoint'].tolist()

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

    print(f"Validation set: {len(val_dataset)} samples\n")

    # Get predictions from each checkpoint
    print("="*70)
    print("COLLECTING PREDICTIONS FROM TOP-K CHECKPOINTS")
    print("="*70)

    predictions_list = []
    y_true = None

    for ckpt_path in checkpoint_paths:
        y_pred, y_true_temp = get_predictions_for_checkpoint(
            ckpt_path, val_loader, tokenizer, device
        )
        predictions_list.append(y_pred)

        if y_true is None:
            y_true = y_true_temp

    # Ensemble predictions (average)
    print("\n" + "="*70)
    print("ENSEMBLING PREDICTIONS")
    print("="*70)
    print(f"Averaging predictions from {len(predictions_list)} checkpoints...")

    y_pred_ensemble = ensemble_predictions(predictions_list)
    print("✓ Ensemble complete")

    # Compute AUROCs for ensemble
    print("\nComputing AUROCs for ensemble...")
    aucs_ensemble, mean_auc_ensemble = compute_aurocs(y_pred_ensemble, y_true)

    # Get individual checkpoint AUROCs for comparison
    individual_aucs = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        aucs_single, mean_auc_single = compute_aurocs(predictions_list[i], y_true)
        individual_aucs.append(mean_auc_single)

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print()

    print("Individual checkpoint AUROCs:")
    for i, (ckpt_path, mean_auc) in enumerate(zip(checkpoint_paths, individual_aucs)):
        step = int(top_k_df.iloc[i]['step'])
        print(f"  Checkpoint {i+1} (step {step:5d}): {mean_auc:.4f}")

    print()
    print(f"Ensemble AUROC: {mean_auc_ensemble:.4f}")
    print()

    # Compute improvement
    best_individual = max(individual_aucs)
    improvement = mean_auc_ensemble - best_individual

    print(f"Best individual checkpoint: {best_individual:.4f}")
    print(f"Improvement from ensemble:  {improvement:+.4f} ({improvement*100:+.2f}%)")
    print()

    # Individual pathology AUROCs
    print("Individual pathology AUROCs (ensemble):")
    for label, auc in zip(CHEXPERT_LABELS, aucs_ensemble):
        if not np.isnan(auc):
            print(f"  {label:30s}: {auc:.4f}")

    print("="*70)

    # Save results
    results = {
        'ensemble_config': {
            'top_k': args.top_k,
            'checkpoints': checkpoint_paths,
            'checkpoint_steps': [int(row['step']) for _, row in top_k_df.iterrows()]
        },
        'individual_aurocs': individual_aucs,
        'ensemble_auroc': mean_auc_ensemble,
        'best_individual_auroc': best_individual,
        'improvement': improvement,
        'pathology_aurocs': {
            label: auc for label, auc in zip(CHEXPERT_LABELS, aucs_ensemble) if not np.isnan(auc)
        }
    }

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {args.output_json}")

    # Recommendation
    print("\n" + "="*70)
    if improvement > 0.005:  # 0.5% improvement
        print("✓ RECOMMENDATION: Use ensemble (significant improvement)")
        print(f"  Report ensemble AUROC: {mean_auc_ensemble:.3f}")
    else:
        print("→ RECOMMENDATION: Use single best checkpoint")
        print(f"  Ensemble improvement ({improvement:.4f}) is minimal")
        print(f"  Report single best AUROC: {best_individual:.3f}")
    print("="*70)


if __name__ == '__main__':
    main()
