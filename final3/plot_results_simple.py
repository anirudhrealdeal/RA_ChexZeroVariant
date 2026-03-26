#!/usr/bin/env python3
"""
Simple plotting script with better error handling and debugging.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CheXpert pathology labels
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]


def plot_training_loss(metrics_csv, output_path):
    """Plot training and validation loss."""
    print(f"\n1. Reading training metrics from: {metrics_csv}")

    if not os.path.exists(metrics_csv):
        print(f"ERROR: File not found: {metrics_csv}")
        return False

    df = pd.read_csv(metrics_csv)
    print(f"   Found {len(df)} data points")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Step range: {df['step'].min()} to {df['step'].max()}")

    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['train_loss'], 'b-o', label='Training Loss', markersize=3)
    plt.plot(df['step'], df['val_loss'], 'r-s', label='Validation Loss', markersize=3)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Contrastive Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to: {output_path}")
    plt.close()
    return True


def plot_auroc_trend(auroc_csv, output_path):
    """Plot mean AUROC over training steps."""
    print(f"\n2. Reading AUROC results from: {auroc_csv}")

    if not os.path.exists(auroc_csv):
        print(f"ERROR: File not found: {auroc_csv}")
        return False

    df = pd.read_csv(auroc_csv)
    print(f"   Found {len(df)} checkpoints")
    print(f"   Columns: {list(df.columns)}")

    if 'step' not in df.columns or 'mean_auroc' not in df.columns:
        print(f"ERROR: Required columns 'step' or 'mean_auroc' not found!")
        print(f"   Available columns: {list(df.columns)}")
        return False

    df = df.sort_values('step')
    print(f"   Step range: {df['step'].min()} to {df['step'].max()}")
    print(f"   AUROC range: {df['mean_auroc'].min():.4f} to {df['mean_auroc'].max():.4f}")

    # Find best checkpoint
    best_idx = df['mean_auroc'].idxmax()
    best_step = int(df.loc[best_idx, 'step'])
    best_auroc = float(df.loc[best_idx, 'mean_auroc'])
    print(f"   Best: Step {best_step}, AUROC = {best_auroc:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['mean_auroc'], 'o-', color='orange',
             linewidth=2, markersize=6, label='Mean AUROC (14 pathologies)')
    plt.scatter([best_step], [best_auroc], s=300, color='red', marker='*',
               label=f'Best (Step {best_step})', zorder=5, edgecolors='black', linewidths=2)
    plt.axhline(y=best_auroc, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Mean AUROC', fontsize=12)
    plt.title('Validation AUROC Trend Over Training', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.4, 1.0])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to: {output_path}")
    plt.close()
    return True


def plot_individual_aurocs(auroc_csv, output_path):
    """Plot individual pathology AUROCs at best checkpoint."""
    print(f"\n3. Plotting individual pathology AUROCs")

    if not os.path.exists(auroc_csv):
        print(f"ERROR: File not found: {auroc_csv}")
        return False

    df = pd.read_csv(auroc_csv)

    # Find best checkpoint
    best_idx = df['mean_auroc'].idxmax()
    best_row = df.loc[best_idx]
    best_step = int(best_row['step'])

    print(f"   Using checkpoint at step {best_step}")

    # Extract individual AUROCs
    aurocs = []
    labels = []
    for label in CHEXPERT_LABELS:
        col_name = f'{label}_auroc'
        if col_name in best_row and not pd.isna(best_row[col_name]):
            aurocs.append(best_row[col_name])
            labels.append(label)

    print(f"   Found {len(aurocs)} pathologies")

    # Sort by AUROC
    sorted_indices = np.argsort(aurocs)[::-1]
    aurocs = [aurocs[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    mean_auroc = np.mean(aurocs)
    print(f"   Mean AUROC: {mean_auroc:.4f}")

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(labels))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(aurocs)))

    bars = plt.barh(y_pos, aurocs, color=colors, edgecolor='black', linewidth=0.8)

    # Add value labels
    for i, (auroc, bar) in enumerate(zip(aurocs, bars)):
        plt.text(auroc + 0.01, i, f'{auroc:.3f}', va='center', fontsize=10, fontweight='bold')

    # Add mean line
    plt.axvline(x=mean_auroc, color='red', linestyle='--', linewidth=2,
               label=f'Mean AUROC = {mean_auroc:.3f}')

    plt.yticks(y_pos, labels)
    plt.xlabel('AUROC', fontsize=12)
    plt.ylabel('Pathology', fontsize=12)
    plt.title(f'Individual Pathology AUROCs at Best Checkpoint (Step {best_step})',
             fontsize=14, fontweight='bold')
    plt.xlim([0.0, 1.0])
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to: {output_path}")
    plt.close()
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate simple plots')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results/plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.checkpoint_dir, 'training_metrics.csv')
    auroc_csv = 'checkpoint_auroc_results.csv'

    print("="*70)
    print("SIMPLE PLOTTING SCRIPT")
    print("="*70)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Looking for:")
    print(f"  - {metrics_csv}")
    print(f"  - {auroc_csv}")

    success_count = 0

    # Plot 1: Training loss
    if plot_training_loss(metrics_csv, os.path.join(args.output_dir, 'training_loss.png')):
        success_count += 1

    # Plot 2: AUROC trend
    if plot_auroc_trend(auroc_csv, os.path.join(args.output_dir, 'auroc_trend.png')):
        success_count += 1

    # Plot 3: Individual AUROCs
    if plot_individual_aurocs(auroc_csv, os.path.join(args.output_dir, 'individual_aurocs.png')):
        success_count += 1

    print("\n" + "="*70)
    print(f"✓ Generated {success_count}/3 plots successfully")
    print("="*70)


if __name__ == '__main__':
    main()
