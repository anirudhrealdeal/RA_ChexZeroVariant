#!/usr/bin/env python3
"""
Generate publication-quality plots for training results:
1. Training and Validation Loss over Steps
2. Validation AUROC over Steps (for all checkpoints)
3. Individual Pathology AUROCs at Best Checkpoint

Usage:
    python plot_results.py --checkpoint_dir checkpoints --output_dir plots
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.3)
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']

# 14 CheXpert pathology labels
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]


def plot_training_loss(metrics_csv, output_path):
    """
    Plot training and validation loss over steps.

    Args:
        metrics_csv: Path to training_metrics.csv
        output_path: Where to save the plot
    """
    print(f"\nGenerating training loss plot...")

    df = pd.read_csv(metrics_csv)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot train and val loss
    ax.plot(df['step'], df['train_loss'], label='Training Loss',
            linewidth=2, color='#2E86AB', marker='o', markersize=4, markevery=5)
    ax.plot(df['step'], df['val_loss'], label='Validation Loss',
            linewidth=2, color='#A23B72', marker='s', markersize=4, markevery=5)

    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Contrastive Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def plot_auroc_over_steps(auroc_csv, best_step, output_path):
    """
    Plot validation AUROC over steps for all checkpoints.
    Highlight the best checkpoint.

    Args:
        auroc_csv: Path to checkpoint_auroc_results.csv
        best_step: Step number of best checkpoint
        output_path: Where to save the plot
    """
    print(f"\nGenerating AUROC over steps plot...")

    df = pd.read_csv(auroc_csv)
    df = df.sort_values('step')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean AUROC
    ax.plot(df['step'], df['mean_auroc'], label='Mean AUROC (14 pathologies)',
            linewidth=2.5, color='#F18F01', marker='o', markersize=5)

    # Highlight best checkpoint
    best_row = df[df['step'] == best_step].iloc[0]
    ax.scatter([best_step], [best_row['mean_auroc']],
              s=200, color='#C73E1D', marker='*',
              label=f'Best Checkpoint (Step {best_step})',
              zorder=5, edgecolors='black', linewidths=1.5)

    # Add horizontal line at best AUROC
    ax.axhline(y=best_row['mean_auroc'], color='#C73E1D',
              linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean AUROC', fontsize=14, fontweight='bold')
    ax.set_title('Validation AUROC Trend Over Training', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0.5, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def plot_individual_aurocs(auroc_csv, best_step, output_path):
    """
    Plot individual pathology AUROCs at the best checkpoint as a bar chart.

    Args:
        auroc_csv: Path to checkpoint_auroc_results.csv
        best_step: Step number of best checkpoint
        output_path: Where to save the plot
    """
    print(f"\nGenerating individual pathology AUROC plot...")

    df = pd.read_csv(auroc_csv)
    best_row = df[df['step'] == best_step].iloc[0]

    # Extract individual AUROCs
    aurocs = []
    labels = []
    for label in CHEXPERT_LABELS:
        col_name = f'{label}_auroc'
        if col_name in best_row and not pd.isna(best_row[col_name]):
            aurocs.append(best_row[col_name])
            labels.append(label)

    # Sort by AUROC value
    sorted_indices = np.argsort(aurocs)[::-1]  # Descending order
    aurocs = [aurocs[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    # Create color palette (darker for higher AUROCs)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(aurocs)))

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.barh(labels, aurocs, color=colors, edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    for i, (bar, auroc) in enumerate(zip(bars, aurocs)):
        ax.text(auroc + 0.01, i, f'{auroc:.3f}',
               va='center', fontsize=10, fontweight='bold')

    # Add mean AUROC line
    mean_auroc = np.mean(aurocs)
    ax.axvline(x=mean_auroc, color='red', linestyle='--',
              linewidth=2, label=f'Mean AUROC = {mean_auroc:.3f}')

    ax.set_xlabel('AUROC', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pathology', fontsize=14, fontweight='bold')
    ax.set_title(f'Individual Pathology AUROCs at Best Checkpoint (Step {best_step})',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0.5, 1.0])
    ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def generate_summary_report(metrics_csv, auroc_csv, best_step, best_checkpoint, mean_auroc, output_path):
    """
    Generate a text summary report of training results.

    Args:
        metrics_csv: Path to training_metrics.csv
        auroc_csv: Path to checkpoint_auroc_results.csv
        best_step: Step number of best checkpoint
        best_checkpoint: Filename of best checkpoint
        mean_auroc: Mean AUROC of best checkpoint
        output_path: Where to save the report
    """
    print(f"\nGenerating summary report...")

    # Load data
    metrics_df = pd.read_csv(metrics_csv)
    auroc_df = pd.read_csv(auroc_csv)

    best_row = auroc_df[auroc_df['step'] == best_step].iloc[0]

    # Create report
    report = []
    report.append("="*70)
    report.append("TRAINING RESULTS SUMMARY")
    report.append("="*70)
    report.append("")

    report.append("## Training Configuration")
    report.append(f"  Total training steps: {metrics_df['step'].max()}")
    report.append(f"  Validation frequency: {metrics_df['step'].iloc[1] - metrics_df['step'].iloc[0]} steps")
    report.append(f"  Number of checkpoints evaluated: {len(auroc_df)}")
    report.append("")

    report.append("## Best Checkpoint Selection")
    report.append(f"  Selection criterion: Highest Mean AUROC on validation set")
    report.append(f"  Best checkpoint: {best_checkpoint}")
    report.append(f"  Step: {best_step}")
    report.append(f"  Mean AUROC: {mean_auroc:.4f}")
    report.append("")

    # Training loss at best checkpoint
    closest_metric = metrics_df.iloc[(metrics_df['step'] - best_step).abs().argsort()[:1]]
    report.append(f"  Training loss at best step: {closest_metric['train_loss'].values[0]:.4f}")
    report.append(f"  Validation loss at best step: {closest_metric['val_loss'].values[0]:.4f}")
    report.append("")

    report.append("## Individual Pathology AUROCs at Best Checkpoint")
    report.append("")

    # Table of AUROCs
    aurocs_data = []
    for label in CHEXPERT_LABELS:
        col_name = f'{label}_auroc'
        if col_name in best_row and not pd.isna(best_row[col_name]):
            aurocs_data.append((label, best_row[col_name]))

    # Sort by AUROC
    aurocs_data.sort(key=lambda x: x[1], reverse=True)

    report.append(f"  {'Pathology':<35} {'AUROC':>10}")
    report.append(f"  {'-'*35} {'-'*10}")
    for label, auroc in aurocs_data:
        report.append(f"  {label:<35} {auroc:>10.4f}")

    report.append("")
    report.append(f"  {'Mean AUROC':<35} {mean_auroc:>10.4f}")
    report.append("")

    report.append("## Why This Checkpoint Was Selected")
    report.append("")
    report.append(f"  The checkpoint at step {best_step} was selected because it achieved")
    report.append(f"  the highest Mean AUROC ({mean_auroc:.4f}) on the CheXpert validation")
    report.append(f"  set across all 14 pathologies, using the Positive-Negative Similarity")
    report.append(f"  (PNS) zero-shot evaluation strategy from the CheXzero paper.")
    report.append("")

    # Check if best is at end or earlier (overfitting detection)
    final_step = auroc_df['step'].max()
    if best_step < final_step:
        final_auroc = auroc_df[auroc_df['step'] == final_step]['mean_auroc'].values[0]
        report.append(f"  Note: The best checkpoint (step {best_step}, AUROC {mean_auroc:.4f})")
        report.append(f"  outperforms the final checkpoint (step {final_step}, AUROC {final_auroc:.4f}),")
        report.append(f"  indicating that validation AUROC peaked earlier in training.")
        report.append(f"  This validates the importance of checkpoint selection by AUROC")
        report.append(f"  rather than training loss alone.")
    else:
        report.append(f"  The best checkpoint is the final checkpoint, suggesting that")
        report.append(f"  the model continued to improve throughout training.")

    report.append("")
    report.append("="*70)

    # Save report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"  ✓ Saved to: {output_path}")

    # Also print to console
    print("\n" + '\n'.join(report))


def main():
    parser = argparse.ArgumentParser(description='Generate training result plots')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing checkpoints and metrics')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save plots')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # File paths
    metrics_csv = os.path.join(args.checkpoint_dir, 'training_metrics.csv')
    auroc_csv = 'checkpoint_auroc_results.csv'  # In current directory

    # Check files exist
    missing_files = []
    if not os.path.exists(metrics_csv):
        missing_files.append(metrics_csv)
    if not os.path.exists(auroc_csv):
        missing_files.append(auroc_csv)

    if missing_files:
        print("ERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nMake sure you have:")
        print("  1. Completed training (generates training_metrics.csv)")
        print("  2. Run evaluate_checkpoints.py (generates checkpoint_auroc_results.csv)")
        sys.exit(1)

    # Find best checkpoint from AUROC results (highest mean AUROC)
    auroc_df = pd.read_csv(auroc_csv)
    best_idx = auroc_df['mean_auroc'].idxmax()
    best_row = auroc_df.loc[best_idx]
    best_step = int(best_row['step'])
    mean_auroc = float(best_row['mean_auroc'])
    best_checkpoint = f'checkpoint_step{best_step}.pt'

    print("="*70)
    print("GENERATING TRAINING RESULT PLOTS")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Best checkpoint: {best_checkpoint} (Step {best_step})")
    print(f"Best Mean AUROC: {mean_auroc:.4f}")
    print()

    # Generate plots
    plot_training_loss(
        metrics_csv,
        os.path.join(args.output_dir, 'training_loss.png')
    )

    plot_auroc_over_steps(
        auroc_csv,
        best_step,
        os.path.join(args.output_dir, 'validation_auroc_over_steps.png')
    )

    plot_individual_aurocs(
        auroc_csv,
        best_step,
        os.path.join(args.output_dir, 'individual_pathology_aurocs.png')
    )

    generate_summary_report(
        metrics_csv,
        auroc_csv,
        best_step,
        best_checkpoint,
        mean_auroc,
        os.path.join(args.output_dir, 'training_summary.txt')
    )

    print("\n" + "="*70)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*70)
    print(f"\nPlots saved to: {args.output_dir}/")
    print("  - training_loss.png")
    print("  - validation_auroc_over_steps.png")
    print("  - individual_pathology_aurocs.png")
    print("  - training_summary.txt")
    print()


if __name__ == '__main__':
    main()
