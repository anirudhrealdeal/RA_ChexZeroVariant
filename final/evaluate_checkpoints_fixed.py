#!/usr/bin/env python3
"""
Evaluate all training checkpoints on validation set using CheXzero zero-shot PNS strategy.
Select the best checkpoint based on Mean AUROC (not validation loss).

This implements the COMPLETE CheXzero evaluation strategy:
1. Ensemble of Prompts (multiple positive/negative templates, averaged)
2. Positive-Negative Similarity (PNS) for each pathology
3. Pre-computed text embeddings (efficiency)
4. Learnable temperature (logit_scale)
5. U-Positive label strategy (uncertain = positive)
6. AUROC computation for 14 CheXpert labels

Usage:
    python evaluate_checkpoints_fixed.py --checkpoint_dir checkpoints --data_dir ../metadata
"""

import os
import sys
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import h5py
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model and tokenizer from train_plip.py
from train_plip import PLIPModel
from simple_tokenizer import SimpleTokenizer

# 14 CheXpert pathology labels (must match order in validation labels CSV)
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# Ensemble of templates (following CheXzero strategy)
# These are averaged to create robust text representations
POS_TEMPLATES = [
    "{}",
    "findings suggestive of {}",
    "findings consistent with {}",
    "this is consistent with {}",
]

NEG_TEMPLATES = [
    "no {}",
    "no evidence of {}",
    "no sign of {}",
    "absence of {}",
]


class CXREvalDataset(torch.utils.data.Dataset):
    """
    Dataset for zero-shot evaluation with ground truth labels.
    Loads images from HDF5 and binary labels from CSV.
    """
    def __init__(self, h5_path, labels_csv_path, transform):
        self.h5_path = h5_path
        self._h5_file = None
        self.transform = transform

        # Load ground truth labels
        self.labels_df = pd.read_csv(labels_csv_path)
        print(f"Loaded {len(self.labels_df)} samples with ground truth labels")

        # Verify columns
        missing_labels = [l for l in CHEXPERT_LABELS if l not in self.labels_df.columns]
        if missing_labels:
            raise ValueError(f"Missing label columns: {missing_labels}")

        # Apply U-Positive mapping: uncertain (-1) → positive (1)
        # This is the CheXpert competition standard for evaluation
        print("Applying U-Positive mapping (uncertain = positive)...")
        for label in CHEXPERT_LABELS:
            n_uncertain = (self.labels_df[label] == -1).sum()
            if n_uncertain > 0:
                print(f"  {label}: {n_uncertain} uncertain labels → positive")

        self.labels_df[CHEXPERT_LABELS] = self.labels_df[CHEXPERT_LABELS].replace(-1, 1)
        print("✓ U-Positive mapping applied")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Lazy open HDF5 file
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        # Load image
        image = self._h5_file['cxr'][idx]
        image = torch.from_numpy(image).float() / 255.0

        # Convert grayscale to RGB
        if image.dim() == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)
        elif image.dim() == 3:
            image = image.permute(2, 0, 1)

        # Apply transforms (resize + normalize)
        image = self.transform(image)

        # Load 14 binary labels
        labels = self.labels_df.iloc[idx][CHEXPERT_LABELS].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels


def precompute_text_embeddings(model, tokenizer, device):
    """
    Pre-compute text embeddings for all pathologies using ensemble of prompts.
    This is done ONCE before the evaluation loop (efficiency).

    Following CheXzero's zeroshot_classifier function:
    1. For each pathology, encode all positive templates
    2. Average the embeddings to get consensus positive representation
    3. Repeat for negative templates
    4. Return pre-computed embeddings for efficient inference

    Args:
        model: PLIPModel with text encoder
        tokenizer: SimpleTokenizer
        device: torch device

    Returns:
        pos_embeds: (14, 512) tensor of positive embeddings
        neg_embeds: (14, 512) tensor of negative embeddings
    """
    print("\n" + "="*70)
    print("PRE-COMPUTING TEXT EMBEDDINGS (CheXzero Ensemble Strategy)")
    print("="*70)
    print(f"  Using {len(POS_TEMPLATES)} positive templates")
    print(f"  Using {len(NEG_TEMPLATES)} negative templates")
    print()

    pos_embeds_list = []
    neg_embeds_list = []

    with torch.no_grad():
        for label in tqdm(CHEXPERT_LABELS, desc="Encoding text"):
            # === POSITIVE TEMPLATES ===
            # Format all positive templates with the pathology name
            pos_texts = [template.format(label) for template in POS_TEMPLATES]

            # Tokenize all positive texts
            pos_tokens_list = []
            for text in pos_texts:
                tokens = tokenizer.encode(text)
                tokens = tokens[:77] + [0] * (77 - len(tokens))  # Pad to 77
                pos_tokens_list.append(tokens)

            pos_tokens = torch.tensor(pos_tokens_list, dtype=torch.long).to(device)  # (N_pos, 77)

            # Encode all positive templates
            pos_embeddings = model.encode_text(pos_tokens)  # (N_pos, 512)

            # Average to get consensus positive representation (KEY STEP!)
            pos_embed_avg = pos_embeddings.mean(dim=0)  # (512,)
            pos_embed_avg = F.normalize(pos_embed_avg, dim=0)  # Renormalize after averaging
            pos_embeds_list.append(pos_embed_avg)

            # === NEGATIVE TEMPLATES ===
            # Same process for negative templates
            neg_texts = [template.format(label) for template in NEG_TEMPLATES]

            neg_tokens_list = []
            for text in neg_texts:
                tokens = tokenizer.encode(text)
                tokens = tokens[:77] + [0] * (77 - len(tokens))
                neg_tokens_list.append(tokens)

            neg_tokens = torch.tensor(neg_tokens_list, dtype=torch.long).to(device)  # (N_neg, 77)

            neg_embeddings = model.encode_text(neg_tokens)  # (N_neg, 512)
            neg_embed_avg = neg_embeddings.mean(dim=0)  # (512,)
            neg_embed_avg = F.normalize(neg_embed_avg, dim=0)
            neg_embeds_list.append(neg_embed_avg)

    # Stack into matrices
    pos_embeds = torch.stack(pos_embeds_list, dim=0)  # (14, 512)
    neg_embeds = torch.stack(neg_embeds_list, dim=0)  # (14, 512)

    print(f"\n✓ Pre-computed text embeddings:")
    print(f"  Positive embeddings: {pos_embeds.shape}")
    print(f"  Negative embeddings: {neg_embeds.shape}")
    print("="*70 + "\n")

    return pos_embeds, neg_embeds


def compute_probs_from_precomputed(image_embeds, pos_embeds, neg_embeds, logit_scale):
    """
    Compute PNS probabilities using pre-computed text embeddings.
    This is the FAST version that runs inside the evaluation loop.

    Args:
        image_embeds: (B, 512) normalized image embeddings
        pos_embeds: (14, 512) pre-computed positive text embeddings
        neg_embeds: (14, 512) pre-computed negative text embeddings
        logit_scale: learnable temperature parameter

    Returns:
        probs: (B, 14) probabilities for each pathology
    """
    # Compute similarities (simple matrix multiplication since all embeddings are normalized)
    s_pos = image_embeds @ pos_embeds.t()  # (B, 14)
    s_neg = image_embeds @ neg_embeds.t()  # (B, 14)

    # Apply learnable temperature (logit_scale)
    temperature = logit_scale.exp()
    logits_pos = s_pos * temperature  # (B, 14)
    logits_neg = s_neg * temperature  # (B, 14)

    # Stack and apply softmax to get probabilities
    logits = torch.stack([logits_pos, logits_neg], dim=2)  # (B, 14, 2)
    probs = torch.softmax(logits, dim=2)[:, :, 0]  # (B, 14) - probability of positive

    return probs


def evaluate_checkpoint(checkpoint_path, val_loader, tokenizer, device):
    """
    Evaluate a single checkpoint on validation set.

    Returns:
        mean_auc: Mean AUROC across 14 pathologies
        aucs: List of AUROCs for each pathology
    """
    print(f"\nEvaluating: {os.path.basename(checkpoint_path)}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model
    model = PLIPModel(embed_dim=512)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # === PRE-COMPUTE TEXT EMBEDDINGS (ONCE) ===
    pos_embeds, neg_embeds = precompute_text_embeddings(model, tokenizer, device)

    # Get logit_scale for temperature
    logit_scale = model.logit_scale

    # Collect predictions and ground truth
    y_pred_list = []
    y_true_list = []

    print("Running inference on validation set...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="  Evaluating", leave=False):
            images = images.to(device)

            # Encode images (ONLY operation in the loop)
            img_embeds = model.encode_image(images)  # (B, 512) normalized

            # Compute probabilities using pre-computed text embeddings (FAST!)
            probs = compute_probs_from_precomputed(
                img_embeds, pos_embeds, neg_embeds, logit_scale
            )  # (B, 14)

            y_pred_list.append(probs.cpu().numpy())
            y_true_list.append(labels.numpy())

    # Concatenate all batches
    y_pred = np.concatenate(y_pred_list, axis=0)  # (N, 14)
    y_true = np.concatenate(y_true_list, axis=0)  # (N, 14)

    # Compute AUROC for each pathology
    aucs = []
    valid_aucs = []

    print("\nAUROC per pathology:")
    for i, label in enumerate(CHEXPERT_LABELS):
        # Skip if no positive samples (can't compute AUROC)
        n_pos = y_true[:, i].sum()
        n_neg = len(y_true) - n_pos

        if n_pos == 0 or n_neg == 0:
            print(f"  {label:30s}: SKIPPED (no positive or no negative samples)")
            aucs.append(np.nan)
        else:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
                valid_aucs.append(auc)
                print(f"  {label:30s}: {auc:.4f}")
            except Exception as e:
                print(f"  {label:30s}: ERROR - {e}")
                aucs.append(np.nan)

    # Mean AUROC (only over valid labels)
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0

    print(f"\n  Mean AUROC: {mean_auc:.4f}")

    return mean_auc, aucs


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints on validation set (CheXzero strategy)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing checkpoints')
    parser.add_argument('--data_dir', type=str, default='../metadata',
                        help='Directory containing validation data')
    parser.add_argument('--val_h5', type=str, default=None,
                        help='Validation HDF5 file (overrides data_dir)')
    parser.add_argument('--val_labels_csv', type=str, default=None,
                        help='Validation labels CSV (overrides data_dir)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_csv', type=str, default='checkpoint_auroc_results.csv',
                        help='Output CSV file for results')

    args = parser.parse_args()

    # Set up paths
    if args.val_h5 is None:
        args.val_h5 = os.path.join(args.data_dir, 'chexpert_plus_valid.h5')
    if args.val_labels_csv is None:
        args.val_labels_csv = os.path.join(args.data_dir, 'chexpert_plus_valid_labels.csv')

    print("="*70)
    print("CHECKPOINT EVALUATION - CheXzero Zero-Shot Strategy")
    print("="*70)
    print(f"Checkpoint dir:    {args.checkpoint_dir}")
    print(f"Validation H5:     {args.val_h5}")
    print(f"Validation labels: {args.val_labels_csv}")
    print(f"Output CSV:        {args.output_csv}")
    print(f"Device:            {args.device}")
    print()
    print("CheXzero Strategy:")
    print(f"  ✓ Ensemble of Prompts ({len(POS_TEMPLATES)} pos, {len(NEG_TEMPLATES)} neg)")
    print(f"  ✓ Pre-computed text embeddings (efficiency)")
    print(f"  ✓ Positive-Negative Similarity (PNS)")
    print(f"  ✓ Learnable temperature (logit_scale)")
    print(f"  ✓ U-Positive label strategy (uncertain = positive)")
    print()

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

    # Find all checkpoints
    checkpoint_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_step*.pt')))

    if len(checkpoint_paths) == 0:
        print(f"ERROR: No checkpoints found in {args.checkpoint_dir}")
        print("       Looking for files matching: checkpoint_step*.pt")
        sys.exit(1)

    print(f"Found {len(checkpoint_paths)} checkpoints\n")
    print("="*70)

    # Evaluate all checkpoints
    results = []

    for ckpt_path in checkpoint_paths:
        # Extract step number from filename
        basename = os.path.basename(ckpt_path)
        try:
            step = int(basename.split('step')[1].split('.pt')[0])
        except:
            print(f"Warning: Could not parse step number from {basename}, skipping")
            continue

        # Evaluate
        mean_auc, aucs = evaluate_checkpoint(ckpt_path, val_loader, tokenizer, device)

        # Store results
        result = {
            'step': step,
            'checkpoint': ckpt_path,
            'mean_auroc': mean_auc
        }

        # Add individual AUROCs
        for label, auc in zip(CHEXPERT_LABELS, aucs):
            result[f'{label}_auroc'] = auc

        results.append(result)

        print(f"  Step {step}: Mean AUROC = {mean_auc:.4f}")
        print("="*70)

    print("\n" + "="*70)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('step')

    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Results saved to: {args.output_csv}")

    # Find best checkpoint
    best_idx = results_df['mean_auroc'].idxmax()
    best_row = results_df.iloc[best_idx]

    print("\n" + "="*70)
    print("🏆 BEST CHECKPOINT (by Mean AUROC):")
    print("="*70)
    print(f"  Step:        {int(best_row['step'])}")
    print(f"  Mean AUROC:  {best_row['mean_auroc']:.4f}")
    print(f"  Path:        {best_row['checkpoint']}")
    print()
    print("Individual AUROCs:")
    for label in CHEXPERT_LABELS:
        auc = best_row[f'{label}_auroc']
        if not np.isnan(auc):
            print(f"  {label:30s}: {auc:.4f}")
    print("="*70)

    # Save best checkpoint info
    best_info = {
        'best_step': int(best_row['step']),
        'best_checkpoint': best_row['checkpoint'],
        'mean_auroc': best_row['mean_auroc']
    }

    import json
    with open('best_checkpoint_info.json', 'w') as f:
        json.dump(best_info, f, indent=2)

    print(f"\n✓ Best checkpoint info saved to: best_checkpoint_info.json")
    print("\n" + "="*70)
    print("✓ Evaluation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
