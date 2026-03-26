#!/usr/bin/env python3
"""
CheXzero-style Training: Standard CLIP ViT-B/32 (vision + text) fine-tuned end-to-end.
Contrastive learning on CheXpert-Plus + ReXGradient datasets (462K image-text pairs).

Research question: CheXzero trained on ~200K pairs → 0.864 AUROC.
Does 2.3x more data (462K) improve performance with the same architecture?

Architecture:
- Vision: CLIP ViT-B/32 visual encoder (512-dim output, no projection needed)
- Text:   CLIP ViT-B/32 text encoder (512-dim output)
- Shared embedding space: 512-dim (CLIP's native jointly-pretrained space)
- Both encoders fully trainable end-to-end (CheXzero approach)

Training:
- CheXzero hyperparameters: batch_size=64, lr=1e-4, SGD with momentum=0.9
- Epoch-based: 4 epochs, save every 100 batches, log every 10 batches
- Normalization: CXR-specific stats (mean=101.48761, std=83.43944), Normalize then Resize

Usage:
    python train_plip.py --data_dir metadata --checkpoint_dir checkpoints
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Import CLIP components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from clip import load as load_clip
from simple_tokenizer import SimpleTokenizer


class CXRDataset(Dataset):
    """Dataset for chest X-ray images and impressions with CLIP-style augmentations"""

    def __init__(self, h5_path, csv_path, tokenizer, max_length=77, input_resolution=224, is_training=True):
        """
        Args:
            h5_path: Path to HDF5 file with images (stored at 320x320)
            csv_path: Path to CSV file with impressions
            tokenizer: Text tokenizer
            max_length: Maximum token length
            input_resolution: Target resolution for CLIP ViT-B/32 (224)
            is_training: Unused — kept for API compatibility
        """
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_resolution = input_resolution

        # Lazy HDF5 file handle (opened once per DataLoader worker for performance)
        # Opening/closing file on every __getitem__ call creates massive I/O bottleneck
        # With 8 workers, this reduces file opens from ~462k/epoch to just 8 total
        self._h5_file = None

        # CheXzero normalization statistics (CXR-specific, from original CheXzero repo)
        # Original uses [0,255] range: mean=101.48761, std=83.43944
        # Adjusted for our [0,1] pipeline (divided by 255)
        cxr_mean = [101.48761 / 255.0] * 3
        cxr_std  = [83.43944  / 255.0] * 3

        # Original CheXzero order: Normalize THEN Resize
        self.transform = Compose([
            Normalize(mean=cxr_mean, std=cxr_std),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])

        # Load CSV metadata
        self.df = pd.read_csv(csv_path)

        # Verify HDF5 alignment (temporary file open for validation)
        with h5py.File(h5_path, 'r') as f:
            h5_count = len(f['cxr'])

        if h5_count != len(self.df):
            raise ValueError(f"HDF5 ({h5_count}) and CSV ({len(self.df)}) size mismatch!")

        mode = "training" if is_training else "validation"
        print(f"Loaded {mode} dataset with {len(self.df)} samples (target resolution: {input_resolution}x{input_resolution})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Lazy open HDF5 file (once per worker for massive performance gain)
        # Each DataLoader worker gets its own file handle (thread-safe)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r')

        # Load image from cached HDF5 handle (no open/close overhead)
        image = self._h5_file['cxr'][idx]  # Shape: (320, 320) grayscale

        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image).float() / 255.0

        # Convert grayscale to RGB by repeating channels
        # Both DINOv3 and CLIP expect 3-channel input
        if image.dim() == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)  # (H, W) -> (3, H, W)
        elif image.dim() == 3:
            # Already (H, W, C), rearrange to (C, H, W)
            image = image.permute(2, 0, 1)

        # Apply transforms: Normalize then Resize (CheXzero order)
        image = self.transform(image)

        # Get impression text
        impression = str(self.df.iloc[idx]['impression'])
        if pd.isna(impression) or impression == '' or impression == 'nan':
            impression = "No findings"

        # Tokenize text — original CheXzero wraps with SOT/EOT tokens
        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(impression) + [eot_token]

        # Truncate to max_length, keeping EOT at the end
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            tokens[self.max_length - 1] = eot_token

        # Pad to max_length
        tokens = tokens + [0] * (self.max_length - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)

        return image, tokens


class PLIPModel(nn.Module):
    """
    CheXzero-style model: Standard CLIP ViT-B/32 fine-tuned end-to-end on CXR data.

    Same approach as CheXzero but trained on a larger dataset:
    - CheXzero: ~200K CXR pairs → Mean AUROC 0.864
    - Ours:      462K CXR pairs (CheXpert-Plus + ReXGradient)

    Architecture:
    - Vision: CLIP ViT-B/32 visual encoder (512-dim output, no projection needed)
    - Text:   CLIP ViT-B/32 text encoder (512-dim output)
    - Shared embedding space: 512-dim (CLIP's native jointly-pretrained space)
    - Both encoders fully trainable end-to-end
    """

    def __init__(self, embed_dim=512, temperature=0.07):
        super().__init__()

        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

        # Load standard CLIP ViT-B/32 — pure pretrained weights, no CheXzero fine-tuning
        # Vision + text jointly pretrained on 400M image-text pairs: embedding spaces are aligned
        print("Loading standard CLIP ViT-B/32 (vision + text, jointly pretrained on 400M pairs)...")
        clip_model, _ = load_clip("ViT-B/32", device="cpu", jit=False)

        # Vision encoder: outputs 512-dim directly, no projection layer needed
        self.visual = clip_model.visual

        # Text encoder components
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection_clip = clip_model.text_projection

        total_params = sum(p.numel() for p in clip_model.parameters())
        print(f"   ✓ CLIP ViT-B/32 loaded ({total_params:,} parameters)")
        print("   ✓ Both encoders trainable (end-to-end fine-tuning, CheXzero approach)")

        # Learnable logit scale (like CLIP / CheXzero)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, images):
        """
        Encode images using CLIP ViT-B/32 visual encoder.

        Args:
            images: (B, 3, 224, 224)

        Returns:
            (B, 512) normalized embeddings
        """
        embeddings = self.visual(images)  # (B, 512) — CLIP vision outputs 512-dim directly
        return F.normalize(embeddings, dim=-1)

    def encode_text(self, text_tokens):
        """
        Encode text using CLIP text encoder (no additional projection needed)

        Args:
            text_tokens: (B, 77) tensor of token IDs (CLIP's context length limit)

        Returns:
            (B, 512) normalized embeddings
        """
        # CLIP text encoding
        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # Take features from [EOS] token and apply CLIP's text projection
        # This outputs 512-dim embeddings (CLIP's native latent space)
        embeddings = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection_clip

        # L2 normalize (standard for contrastive learning)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def forward(self, images, text_tokens):
        """
        Forward pass: compute image and text embeddings in shared 512-dim space

        Args:
            images: (B, 3, 224, 224) - augmented/resized from 320x320 HDF5 storage
            text_tokens: (B, 77) - tokenized impressions (CLIP's 77 token limit)

        Returns:
            image_embeddings: (B, 512) - L2 normalized in CLIP's latent space
            text_embeddings: (B, 512) - L2 normalized in CLIP's latent space
            logit_scale: scalar - learnable temperature for contrastive loss
        """
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(text_tokens)

        return image_embeddings, text_embeddings, self.logit_scale.exp()


def contrastive_loss(image_embeddings, text_embeddings, logit_scale):
    """
    Compute symmetric contrastive loss (CLIP-style)

    Args:
        image_embeddings: (B, D) normalized
        text_embeddings: (B, D) normalized
        logit_scale: scalar temperature

    Returns:
        loss: scalar
    """
    # Compute similarity matrix
    logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()  # (B, B)
    logits_per_text = logits_per_image.t()  # (B, B)

    # Labels: diagonal is positive pairs
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size, device=image_embeddings.device)

    # Cross-entropy loss in both directions
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    # Symmetric loss
    loss = (loss_i2t + loss_t2i) / 2

    return loss


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model"""
    model.eval()

    total_loss = 0
    num_batches = 0

    for images, text_tokens in tqdm(dataloader, desc="Validating"):
        images = images.to(device)
        text_tokens = text_tokens.to(device)

        # Forward pass
        image_embeddings, text_embeddings, logit_scale = model(images, text_tokens)
        loss = contrastive_loss(image_embeddings, text_embeddings, logit_scale)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(
        description='Train PLIP-style model on CXR data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument('--data_dir', type=str, default='metadata',
                        help='Directory containing preprocessed data')
    parser.add_argument('--train_h5', type=str, default=None,
                        help='Training HDF5 file (overrides data_dir)')
    parser.add_argument('--train_csv', type=str, default=None,
                        help='Training CSV file (overrides data_dir)')
    parser.add_argument('--val_h5', type=str, default=None,
                        help='Validation HDF5 file (overrides data_dir)')
    parser.add_argument('--val_csv', type=str, default=None,
                        help='Validation CSV file (overrides data_dir)')

    # Model hyperparameters (from training_strategy.md)
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Shared embedding dimension (512 = CLIP text latent space)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Initial temperature for contrastive loss')

    # Training hyperparameters (CheXzero original)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (CheXzero best model uses 64)')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Number of training epochs (CheXzero uses 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (CheXzero uses 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (CheXzero uses 0.9)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adamw'],
                        help='Optimizer type (CheXzero uses SGD)')

    # Output paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save checkpoint every N batches (CheXzero uses 100)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log loss every N batches (CheXzero uses 10)')
    parser.add_argument('--val_interval', type=int, default=500,
                        help='Validate every N batches')

    # System
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Set up paths
    if args.train_h5 is None:
        args.train_h5 = os.path.join(args.data_dir, 'combined_train.h5')
    if args.train_csv is None:
        args.train_csv = os.path.join(args.data_dir, 'combined_train.csv')
    if args.val_h5 is None:
        args.val_h5 = os.path.join(args.data_dir, 'chexpert_plus_valid.h5')
    if args.val_csv is None:
        args.val_csv = os.path.join(args.data_dir, 'chexpert_plus_valid.csv')

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = SimpleTokenizer()

    # Create datasets
    print(f"\nLoading training data from {args.train_h5}...")
    train_dataset = CXRDataset(args.train_h5, args.train_csv, tokenizer, is_training=True)

    print(f"\nLoading validation data from {args.val_h5}...")
    val_dataset = CXRDataset(args.val_h5, args.val_csv, tokenizer, is_training=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    print("\nInitializing model...")
    model = PLIPModel(embed_dim=args.embed_dim, temperature=args.temperature)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer (CheXzero uses SGD with momentum)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr
        )

    # No learning rate scheduler (CheXzero uses constant LR)

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop - CheXzero approach: epoch-based
    print(f"\nStarting training for {args.num_epochs} epochs (CheXzero approach)...")
    print(f"  Dataset size: {len(train_dataset)} samples")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Total batches: {len(train_loader) * args.num_epochs}")

    best_val_loss = float('inf')
    batch_ct = 0
    running_loss = 0.0

    # Track training metrics for plotting
    training_metrics = []

    for epoch in range(args.num_epochs):
        model.train()
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")

        for images, text_tokens in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            text_tokens = text_tokens.to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast():
                image_embeddings, text_embeddings, logit_scale = model(images, text_tokens)
                loss = contrastive_loss(image_embeddings, text_embeddings, logit_scale)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            batch_ct += 1

            # Log every N batches (CheXzero: every 10)
            if batch_ct % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                print(f"  Batch {batch_ct}: loss = {avg_loss:.4f}")
                running_loss = 0.0

            # Validate every N batches
            if batch_ct % args.val_interval == 0:
                val_loss = validate(model, val_loader, device)
                print(f"  [Batch {batch_ct}] Val Loss = {val_loss:.4f}")

                training_metrics.append({
                    'batch': batch_ct,
                    'epoch': epoch + 1,
                    'val_loss': val_loss
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'batch': batch_ct,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'args': args
                    }, best_checkpoint_path)
                    print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")

                model.train()

            # Save checkpoint every N batches (CheXzero: every 100)
            if batch_ct % args.save_interval == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_batch{batch_ct}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'batch': batch_ct,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")

        print(f"  Epoch {epoch + 1} complete.")

    # Save training metrics to CSV for plotting
    metrics_df = pd.DataFrame(training_metrics)
    metrics_csv_path = os.path.join(args.checkpoint_dir, 'training_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nTraining metrics saved to: {metrics_csv_path}")

    print("\nTraining complete!")
    print(f"Total batches: {batch_ct}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
