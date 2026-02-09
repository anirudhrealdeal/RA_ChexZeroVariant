#!/usr/bin/env python3
"""
PLIP-Style Training: DINOv2 Vision Encoder + CLIP Text Encoder
Contrastive learning on CheXpert-Plus + ReXGradient datasets
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
    """Dataset for chest X-ray images and impressions"""

    def __init__(self, h5_path, csv_path, tokenizer, max_length=77):
        """
        Args:
            h5_path: Path to HDF5 file with images
            csv_path: Path to CSV file with impressions
            tokenizer: Text tokenizer
            max_length: Maximum token length
        """
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load CSV metadata
        self.df = pd.read_csv(csv_path)

        # Verify HDF5 alignment
        with h5py.File(h5_path, 'r') as f:
            h5_count = len(f['cxr'])

        if h5_count != len(self.df):
            raise ValueError(f"HDF5 ({h5_count}) and CSV ({len(self.df)}) size mismatch!")

        print(f"Loaded dataset with {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image from HDF5
        with h5py.File(self.h5_path, 'r') as f:
            image = f['cxr'][idx]  # Shape: (320, 320) grayscale

        # Convert to tensor and normalize to [0, 1]
        image = torch.from_numpy(image).float() / 255.0

        # Convert grayscale to RGB by repeating channels
        # DINOv2 expects 3-channel input
        if image.dim() == 2:
            image = image.unsqueeze(0).repeat(3, 1, 1)  # (H, W) -> (3, H, W)
        elif image.dim() == 3:
            # Already (H, W, C), rearrange to (C, H, W)
            image = image.permute(2, 0, 1)

        # Normalize with ImageNet stats (for DINOv2)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        # Get impression text
        impression = str(self.df.iloc[idx]['impression'])
        if pd.isna(impression) or impression == '' or impression == 'nan':
            impression = "No findings"

        # Tokenize text
        tokens = self.tokenizer.encode(impression)

        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)

        return image, tokens


class PLIPModel(nn.Module):
    """
    PLIP-style model: DINOv2 vision encoder + CLIP text encoder
    with contrastive learning objective
    """

    def __init__(self, embed_dim=512, temperature=0.07):
        """
        Args:
            embed_dim: Embedding dimension for both vision and text
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

        # Load DINOv2 vision encoder (ViT-B/14)
        print("Loading DINOv2 ViT-B/14...")
        self.vision_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        # DINOv2 ViT-B/14 outputs 768-dim features
        vision_width = 768

        # Projection head for vision features
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_width, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Load CLIP text encoder
        print("Loading CLIP text encoder...")
        clip_model, _ = load_clip("ViT-B/32", device="cpu", jit=False)

        self.text_encoder = clip_model.encode_text
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection_clip = clip_model.text_projection

        # CLIP text outputs 512-dim features
        text_width = 512

        # Projection head for text features (if embed_dim != 512)
        if embed_dim != text_width:
            self.text_projection = nn.Sequential(
                nn.Linear(text_width, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        else:
            self.text_projection = nn.Identity()

        # Learnable logit scale (like CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, images):
        """
        Encode images using DINOv2 + projection

        Args:
            images: (B, 3, H, W) tensor

        Returns:
            (B, embed_dim) normalized embeddings
        """
        # DINOv2 encoding
        features = self.vision_encoder(images)  # (B, 768)

        # Project to embed_dim
        embeddings = self.vision_projection(features)  # (B, embed_dim)

        # L2 normalize
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def encode_text(self, text_tokens):
        """
        Encode text using CLIP text encoder + projection

        Args:
            text_tokens: (B, seq_len) tensor of token IDs

        Returns:
            (B, embed_dim) normalized embeddings
        """
        # CLIP text encoding
        x = self.token_embedding(text_tokens)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # Take features from [EOS] token
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_projection_clip

        # Project to embed_dim (if needed)
        embeddings = self.text_projection(x)  # (B, embed_dim)

        # L2 normalize
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def forward(self, images, text_tokens):
        """
        Forward pass: compute image and text embeddings

        Args:
            images: (B, 3, H, W)
            text_tokens: (B, seq_len)

        Returns:
            image_embeddings: (B, embed_dim)
            text_embeddings: (B, embed_dim)
            logit_scale: scalar
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


def train_epoch(model, dataloader, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for images, text_tokens in pbar:
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

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'logit_scale': logit_scale.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


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

    # Model hyperparameters
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Initial temperature for contrastive loss')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Number of warmup epochs')

    # Output paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')

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
    train_dataset = CXRDataset(args.train_h5, args.train_csv, tokenizer)

    print(f"\nLoading validation data from {args.val_h5}...")
    val_dataset = CXRDataset(args.val_h5, args.val_csv, tokenizer)

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

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        else:
            return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        # Print epoch results
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args
            }, best_checkpoint_path)
            print(f"  Saved best model: {best_checkpoint_path}")

        print()

    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
