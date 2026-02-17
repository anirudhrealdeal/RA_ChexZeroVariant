#!/usr/bin/env python3
"""
PLIP-Style Training: Pretrained DINOv3 ViT-B/16 + Pretrained CLIP Text Encoder
Contrastive learning on CheXpert-Plus + ReXGradient datasets (462K image-text pairs)

Training Strategy (following training_strategy.md):
- Vision: Pretrained DINOv3 ViT-B/16 (expects 224x224, outputs 768-dim)
- Text: Pretrained CLIP ViT-B/32 text encoder (outputs 512-dim)
- Projection: Vision 768 → 512 (simple linear layer to match text space)
- Shared embedding space: 512-dim (CLIP's latent space)
- Images stored at 320x320 in HDF5, resized to 224x224 during training
- CheXzero hyperparameters: batch_size=64, lr=1e-4, SGD with momentum=0.9
- PLIP strategy: 25,000 total steps, validate/save every 500 steps
- Augmentations (train only): RandomResizedCrop(224, scale=0.9-1.0) + RandomHorizontalFlip
- Normalization: CLIP stats (not ImageNet)

Usage:
    python train_plip.py --data_dir metadata --checkpoint_dir checkpoints
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import Compose, Resize, RandomResizedCrop, RandomHorizontalFlip, Normalize, InterpolationMode
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
            input_resolution: Target resolution for model input (224 for pretrained DINOv3)
            is_training: If True, applies training augmentations (RandomResizedCrop + Flip)
        """
        self.h5_path = h5_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_resolution = input_resolution
        self.is_training = is_training

        # Lazy HDF5 file handle (opened once per DataLoader worker for performance)
        # Opening/closing file on every __getitem__ call creates massive I/O bottleneck
        # With 8 workers, this reduces file opens from ~462k/epoch to just 8 total
        self._h5_file = None

        # CLIP normalization statistics (from OpenAI CLIP weights)
        # These are the stats the CLIP text encoder was trained with
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]

        # Training augmentations (following PLIP/medical CLIP best practices)
        if is_training:
            self.transform = Compose([
                RandomResizedCrop(
                    input_resolution,
                    scale=(0.9, 1.0),  # Slight crop variation for robustness
                    interpolation=InterpolationMode.BICUBIC
                ),
                RandomHorizontalFlip(p=0.5),  # Anatomically safe for chest X-rays
                Normalize(mean=clip_mean, std=clip_std)
            ])
        else:
            # Validation: no augmentation, just resize and normalize
            self.transform = Compose([
                Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=clip_mean, std=clip_std)
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

        # Apply transforms: augmentation + resize + CLIP normalization
        # Training: RandomResizedCrop + HorizontalFlip + Normalize
        # Validation: Resize + Normalize only
        image = self.transform(image)

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
    PLIP-style model: Pretrained DINOv3 vision encoder + Pretrained CLIP text encoder
    with contrastive learning objective

    Architecture (following training_strategy.md):
    - Vision: Pretrained DINOv3 ViT-B/16 (expects 224x224 input, outputs 768-dim)
    - Text: Pretrained CLIP ViT-B/32 text encoder (outputs 512-dim)
    - Projection: Simple linear layer mapping vision 768 → 512 to match text space
    - Shared embedding space: 512-dim (CLIP's latent space)
    """

    def __init__(self, embed_dim=512, temperature=0.07):
        """
        Args:
            embed_dim: Shared embedding dimension (512 to match CLIP text space)
            temperature: Initial temperature parameter for contrastive loss
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.ones([]) * temperature)

        # Load pretrained DINOv3 vision encoder (ViT-B/16)
        # Pretrained on large-scale web data (LVD-1689M dataset)
        # Expects 224x224 input, outputs 768-dim features
        print("Loading pretrained DINOv3 ViT-B/16 from local checkpoint...")
        import timm

        # Create DINOv3 ViT-B/16 architecture (without pretrained weights)
        self.vision_encoder = timm.create_model('vit_base_patch16_dinov3', pretrained=False)

        # Load checkpoint from local file
        checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'encoders',
                                       'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load state dict into model
        missing_keys, unexpected_keys = self.vision_encoder.load_state_dict(checkpoint, strict=False)
        if missing_keys:
            print(f"   Warning: Missing keys: {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"   Warning: Unexpected keys: {len(unexpected_keys)} keys")
        print(f"   ✓ DINOv3 ViT-B/16 loaded from: {checkpoint_path}")

        # Freeze the vision encoder (using pretrained features only)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # DINOv3 ViT-B/16 outputs 768-dim features
        vision_width = 768

        # Simple linear projection: 768 → 512 (no MLP, following training_strategy.md)
        # This "bridge" layer aligns DINOv3 features to CLIP's text latent space
        self.vision_projection = nn.Linear(vision_width, embed_dim)

        # Load CheXzero's medically fine-tuned text encoder (ViT-B/32)
        # This text encoder was fine-tuned on CXR reports for medical language understanding
        print("Loading CheXzero's medically fine-tuned text encoder (ViT-B/32)...")

        # First load base CLIP architecture
        clip_model, _ = load_clip("ViT-B/32", device="cpu", jit=False)

        # Load CheXzero checkpoint with medical fine-tuning
        # Try both possible filenames (with and without step/metric suffix)
        chexzero_dir = os.path.expanduser("~/.cache/chexzero")
        possible_names = [
            "best_64_5e-05_original_22000_0.864.pt",
            "best_64_5e-05_original.pt"
        ]

        chexzero_path = None
        for name in possible_names:
            path = os.path.join(chexzero_dir, name)
            if os.path.exists(path):
                chexzero_path = path
                break

        if chexzero_path is None:
            raise FileNotFoundError(
                f"CheXzero weights not found in {chexzero_dir}\n"
                f"Please download from Google Drive: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno\n"
                f"Save as: {os.path.join(chexzero_dir, possible_names[0])}"
            )

        print(f"Loading CheXzero weights from: {chexzero_path}")

        chexzero_checkpoint = torch.load(chexzero_path, map_location="cpu")

        # Extract text encoder weights from CheXzero checkpoint
        # CheXzero saves weights with 'model.' or 'module.model.' prefix
        state_dict = chexzero_checkpoint.get('model_state_dict', chexzero_checkpoint)

        # Load CheXzero's fine-tuned text encoder weights into CLIP model
        self._load_chexzero_text_weights(clip_model, state_dict)

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection_clip = clip_model.text_projection

        # Freeze text encoder (using CheXzero's fine-tuned weights as-is)
        for param in [self.token_embedding.parameters(),
                      self.transformer.parameters(),
                      self.ln_final.parameters()]:
            for p in param:
                p.requires_grad = False

        print("✓ CheXzero text encoder loaded successfully")

        # Text encoder outputs 512-dim features (matches our shared embedding space)
        # No additional projection needed since embed_dim=512

        # Learnable logit scale (like CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _load_chexzero_text_weights(self, clip_model, state_dict):
        """
        Load CheXzero's fine-tuned text encoder weights into CLIP model.

        CheXzero checkpoint may have keys with prefixes like:
        - 'model.transformer.resblocks.0.attn.in_proj_weight'
        - 'module.model.transformer.resblocks.0.attn.in_proj_weight'

        We need to strip prefixes and load into the base CLIP model.
        """
        # Extract text encoder keys and strip prefixes
        text_encoder_state = {}

        for key, value in state_dict.items():
            # Remove 'module.' or 'model.' prefixes if present
            clean_key = key.replace('module.model.', '').replace('model.', '').replace('module.', '')

            # Only load text encoder components (exclude vision encoder)
            text_components = ['token_embedding', 'positional_embedding', 'transformer',
                              'ln_final', 'text_projection']

            if any(comp in clean_key for comp in text_components):
                text_encoder_state[clean_key] = value

        # Load weights into CLIP model (only text encoder parts)
        clip_model.load_state_dict(text_encoder_state, strict=False)

        print(f"   Loaded {len(text_encoder_state)} text encoder parameters from CheXzero")

    def encode_image(self, images):
        """
        Encode images using pretrained DINOv3 + simple linear projection

        Args:
            images: (B, 3, 224, 224) tensor (pretrained DINOv3 ViT-B/16 expects 224x224)

        Returns:
            (B, 512) normalized embeddings
        """
        # DINOv3 ViT-B/16 encoding (pretrained on large-scale web data)
        # timm's forward_features returns: (B, num_tokens, 768)
        # where num_tokens = 1 (CLS) + 196 (14x14 patches) + 4 (register tokens) = 201
        features_all = self.vision_encoder.forward_features(images)

        # Extract CLS token embedding (first token)
        # Shape: (B, 201, 768) -> take [:, 0, :] -> (B, 768)
        features = features_all[:, 0, :]  # (B, 768)

        # Simple linear projection to CLIP's 512-dim latent space
        embeddings = self.vision_projection(features)  # (B, 512)

        # L2 normalize (standard for contrastive learning)
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

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
def validate(model, dataloader, device, verbose=True):
    """Validate model"""
    model.eval()

    total_loss = 0
    num_batches = 0

    for images, text_tokens in tqdm(dataloader, desc="Validating", disable=not verbose):
        images = images.to(device)
        text_tokens = text_tokens.to(device)

        # Forward pass
        with torch.no_grad():
            image_embeddings, text_embeddings, logit_scale = model(images, text_tokens)
            loss = contrastive_loss(image_embeddings, text_embeddings, logit_scale)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)

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

    # Training hyperparameters (from training_strategy.md + CheXzero paper)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (CheXzero paper best model uses 64)')
    parser.add_argument('--max_steps', type=int, default=25000,
                        help='Total training steps (PLIP strategy, not epoch-based)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (CheXzero uses 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (CheXzero uses 0.9)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adamw'],
                        help='Optimizer type (CheXzero uses SGD)')

    # Output paths (PLIP strategy: save by steps)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps (PLIP: 500)')
    parser.add_argument('--val_steps', type=int, default=500,
                        help='Validate every N steps (PLIP: 500, select best model)')

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
    if is_main_process:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set device
    if world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if is_main_process:
        print(f"Using device: {device}")
        print(f"World size: {world_size} GPUs")

    # Initialize tokenizer
    if is_main_process:
        print("Loading tokenizer...")
    tokenizer = SimpleTokenizer()

    # Create datasets
    if is_main_process:
        print(f"\nLoading training data from {args.train_h5}...")
    train_dataset = CXRDataset(args.train_h5, args.train_csv, tokenizer, is_training=True)

    if is_main_process:
        print(f"\nLoading validation data from {args.val_h5}...")
    val_dataset = CXRDataset(args.val_h5, args.val_csv, tokenizer, is_training=False)

    # Create distributed samplers for multi-GPU training
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    if is_main_process:
        print("\nInitializing model...")
    model = PLIPModel(embed_dim=args.embed_dim, temperature=args.temperature)
    model = model.to(device)

    # Wrap model with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process:
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

    # Training loop - PLIP strategy: step-based (not epoch-based), validate frequently
    if is_main_process:
        print(f"\nStarting training for {args.max_steps} steps (PLIP strategy: step-based, frequent validation)...")
        print(f"  Dataset size: {len(train_dataset)} samples")
        print(f"  Batch size: {args.batch_size} per GPU")
        print(f"  Effective batch size: {args.batch_size * world_size}")
        print(f"  Steps per epoch: ~{len(train_loader)}")
        print(f"  Estimated epochs: ~{args.max_steps / len(train_loader):.2f}")

    best_val_loss = float('inf')
    global_step = 0
    running_loss = 0
    num_loss_steps = 0

    # Track training metrics for plotting
    training_metrics = []

    model.train()
    pbar = tqdm(total=args.max_steps, desc="Training", disable=not is_main_process)

    # Infinite data loader (keep cycling through epochs until max_steps)
    train_iter = iter(train_loader)

    while global_step < args.max_steps:
        try:
            images, text_tokens = next(train_iter)
        except StopIteration:
            # Restart data loader for new epoch
            train_iter = iter(train_loader)
            images, text_tokens = next(train_iter)

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
        running_loss += loss.item()
        num_loss_steps += 1
        global_step += 1

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({'loss': loss.item(), 'step': global_step})

        # Validate every N steps (PLIP strategy: catch overfitting early, select best model)
        if global_step % args.val_steps == 0:
            # Synchronize all processes before validation
            if world_size > 1:
                dist.barrier()

            avg_train_loss = running_loss / num_loss_steps
            val_loss = validate(model, val_loader, device, verbose=is_main_process)

            if is_main_process:
                print(f"\n  Step {global_step}/{args.max_steps}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

                # Log metrics for plotting
                training_metrics.append({
                    'step': global_step,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss
                })

            # Reset running loss
            running_loss = 0
            num_loss_steps = 0

            # Save best model based on validation loss (only on main process)
            if is_main_process and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                # Extract model state dict (unwrap DDP if needed)
                model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    'step': global_step,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'args': args
                }, best_checkpoint_path)
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")

            model.train()  # Back to training mode

        # Save checkpoint every N steps (PLIP strategy, only on main process)
        if global_step % args.save_steps == 0 and is_main_process:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_step{global_step}.pt')
            # Extract model state dict (unwrap DDP if needed)
            model_state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                'step': global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    pbar.close()

    # Save training metrics to CSV for plotting (only on main process)
    if is_main_process:
        metrics_df = pd.DataFrame(training_metrics)
        metrics_csv_path = os.path.join(args.checkpoint_dir, 'training_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"\nTraining metrics saved to: {metrics_csv_path}")

        print("\nTraining complete!")
        print(f"Total steps: {global_step}")
        print(f"Best validation loss: {best_val_loss:.4f}")

    # Cleanup distributed training
    cleanup_distributed()


if __name__ == '__main__':
    main()
