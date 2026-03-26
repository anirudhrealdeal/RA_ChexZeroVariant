#!/usr/bin/env python3
"""
Pre-download pretrained models on login node (which has internet access).
This caches the models so compute nodes can use them without internet.

Run this ONCE on the login node before submitting training jobs:
    python3 download_pretrained_models.py
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

print("="*70)
print("DOWNLOADING PRETRAINED MODELS TO CACHE")
print("="*70)
print()

# 1. Download DINOv3 ViT-B/16
print("1. Downloading DINOv3 ViT-B/16 from Facebook Research...")
print("   (This may take a few minutes, ~300-400MB)")
try:
    dinov3 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb16')
    print("   ✓ DINOv3 ViT-B/16 downloaded successfully!")
    print(f"   Cached at: {os.path.expanduser('~/.cache/torch/hub')}")
except Exception as e:
    print(f"   ✗ ERROR downloading DINOv3: {e}")
    sys.exit(1)

print()

# 2. Download CLIP ViT-B/32
print("2. Downloading CLIP ViT-B/32 from OpenAI...")
print("   (This may take a minute, ~300MB)")
try:
    from clip import load as load_clip
    clip_model, _ = load_clip("ViT-B/32", device="cpu", jit=False)
    print("   ✓ CLIP ViT-B/32 downloaded successfully!")
    print(f"   Cached at: {os.path.expanduser('~/.cache/clip')}")
except Exception as e:
    print(f"   ✗ ERROR downloading CLIP: {e}")
    sys.exit(1)

print()
print("="*70)
print("✓ ALL PRETRAINED MODELS DOWNLOADED SUCCESSFULLY!")
print("="*70)
print()
print("Models are cached at:")
print(f"  - DINOv3: {os.path.expanduser('~/.cache/torch/hub')}")
print(f"  - CLIP:   {os.path.expanduser('~/.cache/clip')}")
print()
print("You can now submit your training job:")
print("  sbatch run_training_job.sh")
print()
print("The compute nodes will use these cached models (no internet needed).")
print("="*70)
