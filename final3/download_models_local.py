#!/usr/bin/env python3
"""
Download pretrained models LOCALLY (on your laptop/desktop with internet).
Then transfer the cache directories to CUBIC.

Run this on your LOCAL machine (not CUBIC):
    python3 download_models_local.py
"""

import os
import sys
import torch
import urllib.request
import ssl

print("="*70)
print("DOWNLOADING PRETRAINED MODELS LOCALLY")
print("="*70)
print()

# Create cache directories
clip_cache = os.path.expanduser("~/.cache/clip")
chexzero_cache = os.path.expanduser("~/.cache/chexzero")

for d in [clip_cache, chexzero_cache]:
    os.makedirs(d, exist_ok=True)

# --- 1. CHECK DINOv3 ---
print("1. Checking DINOv3 ViT-B/16 checkpoint...")
# DINOv3 is in parent directory's encoders folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dinov3_path = os.path.join(project_root, 'encoders', 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')

if os.path.exists(dinov3_path):
    size_mb = os.path.getsize(dinov3_path) / (1024 * 1024)
    print(f"   ✓ DINOv3 found at: {dinov3_path}")
    print(f"   Size: {size_mb:.1f} MB")
else:
    print(f"   ✗ ERROR: DINOv3 not found at: {dinov3_path}")
    print("   Please ensure you downloaded it from Meta")
    sys.exit(1)

print()

# --- 2. CHECK CHEXZERO ---
print("2. Checking CheXzero pretrained weights (~600MB)...")
# Try both possible filenames
chexzero_files = [
    os.path.join(chexzero_cache, "best_64_5e-05_original_22000_0.864.pt"),
    os.path.join(chexzero_cache, "best_64_5e-05_original.pt")
]

chexzero_found = None
for chexzero_file in chexzero_files:
    if os.path.exists(chexzero_file):
        chexzero_found = chexzero_file
        break

if chexzero_found:
    size_mb = os.path.getsize(chexzero_found) / (1024 * 1024)
    print(f"   ✓ CheXzero found: {chexzero_found}")
    print(f"   Size: {size_mb:.1f} MB")
else:
    print(f"   ✗ CheXzero NOT found")
    print("\n   MANUAL DOWNLOAD REQUIRED:")
    print("   1. Go to: https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno?usp=sharing")
    print("   2. Download 'best_64_5e-05_original_22000_0.864.pt'")
    print(f"   3. Save to: {chexzero_files[0]}")
    print("\n   After downloading, run this script again to continue")
    sys.exit(1)

print()

# --- 3. DOWNLOAD BASE CLIP ---
print("3. Downloading base CLIP ViT-B/32 architecture...")

# Add project root to Python path to import clip module
sys.path.insert(0, project_root)

try:
    from clip import load as load_clip

    # This will download the model to ~/.cache/clip/
    print("   Loading CLIP ViT-B/32...")
    model, preprocess = load_clip("ViT-B/32", device="cpu", jit=False)

    print("   ✓ Base CLIP downloaded to ~/.cache/clip/")

    # Verify the file exists and is not empty
    clip_file = os.path.join(clip_cache, "ViT-B-32.pt")
    if os.path.exists(clip_file):
        size_mb = os.path.getsize(clip_file) / (1024 * 1024)
        print(f"   Size: {size_mb:.1f} MB")

except Exception as e:
    print(f"   ✗ CLIP download failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*70)
print("✓ ALL MODELS READY!")
print("="*70)
print()
print("Models located at:")
print(f"  - DINOv3:   {dinov3_path}")
print(f"  - CheXzero: {chexzero_file}")
print(f"  - CLIP:     ~/.cache/clip/")
print()
print("="*70)
print("NEXT STEP: Transfer to CUBIC")
print("="*70)
print()
print("Run these commands to transfer to CUBIC:")
print()
print(f"  # 1. Transfer entire project directory (includes DINOv3 in encoders/)")
print(f"  rsync -avzP {project_root}/ \\")
print(f"    kumaranir@cubic-login.uphs.upenn.edu:~/RA_ChexZeroVariant/")
print()
print(f"  # 2. Transfer CheXzero weights")
print(f"  rsync -avzP ~/.cache/chexzero/ \\")
print(f"    kumaranir@cubic-login.uphs.upenn.edu:~/.cache/chexzero/")
print()
print(f"  # 3. Transfer CLIP cache")
print(f"  rsync -avzP ~/.cache/clip/ \\")
print(f"    kumaranir@cubic-login.uphs.upenn.edu:~/.cache/clip/")
print()
print("After transfer completes, on CUBIC:")
print("  1. Activate your virtual environment")
print("  2. Install dependencies: pip install -r final/requirements.txt")
print("  3. Submit training job: cd final && sbatch run_training_job.sh")
print("="*70)
