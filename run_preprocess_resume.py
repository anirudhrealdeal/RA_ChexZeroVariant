#!/usr/bin/env python3
"""
Resume Preprocessing - Skip CheXpert, Process ReXGradient + Merge

This script resumes from where the previous job stopped:
- Skips CheXpert-Plus processing (already done)
- Re-extracts CheXpert-Plus metadata from CSV (for merge)
- Processes ReXGradient dataset
- Merges both HDF5 files
- Creates combined CSV and validation set

Usage:
    python run_preprocess_resume.py
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import h5py
from pathlib import Path

# Import existing functions
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_process import img_to_hdf5
from run_preprocess_combined import (
    prepare_chexpert_plus_data,
    prepare_rexgradient_data,
    merge_hdf5_files,
    create_combined_csv,
    verify_alignment
)


def main():
    parser = argparse.ArgumentParser(
        description='Resume preprocessing: skip CheXpert, process ReXGradient + merge',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # CheXpert-Plus paths (for metadata extraction only)
    parser.add_argument('--cp_csv', type=str,
                        default='/cbica/projects/CXR/data/CheXpert/chexpertplus/df_chexpert_plus_240401.csv',
                        help='Path to CheXpert-Plus CSV (for metadata)')
    parser.add_argument('--cp_image_base', type=str,
                        default='/cbica/projects/CXR/data/CheXpert/chexpertplus/PNG/PNG',
                        help='Base path to CheXpert-Plus images')

    # ReXGradient paths
    parser.add_argument('--rx_json', type=str,
                        default='/cbica/projects/CXR/data/RexGradient/data/metadata/train_metadata_view_position.json',
                        help='Path to ReXGradient JSON')
    parser.add_argument('--rx_image_base', type=str,
                        default='/cbica/projects/CXR/data/RexGradient/data/images',
                        help='Base path to ReXGradient images')

    # Output paths
    parser.add_argument('--output_dir', type=str,
                        default='metadata',
                        help='Directory with existing/output files')
    parser.add_argument('--resolution', type=int, default=320,
                        help='Target image resolution')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("RESUME PREPROCESSING (Skip CheXpert, Process ReXGradient + Merge)")
    print("="*80)

    # Check that CheXpert HDF5 already exists
    cp_h5_path = os.path.join(args.output_dir, 'chexpert_plus_train.h5')
    if not os.path.exists(cp_h5_path):
        print(f"ERROR: {cp_h5_path} not found!")
        print("Run the full preprocessing script first.")
        return

    print(f"\n✓ Found existing CheXpert HDF5: {cp_h5_path}")

    # ========== STEP 1: Extract CheXpert Metadata (Fast - No Image Processing) ==========
    print(f"\n{'='*80}")
    print("Extracting CheXpert-Plus metadata from CSV (no image processing)...")
    print(f"{'='*80}")

    cp_paths, cp_impressions, cp_patient_ids = prepare_chexpert_plus_data(
        csv_path=args.cp_csv,
        image_base_path=args.cp_image_base,
        split='train'
    )

    print(f"✓ Extracted {len(cp_paths)} CheXpert-Plus metadata entries")

    # ========== STEP 2: Process ReXGradient (Skip if already done) ==========
    rx_h5_path = os.path.join(args.output_dir, 'rexgradient_train.h5')

    if os.path.exists(rx_h5_path):
        print(f"\n{'='*80}")
        print(f"✓ Found existing ReXGradient HDF5: {rx_h5_path}")
        print(f"Skipping ReXGradient image processing (already complete)")
        print(f"{'='*80}")

        # Still need to extract metadata for CSV creation
        rx_paths, rx_impressions, rx_patient_ids = prepare_rexgradient_data(
            json_path=args.rx_json,
            image_base_path=args.rx_image_base
        )
    else:
        print(f"\n{'='*80}")
        print(f"Processing ReXGradient dataset...")
        print(f"{'='*80}")

        rx_paths, rx_impressions, rx_patient_ids = prepare_rexgradient_data(
            json_path=args.rx_json,
            image_base_path=args.rx_image_base
        )

        # Convert to HDF5
        print(f"\nConverting ReXGradient images to HDF5...")
        img_to_hdf5(rx_paths, rx_h5_path, resolution=args.resolution)
        print(f"✓ Saved: {rx_h5_path}")

    # ========== STEP 3: Merge HDF5 Files ==========
    combined_h5_path = os.path.join(args.output_dir, 'combined_train.h5')
    merge_hdf5_files(cp_h5_path, rx_h5_path, combined_h5_path)

    # ========== STEP 4: Create Combined CSV ==========
    combined_csv_path = os.path.join(args.output_dir, 'combined_train.csv')
    combined_df = create_combined_csv(
        cp_paths, cp_impressions, cp_patient_ids,
        rx_paths, rx_impressions, rx_patient_ids,
        combined_csv_path
    )

    # Verify alignment
    print(f"\nVerifying HDF5-CSV alignment...")
    verify_alignment(combined_h5_path, combined_csv_path)

    # ========== STEP 5: Process Validation Set ==========
    print(f"\n{'='*80}")
    print("Processing validation set...")
    print(f"{'='*80}")

    cp_val_paths, cp_val_impressions, cp_val_patient_ids = prepare_chexpert_plus_data(
        csv_path=args.cp_csv,
        image_base_path=args.cp_image_base,
        split='valid'
    )

    cp_val_h5_path = os.path.join(args.output_dir, 'chexpert_plus_valid.h5')
    print(f"\nConverting validation images to HDF5...")
    img_to_hdf5(cp_val_paths, cp_val_h5_path, resolution=args.resolution)
    print(f"✓ Saved: {cp_val_h5_path}")

    # Save validation CSV
    cp_val_csv = os.path.join(args.output_dir, 'chexpert_plus_valid.csv')
    val_df = pd.DataFrame({
        'image_path': cp_val_paths,
        'patient_id': cp_val_patient_ids,
        'impression': cp_val_impressions,
        'dataset_index': range(len(cp_val_paths))
    })
    val_df.to_csv(cp_val_csv, index=False)
    print(f"✓ Saved validation CSV: {cp_val_csv}")

    # Verify validation alignment
    print(f"\nVerifying validation HDF5-CSV alignment...")
    verify_alignment(cp_val_h5_path, cp_val_csv)

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)

    print(f"\n Generated Files:")
    print(f"\n  Training:")
    print(f"    Images:  {combined_h5_path}")
    print(f"    Metadata: {combined_csv_path}")

    print(f"\n  Validation:")
    print(f"    Images:  {cp_val_h5_path}")
    print(f"    Metadata: {cp_val_csv}")

    print(f"\n Dataset Statistics:")
    print(f"    Training Set:   {len(combined_df)} samples")
    print(f"      - CheXpert-Plus: {len(cp_paths)} images")
    print(f"      - ReXGradient:   {len(rx_paths)} images")
    print(f"    Validation Set: {len(val_df)} samples")

    print(f"\n CSV Structure:")
    print(f"    Columns: {list(combined_df.columns)}")


if __name__ == "__main__":
    main()
