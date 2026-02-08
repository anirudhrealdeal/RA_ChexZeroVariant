#!/usr/bin/env python3
"""
Combined Dataset Preprocessing for CheXpert-Plus + ReXGradient
Converts images to HDF5 and creates unified CSV with impressions and metadata

Usage:
    python run_preprocess_combined.py

Output:
    /cbica/projects/CXR/processed/
        ├── combined_train.h5            # Training images (CheXpert-Plus + ReXGradient)
        ├── combined_train.csv           # Training metadata with impressions
        ├── chexpert_plus_valid.h5       # Validation images (CheXpert-Plus only)
        └── chexpert_plus_valid.csv      # Validation metadata with impressions
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

# Import existing preprocessing functions
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_process import img_to_hdf5


def prepare_chexpert_plus_data(csv_path, image_base_path, split='train'):
    """
    Prepare CheXpert-Plus data: filter by split and extract paths + impressions + metadata

    Args:
        csv_path: Path to CheXpert-Plus CSV file
        image_base_path: Base directory containing PNG images
        split: 'train', 'valid', or 'test'

    Returns:
        image_paths: List of absolute paths to images
        impressions: List of impression texts
        patient_ids: List of patient IDs
    """
    print(f"\n{'='*80}")
    print(f"Processing CheXpert-Plus ({split} split)...")
    print(f"{'='*80}")

    # Load CSV
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows in CSV: {len(df)}")

    # Filter by split
    df_filtered = df[df['split'] == split].copy()
    print(f"Rows with split='{split}': {len(df_filtered)}")

    if len(df_filtered) == 0:
        print(f"⚠ WARNING: No data found for split='{split}'")
        return [], [], []

    # Extract image paths, impressions, and metadata
    image_paths = []
    impressions = []
    patient_ids = []
    failed_count = 0

    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Extracting data"):
        # Image path (relative to base)
        rel_path = row['path_to_image']
        abs_path = os.path.join(image_base_path, rel_path)

        # Check if file exists
        if not os.path.exists(abs_path):
            failed_count += 1
            if failed_count <= 3:  # Show first 3 failures
                print(f"  Warning: File not found: {abs_path}")
            continue

        # Patient ID
        patient_id = row['deid_patient_id']

        # Impression text
        impression = row['section_impression']

        # Handle NaN impressions
        if pd.isna(impression) or impression == '':
            impression = " "  # Use space for empty impressions (CheXzero convention)

        image_paths.append(abs_path)
        impressions.append(str(impression))
        patient_ids.append(str(patient_id))

    # Summary
    if failed_count > 0:
        print(f"  Warning: {failed_count} files not found (skipped)")

    print(f"Extracted {len(image_paths)} valid image paths")
    print(f"Extracted {len(impressions)} impressions")
    print(f"Extracted {len(patient_ids)} patient IDs")

    return image_paths, impressions, patient_ids


def prepare_rexgradient_data(json_path, image_base_path):
    """
    Prepare ReXGradient data: expand multiple images per study from JSON

    Args:
        json_path: Path to ReXGradient JSON metadata file
        image_base_path: Base directory containing images

    Returns:
        image_paths: List of absolute paths to images (expanded)
        impressions: List of impression texts (replicated per image)
        patient_ids: List of patient IDs (replicated per image)
    """
    print(f"\n{'='*80}")
    print(f"Processing ReXGradient...")
    print(f"{'='*80}")

    # Load JSON
    print(f"Loading JSON: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Total studies in JSON: {len(data)}")

    image_paths = []
    impressions = []
    patient_ids = []
    failed_count = 0

    # Each study can have multiple images (PA, LATERAL, etc.)
    for study_id, study_data in tqdm(data.items(), desc="Expanding images"):
        # Get patient ID
        patient_id = study_data.get('PatientID', 'UNKNOWN')

        # Get impression (same for all images in study)
        impression = study_data.get('Impression', '')
        if pd.isna(impression) or impression == '':
            impression = " "  # Use space for empty impressions

        # Get all image paths for this study
        image_path_list = study_data.get('ImagePath', [])

        # Create one training instance per image
        # (Same impression replicated for all images in the study)
        for rel_path in image_path_list:
            # Convert relative path to absolute
            # rel_path format: "../deid_png/PATIENT/ACCESSION/studies/.../instances/....png"
            clean_path = rel_path.replace('../', '')
            abs_path = os.path.join(image_base_path, clean_path)

            # Check if file exists
            if not os.path.exists(abs_path):
                failed_count += 1
                if failed_count <= 3:  # Show first 3 failures
                    print(f"  Warning: File not found: {abs_path}")
                continue

            image_paths.append(abs_path)
            impressions.append(str(impression))
            patient_ids.append(str(patient_id))

    # Summary
    if failed_count > 0:
        print(f"Warning: {failed_count} files not found (skipped)")

    print(f"Extracted {len(image_paths)} valid image paths")
    print(f"Expanded from {len(data)} studies to {len(image_paths)} images")
    print(f"Each study impression replicated for all its images")

    return image_paths, impressions, patient_ids


def merge_hdf5_files(file1, file2, output_file):
    """
    Merge two HDF5 files into one combined file

    Args:
        file1: Path to first HDF5 file (CheXpert-Plus)
        file2: Path to second HDF5 file (ReXGradient)
        output_file: Path to output merged HDF5 file
    """
    print(f"\n{'='*80}")
    print("Merging HDF5 files...")
    print(f"{'='*80}")

    with h5py.File(file1, 'r') as f1, \
         h5py.File(file2, 'r') as f2, \
         h5py.File(output_file, 'w') as out:

        # Read data from both files
        data1 = f1['cxr'][:]
        data2 = f2['cxr'][:]

        print(f"  CheXpert-Plus: {len(data1)} images, shape {data1.shape}")
        print(f"  ReXGradient:   {len(data2)} images, shape {data2.shape}")

        # Concatenate along first axis (stack images)
        combined = np.concatenate([data1, data2], axis=0)

        print(f"  Combined:      {len(combined)} images, shape {combined.shape}")

        # Save to output file with compression
        out.create_dataset('cxr', data=combined, compression='gzip', compression_opts=4)

        # Store metadata as attributes
        out.attrs['n_chexpert_plus'] = len(data1)
        out.attrs['n_rexgradient'] = len(data2)
        out.attrs['total'] = len(combined)
        out.attrs['resolution'] = data1.shape[1]  # Assuming square images

        print(f"  ✓ Merged: {len(data1)} + {len(data2)} = {len(combined)} images")
        print(f"  ✓ Saved to: {output_file}")


def create_combined_csv(cp_paths, cp_impressions, cp_patient_ids,
                        rx_paths, rx_impressions, rx_patient_ids,
                        output_csv):
    """
    Create unified CSV with impressions and full metadata

    Args:
        cp_paths: CheXpert-Plus image paths
        cp_impressions: CheXpert-Plus impressions
        cp_patient_ids: CheXpert-Plus patient IDs
        rx_paths: ReXGradient image paths
        rx_impressions: ReXGradient impressions
        rx_patient_ids: ReXGradient patient IDs
        output_csv: Output CSV path

    Returns:
        combined_df: Combined DataFrame
    """
    print(f"\n{'='*80}")
    print("Creating combined CSV with metadata...")
    print(f"{'='*80}")

    # Create CheXpert-Plus DataFrame
    cp_df = pd.DataFrame({
        'image_path': cp_paths,
        'patient_id': cp_patient_ids,
        'impression': cp_impressions,
        'source': 'CheXpert-Plus'
    })

    # Create ReXGradient DataFrame
    rx_df = pd.DataFrame({
        'image_path': rx_paths,
        'patient_id': rx_patient_ids,
        'impression': rx_impressions,
        'source': 'ReXGradient'
    })

    # Combine (CheXpert-Plus first, then ReXGradient)
    combined_df = pd.concat([cp_df, rx_df], ignore_index=True)

    # Add dataset index (matches HDF5 row index)
    combined_df['dataset_index'] = range(len(combined_df))

    print(f"  CheXpert-Plus: {len(cp_df)} rows")
    print(f"  ReXGradient:   {len(rx_df)} rows")
    print(f"  Combined:      {len(combined_df)} rows")

    # Save to CSV
    combined_df.to_csv(output_csv, index=False)

    print(f"  ✓ Saved to: {output_csv}")
    print(f"  ✓ Columns: {list(combined_df.columns)}")

    return combined_df


def verify_alignment(h5_path, csv_path):
    """
    Verify that HDF5 and CSV have matching row counts

    Args:
        h5_path: Path to HDF5 file
        csv_path: Path to CSV file
    """
    with h5py.File(h5_path, 'r') as h5f:
        n_images = len(h5f['cxr'])

    df = pd.read_csv(csv_path)
    n_rows = len(df)

    if n_images == n_rows:
        print(f"  ✓ Alignment verified: {n_images} images = {n_rows} CSV rows")
    else:
        print(f"  ✗ ALIGNMENT ERROR: {n_images} images ≠ {n_rows} CSV rows")
        raise ValueError("HDF5 and CSV row counts do not match!")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CheXpert-Plus + ReXGradient datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # CheXpert-Plus paths
    parser.add_argument('--cp_csv', type=str,
                        default='/cbica/projects/CXR/data/CheXpert/chexpertplus/df_chexpert_plus_240401.csv',
                        help='Path to CheXpert-Plus CSV metadata file')
    parser.add_argument('--cp_image_base', type=str,
                        default='/cbica/projects/CXR/data/CheXpert/chexpertplus/PNG/PNG',
                        help='Base path to CheXpert-Plus PNG images')

    # ReXGradient paths
    parser.add_argument('--rx_json', type=str,
                        default='/cbica/projects/CXR/data/RexGradient/data/metadata/train_metadata_view_position.json',
                        help='Path to ReXGradient JSON metadata file')
    parser.add_argument('--rx_image_base', type=str,
                        default='/cbica/projects/CXR/data/RexGradient/data/images',
                        help='Base path to ReXGradient images')

    # Output paths
    parser.add_argument('--output_dir', type=str,
                        default='metadata',
                        help='Directory to save processed files (relative to script location or absolute path)')
    parser.add_argument('--resolution', type=int, default=320,
                        help='Target image resolution (320x320 for CheXzero)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print configuration
    print("\n" + "="*80)
    print("COMBINED DATASET PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  Output directory:    {args.output_dir}")
    print(f"  Target resolution:   {args.resolution}x{args.resolution}")
    print(f"\nInput paths:")
    print(f"  CheXpert CSV:        {args.cp_csv}")
    print(f"  CheXpert Images:     {args.cp_image_base}")
    print(f"  ReXGradient JSON:    {args.rx_json}")
    print(f"  ReXGradient Images:  {args.rx_image_base}")

    # ========== STEP 1: Prepare CheXpert-Plus Training Data ==========
    cp_paths, cp_impressions, cp_patient_ids = prepare_chexpert_plus_data(
        csv_path=args.cp_csv,
        image_base_path=args.cp_image_base,
        split='train'
    )

    # Convert to HDF5
    cp_h5_path = os.path.join(args.output_dir, 'chexpert_plus_train.h5')
    print(f"\nConverting CheXpert-Plus images to HDF5...")
    # print(f"This may take 1-3 hours depending on dataset size...")
    img_to_hdf5(cp_paths, cp_h5_path, resolution=args.resolution)
    print(f"Saved: {cp_h5_path}")

    # ========== STEP 2: Prepare ReXGradient Training Data ==========
    rx_paths, rx_impressions, rx_patient_ids = prepare_rexgradient_data(
        json_path=args.rx_json,
        image_base_path=args.rx_image_base
    )

    # Convert to HDF5
    rx_h5_path = os.path.join(args.output_dir, 'rexgradient_train.h5')
    print(f"\nConverting ReXGradient images to HDF5...")
    # print(f"This may take 30min-2 hours depending on dataset size...")
    img_to_hdf5(rx_paths, rx_h5_path, resolution=args.resolution)
    print(f"Saved: {rx_h5_path}")

    # ========== STEP 3: Merge HDF5 Files ==========
    combined_h5_path = os.path.join(args.output_dir, 'combined_train.h5')
    merge_hdf5_files(cp_h5_path, rx_h5_path, combined_h5_path)

    # ========== STEP 4: Create Combined CSV with Metadata ==========
    combined_csv_path = os.path.join(args.output_dir, 'combined_train.csv')
    combined_df = create_combined_csv(
        cp_paths, cp_impressions, cp_patient_ids,
        rx_paths, rx_impressions, rx_patient_ids,
        combined_csv_path
    )

    # Verify alignment
    print(f"\nVerifying HDF5-CSV alignment...")
    verify_alignment(combined_h5_path, combined_csv_path)

    # ========== STEP 5: Process Validation Set (CheXpert-Plus Only) ==========
    print(f"\n{'='*80}")
    print("Processing CheXpert-Plus validation set...")
    print(f"{'='*80}")

    cp_val_paths, cp_val_impressions, cp_val_patient_ids = prepare_chexpert_plus_data(
        csv_path=args.cp_csv,
        image_base_path=args.cp_image_base,
        split='valid'
    )

    # Convert validation images to HDF5
    cp_val_h5_path = os.path.join(args.output_dir, 'chexpert_plus_valid.h5')
    print(f"\nConverting validation images to HDF5...")
    img_to_hdf5(cp_val_paths, cp_val_h5_path, resolution=args.resolution)
    print(f"Saved: {cp_val_h5_path}")

    # Save validation CSV with metadata
    cp_val_csv = os.path.join(args.output_dir, 'chexpert_plus_valid.csv')
    val_df = pd.DataFrame({
        'image_path': cp_val_paths,
        'patient_id': cp_val_patient_ids,
        'impression': cp_val_impressions,
        'dataset_index': range(len(cp_val_paths))
    })
    val_df.to_csv(cp_val_csv, index=False)
    print(f"Saved validation CSV to: {cp_val_csv}")

    # Verify validation alignment
    print(f"\nVerifying validation HDF5-CSV alignment...")
    verify_alignment(cp_val_h5_path, cp_val_csv)

    # ========== STEP 6: Process Test Set (CheXpert-Plus Only) - COMMENTED OUT ==========
    # Uncomment this section when you're ready to process the test set
    """
    print(f"\n{'='*80}")
    print("Processing CheXpert-Plus test set...")
    print(f"{'='*80}")

    cp_test_paths, cp_test_impressions, cp_test_patient_ids = prepare_chexpert_plus_data(
        csv_path=args.cp_csv,
        image_base_path=args.cp_image_base,
        split='test'
    )

    # Convert test images to HDF5
    cp_test_h5_path = os.path.join(args.output_dir, 'chexpert_plus_test.h5')
    print(f"\nConverting test images to HDF5...")
    img_to_hdf5(cp_test_paths, cp_test_h5_path, resolution=args.resolution)
    print(f"Saved: {cp_test_h5_path}")

    # Save test CSV with metadata
    cp_test_csv = os.path.join(args.output_dir, 'chexpert_plus_test.csv')
    test_df = pd.DataFrame({
        'image_path': cp_test_paths,
        'patient_id': cp_test_patient_ids,
        'impression': cp_test_impressions,
        'dataset_index': range(len(cp_test_paths))
    })
    test_df.to_csv(cp_test_csv, index=False)
    print(f"Saved test CSV to: {cp_test_csv}")

    # Verify test alignment
    print(f"\nVerifying test HDF5-CSV alignment...")
    verify_alignment(cp_test_h5_path, cp_test_csv)
    """

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)

    print("\n Generated Files:")
    print(f"\n  Training:")
    print(f"    Images:  {combined_h5_path}")
    print(f"    Metadata: {combined_csv_path}")

    print(f"\n  Validation:")
    print(f"    Images:  {cp_val_h5_path}")
    print(f"    Metadata: {cp_val_csv}")

    # print(f"\n  Test (when uncommented):")
    # print(f"    Images:  {os.path.join(args.output_dir, 'chexpert_plus_test.h5')}")
    # print(f"    Metadata: {os.path.join(args.output_dir, 'chexpert_plus_test.csv')}")

    print(f"\n Dataset Statistics:")
    print(f"    Training Set:   {len(combined_df)} samples")
    print(f"      - CheXpert-Plus: {len(cp_paths)} images")
    print(f"      - ReXGradient:   {len(rx_paths)} images")
    print(f"    Validation Set: {len(val_df)} samples")
    print(f"      - CheXpert-Plus: {len(cp_val_paths)} images")

    print(f"\n CSV Structure:")
    print(f"    Columns: {list(combined_df.columns)}")
    print(f"    - image_path:    Full path to original image")
    print(f"    - patient_id:    Patient identifier")
    print(f"    - impression:    Text report (used for training)")
    print(f"    - source:        Dataset name")
    print(f"    - dataset_index: Row index matching HDF5")


if __name__ == "__main__":
    main()
