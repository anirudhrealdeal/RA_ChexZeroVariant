#!/usr/bin/env python3
"""
Combined Dataset Preprocessing for CheXpert-Plus + ReXGradient
Converts images to HDF5 and creates unified CSV with impressions and metadata

Usage:
    python preprocess.py

Output:
    metadata/
        ├── combined_train.h5            # Training images (CheXpert-Plus + ReXGradient)
        ├── combined_train.csv           # Training metadata with impressions
        ├── chexpert_plus_valid.h5       # Validation images (CheXpert-Plus only)
        └── chexpert_plus_valid.csv      # Validation metadata with impressions
"""

import os
import json
import argparse
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

# Import existing preprocessing functions
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_process import img_to_hdf5


def prepare_chexpert_plus_data(csv_path, image_base_path, split='train'):
    """
    Extract image paths and impressions from CheXpert-Plus CSV

    Args:
        csv_path: Path to CheXpert-Plus metadata CSV
        image_base_path: Base directory containing PNG images
        split: 'train', 'valid', or 'test'

    Returns:
        tuple: (image_paths, impressions, patient_ids)
    """
    print(f"\n{'='*80}")
    print(f"Processing CheXpert-Plus {split} set...")
    print(f"{'='*80}")

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} total rows")

    # Filter by split
    df_filtered = df[df['split'] == split].copy()
    print(f"Filtered to {len(df_filtered)} rows for split='{split}'")

    # Extract paths, impressions, and patient IDs
    image_paths = []
    impressions = []
    patient_ids = []
    failed_count = 0

    for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Extracting data"):
        # Image path (relative to base)
        rel_path = row['path_to_image']
        # Fix file extension: CSV lists .jpg but actual files are .png
        rel_path = rel_path.replace('.jpg', '.png')
        abs_path = os.path.join(image_base_path, rel_path)

        # Check if file exists
        if not os.path.exists(abs_path):
            failed_count += 1
            if failed_count <= 3:  # Show first 3 failures
                print(f"  Warning: File not found: {abs_path}")
            continue

        # Extract data
        image_paths.append(abs_path)
        impressions.append(str(row['impression']) if pd.notna(row['impression']) else '')
        patient_ids.append(str(row['patient_id']) if pd.notna(row['patient_id']) else '')

    if failed_count > 0:
        print(f"  Warning: {failed_count} images not found (skipped)")

    print(f"Successfully extracted {len(image_paths)} images with impressions")

    return image_paths, impressions, patient_ids


def prepare_rexgradient_data(json_path, image_base_path):
    """
    Extract image paths and impressions from ReXGradient JSON
    Expands multi-view studies into individual images

    Args:
        json_path: Path to ReXGradient metadata JSON
        image_base_path: Base directory containing images

    Returns:
        tuple: (image_paths, impressions, patient_ids)
    """
    print(f"\n{'='*80}")
    print(f"Processing ReXGradient dataset...")
    print(f"{'='*80}")

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} studies from JSON")

    # Expand each study into individual images
    image_paths = []
    impressions = []
    patient_ids = []

    for study_id, study_data in tqdm(data.items(), desc="Expanding images"):
        # Get impression for this study
        impression = study_data.get('impression', '')
        patient_id = study_data.get('subject_id', study_id)

        # Get all image paths for this study
        study_images = study_data.get('image_path', [])

        # Add each image with the same impression
        for img_rel_path in study_images:
            img_abs_path = os.path.join(image_base_path, img_rel_path)

            if os.path.exists(img_abs_path):
                image_paths.append(img_abs_path)
                impressions.append(impression)
                patient_ids.append(str(patient_id))

    print(f"Expanded from {len(data)} studies to {len(image_paths)} images")
    print(f"Each study impression replicated for all its images")

    return image_paths, impressions, patient_ids


def merge_hdf5_files(file1, file2, output_file):
    """
    Merge two HDF5 files into one combined file using chunked copying
    to avoid loading entire datasets into memory

    Args:
        file1: Path to first HDF5 file (CheXpert-Plus)
        file2: Path to second HDF5 file (ReXGradient)
        output_file: Path to output merged HDF5 file
    """
    print(f"\n{'='*80}")
    print("Merging HDF5 files (chunked, memory-efficient)...")
    print(f"{'='*80}")

    with h5py.File(file1, 'r') as f1, \
         h5py.File(file2, 'r') as f2, \
         h5py.File(output_file, 'w') as out:

        # Get shapes without loading data
        shape1 = f1['cxr'].shape
        shape2 = f2['cxr'].shape

        print(f"  CheXpert-Plus: {len(shape1)} dims, shape {shape1}")
        print(f"  ReXGradient:   {len(shape2)} dims, shape {shape2}")

        # Verify shapes match (except first dimension)
        if shape1[1:] != shape2[1:]:
            raise ValueError(f"Shape mismatch: {shape1} vs {shape2}")

        # Create output dataset with combined shape
        n1 = shape1[0]
        n2 = shape2[0]
        total_n = n1 + n2
        combined_shape = (total_n,) + shape1[1:]
        print(f"  Combined:      {total_n} images, shape {combined_shape}")

        # Create dataset with compression and chunking for efficient I/O
        chunk_size = (min(1000, total_n),) + shape1[1:]
        out.create_dataset('cxr', shape=combined_shape, dtype=f1['cxr'].dtype,
                          compression='gzip', compression_opts=4,
                          chunks=chunk_size)

        # Copy data1 in chunks (memory-efficient)
        print(f"  Copying CheXpert-Plus data...")
        chunk_batch = 1000
        for i in tqdm(range(0, n1, chunk_batch), desc="  CheXpert chunks"):
            end_idx = min(i + chunk_batch, n1)
            out['cxr'][i:end_idx] = f1['cxr'][i:end_idx]

        # Copy data2 in chunks (memory-efficient)
        print(f"  Copying ReXGradient data...")
        for i in tqdm(range(0, n2, chunk_batch), desc="  ReXGradient chunks"):
            end_idx = min(i + chunk_batch, n2)
            out['cxr'][n1 + i:n1 + end_idx] = f2['cxr'][i:end_idx]

        # Store metadata as attributes
        out.attrs['n_chexpert_plus'] = n1
        out.attrs['n_rexgradient'] = n2
        out.attrs['total'] = total_n
        out.attrs['resolution'] = shape1[1]  # Height dimension

        print(f"  ✓ Merged: {n1} + {n2} = {total_n} images")
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
        output_csv: Path to save combined CSV

    Returns:
        pd.DataFrame: Combined metadata DataFrame
    """
    print(f"\n{'='*80}")
    print("Creating combined CSV with metadata...")
    print(f"{'='*80}")

    # Create CheXpert-Plus DataFrame
    cp_df = pd.DataFrame({
        'image_path': cp_paths,
        'patient_id': cp_patient_ids,
        'impression': cp_impressions,
        'source': 'chexpert_plus',
        'dataset_index': range(len(cp_paths))
    })

    # Create ReXGradient DataFrame
    rx_df = pd.DataFrame({
        'image_path': rx_paths,
        'patient_id': rx_patient_ids,
        'impression': rx_impressions,
        'source': 'rexgradient',
        'dataset_index': range(len(cp_paths), len(cp_paths) + len(rx_paths))
    })

    # Combine DataFrames
    combined_df = pd.concat([cp_df, rx_df], ignore_index=True)

    # Save to CSV
    combined_df.to_csv(output_csv, index=False)

    print(f"  ✓ Saved combined CSV: {output_csv}")
    print(f"    Total rows: {len(combined_df)}")
    print(f"    - CheXpert-Plus: {len(cp_df)} rows")
    print(f"    - ReXGradient:   {len(rx_df)} rows")

    return combined_df


def verify_alignment(h5_path, csv_path):
    """
    Verify that HDF5 and CSV are properly aligned

    Args:
        h5_path: Path to HDF5 file
        csv_path: Path to CSV file
    """
    with h5py.File(h5_path, 'r') as f:
        h5_count = len(f['cxr'])

    df = pd.read_csv(csv_path)
    csv_count = len(df)

    if h5_count == csv_count:
        print(f"  ✓ Alignment verified: {h5_count} images in HDF5 match {csv_count} rows in CSV")
    else:
        print(f"  ✗ ALIGNMENT ERROR: {h5_count} images in HDF5 vs {csv_count} rows in CSV")
        raise ValueError("HDF5-CSV alignment mismatch!")


def main():
    parser = argparse.ArgumentParser(
        description='Combined preprocessing for CheXpert-Plus + ReXGradient datasets',
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
                        help='Directory to save processed files')
    parser.add_argument('--resolution', type=int, default=320,
                        help='Target image resolution (320x320)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== STEP 1: Process CheXpert-Plus Training Set ==========
    cp_paths, cp_impressions, cp_patient_ids = prepare_chexpert_plus_data(
        csv_path=args.cp_csv,
        image_base_path=args.cp_image_base,
        split='train'
    )

    # Convert to HDF5
    cp_h5_path = os.path.join(args.output_dir, 'chexpert_plus_train.h5')
    print(f"\nConverting CheXpert-Plus images to HDF5...")
    img_to_hdf5(cp_paths, cp_h5_path, resolution=args.resolution)
    print(f"✓ Saved: {cp_h5_path}")

    # ========== STEP 2: Process ReXGradient Training Set ==========
    rx_paths, rx_impressions, rx_patient_ids = prepare_rexgradient_data(
        json_path=args.rx_json,
        image_base_path=args.rx_image_base
    )

    # Convert to HDF5
    rx_h5_path = os.path.join(args.output_dir, 'rexgradient_train.h5')
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

    print("\n Generated Files:")
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
    print(f"    - image_path:    Full path to original image")
    print(f"    - patient_id:    Patient identifier")
    print(f"    - impression:    Text report (used for training)")
    print(f"    - source:        Dataset name")
    print(f"    - dataset_index: Row index matching HDF5")


if __name__ == "__main__":
    main()
