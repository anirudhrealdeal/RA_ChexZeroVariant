#!/usr/bin/env python3
"""
Extract validation labels from report_fixed.json (JSONL format)
Converts to CSV with proper paths and handles null/-1 values
"""
import json
import pandas as pd
import sys
import os

# 14 CheXpert pathology labels (order matters for evaluation)
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

def convert_label_value(value):
    """
    Convert CheXbert label values to binary (U-Positive strategy):
    - 1.0 → 1 (present)
    - -1.0 → 1 (uncertain, treated as POSITIVE per CheXzero/CheXpert competition)
    - 0.0 → 0 (absent)
    - null → 0 (not mentioned, treated as absent)

    Note: This follows the CheXpert competition U-Positive policy where
    uncertain labels are treated as positive for evaluation.
    """
    if value == 1.0 or value == -1.0:  # U-Positive: uncertain = positive
        return 1
    else:
        return 0

def extract_validation_labels(
    input_jsonl_path,
    output_csv_path,
    base_data_dir='/cbica/projects/CXR/data/CheXpert/chexpertplus/PNG/PNG'
):
    """
    Extract validation labels from JSONL file and save as CSV

    Args:
        input_jsonl_path: Path to report_fixed.json (JSONL format)
        output_csv_path: Where to save the CSV with labels
        base_data_dir: Base directory for image paths
    """
    validation_data = []

    print(f"Reading {input_jsonl_path}...")

    with open(input_jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  Processed {line_num} lines...")

            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line {line_num}")
                continue

            path = entry.get('path_to_image', '')

            # Filter for validation set only
            if 'valid/' not in path and '/valid/' not in path:
                continue

            # Fix path: .jpg → .png and add base directory
            path_fixed = path.replace('.jpg', '.png')
            full_path = os.path.join(base_data_dir, path_fixed)

            # Extract 14 labels and convert to binary
            labels = {
                label: convert_label_value(entry.get(label))
                for label in CHEXPERT_LABELS
            }

            # Create row
            row = {'image_path': full_path}
            row.update(labels)
            validation_data.append(row)

    print(f"\nFound {len(validation_data)} validation images")

    if len(validation_data) == 0:
        print("ERROR: No validation images found!")
        print("Check if paths contain 'valid' in the JSONL file")
        sys.exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(validation_data)

    # Reorder columns: image_path first, then 14 labels in order
    columns = ['image_path'] + CHEXPERT_LABELS
    df = df[columns]

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved validation labels to: {output_csv_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Print label distribution
    print("\nLabel distribution (count of positive cases):")
    for label in CHEXPERT_LABELS:
        pos_count = df[label].sum()
        print(f"  {label:30s}: {pos_count:4d} / {len(df)} ({100*pos_count/len(df):.1f}%)")

    return df

if __name__ == '__main__':
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_jsonl = os.path.join(project_root, 'report_fixed.json')
    output_csv = os.path.join(project_root, 'metadata', 'chexpert_plus_valid_labels.csv')

    print("="*60)
    print("CheXpert Validation Label Extraction")
    print("="*60)
    print(f"Input:  {input_jsonl}")
    print(f"Output: {output_csv}")
    print()

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Extract labels
    df = extract_validation_labels(input_jsonl, output_csv)

    print("\n" + "="*60)
    print("✓ Done!")
    print("="*60)
