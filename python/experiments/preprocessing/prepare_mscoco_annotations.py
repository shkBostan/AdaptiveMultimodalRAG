"""
Prepare MSCoco processed annotations for tests and pipeline.

This script:
1. Reads raw COCO annotation JSONs
2. Copies and renames them for processed folder
3. Keeps train and val separate for compatibility with fixtures/tests

Author: s Bostan
Created on: Jan, 2026
"""

import json
from pathlib import Path
import shutil

# Paths
raw_dir = Path("experiments/datasets/mscoco/raw/annotations")
processed_dir = Path("experiments/datasets/mscoco/processed/annotations")
processed_dir.mkdir(parents=True, exist_ok=True)

# Mapping raw filenames to processed filenames
file_map = {
    "captions_train2017.json": "captions_train.json",
    "captions_val2017.json": "captions_val.json"
}

# Copy and rename files
for raw_name, processed_name in file_map.items():
    raw_file = raw_dir / raw_name
    processed_file = processed_dir / processed_name

    if not raw_file.exists():
        raise FileNotFoundError(f"Raw annotation file not found: {raw_file}")

    # Optional: read and re-save to ensure valid JSON formatting
    with open(raw_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Processed annotation saved: {processed_file}")

print("All MSCoco processed annotations are ready for tests.")
