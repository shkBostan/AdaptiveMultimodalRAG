"""
Generate deterministic train/val/test splits for MS COCO documents.

This script:
1. Loads documents from documents.jsonl
2. Creates deterministic splits ensuring no data leakage
3. Ensures images appear in only one split
4. Saves split mapping as JSON

Author: s Bostan
Created on: Jan, 2026
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)


def load_documents(documents_file: Path) -> List[Dict[str, Any]]:
    """
    Load documents from JSONL file.
    
    Args:
        documents_file: Path to documents.jsonl file
        
    Returns:
        List of document dictionaries
        
    Raises:
        FileNotFoundError: If documents file doesn't exist
    """
    if not documents_file.exists():
        raise FileNotFoundError(f"Documents file not found: {documents_file}")
    
    logger.info(f"Loading documents from: {documents_file}")
    documents = []
    
    with open(documents_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                documents.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def group_documents_by_image(documents: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """
    Group document IDs by image_id to prevent data leakage.
    
    All documents with the same image_id must be in the same split.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Dictionary mapping image_id to list of document IDs
    """
    image_to_docs = {}
    
    for doc in documents:
        image_id = doc.get('metadata', {}).get('image_id')
        if image_id is None:
            logger.warning(f"Document {doc.get('id')} has no image_id, skipping")
            continue
        
        if image_id not in image_to_docs:
            image_to_docs[image_id] = []
        image_to_docs[image_id].append(doc['id'])
    
    logger.info(f"Grouped {len(documents)} documents into {len(image_to_docs)} unique images")
    return image_to_docs


def create_deterministic_splits(
    image_to_docs: Dict[int, List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create deterministic train/val/test splits.
    
    Splits are created at the image level to prevent data leakage.
    All documents with the same image_id are assigned to the same split.
    
    Args:
        image_to_docs: Dictionary mapping image_id to document IDs
        train_ratio: Ratio for train split (default: 0.8)
        val_ratio: Ratio for val split (default: 0.1)
        test_ratio: Ratio for test split (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to lists of document IDs
        
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio}: "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )
    
    set_seed(seed)
    import random
    
    # Get sorted list of image IDs for deterministic ordering
    image_ids = sorted(image_to_docs.keys())
    random.shuffle(image_ids)  # Shuffle with fixed seed
    
    # Calculate split sizes
    n_images = len(image_ids)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    # n_test = n_images - n_train - n_val  # Remaining images go to test
    
    # Assign images to splits
    train_images = image_ids[:n_train]
    val_images = image_ids[n_train:n_train + n_val]
    test_images = image_ids[n_train + n_val:]
    
    # Collect document IDs for each split
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    
    for image_id in train_images:
        splits["train"].extend(image_to_docs[image_id])
    
    for image_id in val_images:
        splits["val"].extend(image_to_docs[image_id])
    
    for image_id in test_images:
        splits["test"].extend(image_to_docs[image_id])
    
    # Sort document IDs within each split for reproducibility
    splits["train"] = sorted(splits["train"])
    splits["val"] = sorted(splits["val"])
    splits["test"] = sorted(splits["test"])
    
    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(splits['train'])} documents ({len(train_images)} images)")
    logger.info(f"  Val: {len(splits['val'])} documents ({len(val_images)} images)")
    logger.info(f"  Test: {len(splits['test'])} documents ({len(test_images)} images)")
    
    return splits


def validate_splits(
    splits: Dict[str, List[str]],
    image_to_docs: Dict[int, List[str]]
) -> Dict[str, Any]:
    """
    Validate that splits have no data leakage.
    
    Checks that all documents with the same image_id are in the same split.
    
    Args:
        splits: Dictionary mapping split names to document ID lists
        image_to_docs: Dictionary mapping image_id to document IDs
        
    Returns:
        Validation report dictionary
    """
    # Build reverse index: doc_id -> image_id
    doc_to_image = {}
    for image_id, doc_ids in image_to_docs.items():
        for doc_id in doc_ids:
            doc_to_image[doc_id] = image_id
    
    # Check for leakage
    leakage = []
    split_by_doc = {}
    
    for split_name, doc_ids in splits.items():
        for doc_id in doc_ids:
            split_by_doc[doc_id] = split_name
    
    # Check each image's documents are in the same split
    for image_id, doc_ids in image_to_docs.items():
        split_names = set(split_by_doc.get(doc_id, 'unknown') for doc_id in doc_ids)
        if len(split_names) > 1:
            leakage.append({
                "image_id": image_id,
                "document_ids": doc_ids,
                "splits": list(split_names)
            })
    
    # Check for duplicate documents across splits
    all_docs = []
    for doc_ids in splits.values():
        all_docs.extend(doc_ids)
    
    duplicates = []
    seen = set()
    for doc_id in all_docs:
        if doc_id in seen:
            duplicates.append(doc_id)
        seen.add(doc_id)
    
    validation_report = {
        "valid": len(leakage) == 0 and len(duplicates) == 0,
        "leakage_detected": len(leakage) > 0,
        "duplicates_detected": len(duplicates) > 0,
        "num_leakage_cases": len(leakage),
        "num_duplicates": len(duplicates),
        "leakage_details": leakage[:10] if leakage else [],  # First 10 for reporting
        "duplicate_details": duplicates[:10] if duplicates else []
    }
    
    return validation_report


def save_splits(splits: Dict[str, List[str]], output_file: Path) -> None:
    """
    Save splits to JSON file.
    
    Args:
        splits: Dictionary mapping split names to document ID lists
        output_file: Path to output JSON file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "splits": splits,
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "num_splits": len(splits),
            "split_sizes": {name: len(doc_ids) for name, doc_ids in splits.items()}
        }
    }
    
    logger.info(f"Saving splits to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Splits saved successfully to: {output_file}")


def main(
    documents_file: Path,
    output_file: Optional[Path] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    validate: bool = True
) -> None:
    """
    Main function to generate splits.
    
    Args:
        documents_file: Path to documents.jsonl file
        output_file: Path to output splits JSON file
        train_ratio: Ratio for train split (default: 0.8)
        val_ratio: Ratio for val split (default: 0.1)
        test_ratio: Ratio for test split (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
        validate: Whether to validate splits for data leakage (default: True)
    """
    logger.info("=" * 70)
    logger.info("MS COCO Split Generator")
    logger.info("=" * 70)
    logger.info(f"Documents file: {documents_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    logger.info(f"Random seed: {seed}")
    
    # Load documents
    documents = load_documents(documents_file)
    
    if not documents:
        logger.error("No documents loaded. Cannot generate splits.")
        sys.exit(1)
    
    # Group documents by image_id
    image_to_docs = group_documents_by_image(documents)
    
    if not image_to_docs:
        logger.error("No valid images found. Cannot generate splits.")
        sys.exit(1)
    
    # Create splits
    splits = create_deterministic_splits(
        image_to_docs=image_to_docs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Validate splits
    if validate:
        logger.info("Validating splits for data leakage...")
        validation_report = validate_splits(splits, image_to_docs)
        
        if validation_report["valid"]:
            logger.info("✓ Splits validated: No data leakage detected")
        else:
            logger.error("✗ Splits validation failed:")
            if validation_report["leakage_detected"]:
                logger.error(f"  Data leakage detected: {validation_report['num_leakage_cases']} cases")
                for leakage in validation_report["leakage_details"][:5]:
                    logger.error(f"    Image {leakage['image_id']} appears in splits: {leakage['splits']}")
            if validation_report["duplicates_detected"]:
                logger.error(f"  Duplicate documents detected: {validation_report['num_duplicates']} duplicates")
                logger.error(f"    Examples: {validation_report['duplicate_details'][:5]}")
            sys.exit(1)
    
    # Save splits
    if output_file is None:
        output_file = documents_file.parent / "splits.json"
    
    save_splits(splits, output_file)
    
    logger.info("=" * 70)
    logger.info("Split Generation Complete")
    logger.info("=" * 70)
    logger.info(f"Splits saved to: {output_file}")
    total_docs = sum(len(doc_ids) for doc_ids in splits.values())
    logger.info(f"Total documents in splits: {total_docs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate deterministic train/val/test splits for MS COCO documents"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="experiments/datasets/mscoco/processed/pipeline/documents.jsonl",
        help="Path to documents.jsonl file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output splits.json file (default: same dir as documents)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Val split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation for data leakage"
    )
    
    args = parser.parse_args()
    
    main(
        documents_file=Path(args.documents),
        output_file=Path(args.output) if args.output else None,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        validate=not args.no_validate
    )

