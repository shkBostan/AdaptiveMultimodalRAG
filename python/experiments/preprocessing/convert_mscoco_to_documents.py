"""
Convert MS COCO annotations to pipeline-ready Document JSONL format.

This script:
1. Loads COCO annotation files (train/val splits)
2. Matches captions with corresponding image files
3. Creates Document objects with proper metadata
4. Saves documents to JSONL format for pipeline consumption
5. Generates statistics and validation reports

Author: s Bostan
Created on: Jan, 2026
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.preprocessing import preprocess_text


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


PREPROCESSING_VERSION = "1.0.0"


def load_coco_annotations(annotation_file: Path) -> Dict[str, Any]:
    """
    Load COCO annotation JSON file.
    
    Args:
        annotation_file: Path to COCO annotation JSON file
        
    Returns:
        Dictionary containing COCO annotation data
        
    Raises:
        FileNotFoundError: If annotation file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
    
    logger.info(f"Loading annotations from: {annotation_file}")
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data.get('annotations', []))} annotations")
    return data


def build_image_index(coco_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Build index mapping image_id to image information.
    
    Args:
        coco_data: COCO annotation data dictionary
        
    Returns:
        Dictionary mapping image_id to image info
    """
    image_index = {}
    for img in coco_data.get('images', []):
        image_index[img['id']] = img
    logger.info(f"Indexed {len(image_index)} images")
    return image_index


def create_document_id(split: str, image_id: int, caption_id: int) -> str:
    """
    Create deterministic document ID.
    
    Format: mscoco_{split}_{image_id:012d}_{caption_id}
    
    Args:
        split: Dataset split (train/val)
        image_id: COCO image ID
        caption_id: COCO caption ID
        
    Returns:
        Unique document identifier
    """
    return f"mscoco_{split}_{image_id:012d}_{caption_id}"


def validate_image_path(
    image_path: Path | str,
    doc_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate image path existence and readability.

    Returns structured validation report.
    """
    image_path = Path(image_path)

    result = {
        "valid": False,
        "exists": False,
        "readable": False,
        "doc_id": doc_id,
        "path": str(image_path)
    }

    if not image_path.exists():
        return result

    result["exists"] = True

    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.verify()
        result["readable"] = True
        result["valid"] = True
    except Exception:
        result["readable"] = False

    return result



def create_document(
    split: str,
    image_id: int,
    caption_id: int,
    caption: str,
    image_info: Dict[str, Any],
    image_dir: Path,
    coco_file: str,
    normalize_text: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Create a Document object from COCO annotation.
    
    Args:
        split: Dataset split (train/val)
        image_id: COCO image ID
        caption_id: COCO caption ID
        caption: Caption text
        image_info: Image information dictionary from COCO
        image_dir: Directory containing images
        coco_file: Source COCO annotation filename
        normalize_text: Whether to normalize text content
        
    Returns:
        Document dictionary or None if image is missing/invalid
    """
    # Validate caption text
    if not caption or not caption.strip():
        logger.warning(f"Skipping empty caption: caption_id={caption_id}")
        return None
    
    # Normalize text if requested
    content = preprocess_text(caption, lower=False) if normalize_text else caption.strip()
    if not content:
        logger.warning(f"Skipping caption after normalization: caption_id={caption_id}")
        return None
    
    # Build image path
    file_name = image_info.get('file_name', '')
    if not file_name:
        logger.warning(f"Skipping image without filename: image_id={image_id}")
        return None
    
    image_path = image_dir / file_name
    
    validation = validate_image_path(image_path, doc_id=None)

    if not validation["valid"]:
        logger.warning(f"Image not found or invalid: {image_path}")
        return None

    
    # Create document ID
    doc_id = create_document_id(split, image_id, caption_id)
    
    # Get absolute path for image
    abs_image_path = image_path.resolve()
    
    # Create document
    document = {
        "id": doc_id,
        "content": content,
        "image_path": str(abs_image_path),
        "metadata": {
            "split": split,
            "image_id": image_id,
            "caption_id": caption_id,
            "coco_file": coco_file,
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessed_at": datetime.utcnow().isoformat() + "Z",
            "file_name": file_name,
            "image_width": image_info.get('width'),
            "image_height": image_info.get('height')
        }
    }
    
    return document


def process_split(
    annotation_file: Path,
    image_dir: Path,
    split: str,
    normalize_text: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process a single COCO split (train or val).
    
    Args:
        annotation_file: Path to COCO annotation file
        image_dir: Directory containing images for this split
        split: Split name (train/val)
        normalize_text: Whether to normalize text content
        
    Returns:
        Tuple of (documents list, statistics dictionary)
    """
    # Load annotations
    coco_data = load_coco_annotations(annotation_file)
    
    # Build image index
    image_index = build_image_index(coco_data)
    
    # Statistics
    stats = {
        "total_annotations": len(coco_data.get('annotations', [])),
        "total_images": len(image_index),
        "documents_created": 0,
        "missing_images": 0,
        "empty_captions": 0,
        "skipped": 0
    }
    
    documents = []
    coco_file = annotation_file.name
    
    # Process each annotation
    for ann in coco_data.get('annotations', []):
        image_id = ann.get('image_id')
        caption_id = ann.get('id')
        caption = ann.get('caption', '').strip()
        
        # Get image info
        image_info = image_index.get(image_id)
        if not image_info:
            stats["missing_images"] += 1
            logger.warning(f"Image info not found for image_id={image_id}")
            continue
        
        # Validate caption
        if not caption:
            stats["empty_captions"] += 1
            continue
        
        # Create document
        document = create_document(
            split=split,
            image_id=image_id,
            caption_id=caption_id,
            caption=caption,
            image_info=image_info,
            image_dir=image_dir,
            coco_file=coco_file,
            normalize_text=normalize_text
        )
        
        if document:
            documents.append(document)
            stats["documents_created"] += 1
        else:
            stats["skipped"] += 1
    
    logger.info(
        f"Split '{split}' processed: {stats['documents_created']} documents created, "
        f"{stats['skipped']} skipped"
    )
    
    return documents, stats


def save_documents_jsonl(documents: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Save documents to JSONL file.
    
    Args:
        documents: List of document dictionaries
        output_file: Path to output JSONL file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(documents)} documents to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    logger.info(f"Documents saved successfully to: {output_file}")


def generate_statistics_report(
    all_stats: Dict[str, Dict[str, int]],
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate and optionally save statistics report.
    
    Args:
        all_stats: Dictionary mapping split names to statistics
        output_file: Optional path to save report JSON
        
    Returns:
        Statistics report dictionary
    """
    report = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "splits": all_stats,
        "summary": {
            "total_documents": sum(s["documents_created"] for s in all_stats.values()),
            "total_annotations": sum(s["total_annotations"] for s in all_stats.values()),
            "total_images": sum(s["total_images"] for s in all_stats.values()),
            "total_missing_images": sum(s["missing_images"] for s in all_stats.values()),
            "total_empty_captions": sum(s["empty_captions"] for s in all_stats.values()),
            "total_skipped": sum(s["skipped"] for s in all_stats.values())
        }
    }
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics report saved to: {output_file}")
    
    return report


def main(
    mscoco_dir: Path,
    output_file: Optional[Path] = None,
    train_split: bool = True,
    val_split: bool = True,
    normalize_text: bool = True,
    stats_file: Optional[Path] = None
) -> None:
    """
    Main function to convert MS COCO annotations to documents.
    
    Args:
        mscoco_dir: Path to MS COCO dataset directory (contains processed/)
        output_file: Path to output JSONL file (default: processed/pipeline/documents.jsonl)
        train_split: Whether to process train split
        val_split: Whether to process val split
        normalize_text: Whether to normalize text content
        stats_file: Optional path to save statistics report
    """
    mscoco_dir = Path(mscoco_dir)
    
    # Default paths
    processed_dir = mscoco_dir / "processed"
    annotations_dir = processed_dir / "annotations"
    images_dir = processed_dir / "images"
    
    if output_file is None:
        output_file = processed_dir / "pipeline" / "documents.jsonl"
    
    logger.info("=" * 70)
    logger.info("MS COCO to Documents Converter")
    logger.info("=" * 70)
    logger.info(f"MS COCO directory: {mscoco_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Process train: {train_split}, Process val: {val_split}")
    logger.info(f"Normalize text: {normalize_text}")
    
    all_documents = []
    all_stats = {}
    
    # Process train split
    if train_split:
        train_annotation_file = annotations_dir / "captions_train.json"
        train_image_dir = images_dir / "train"
        
        if train_annotation_file.exists() and train_image_dir.exists():
            train_docs, train_stats = process_split(
                annotation_file=train_annotation_file,
                image_dir=train_image_dir,
                split="train",
                normalize_text=normalize_text
            )
            all_documents.extend(train_docs)
            all_stats["train"] = train_stats
        else:
            logger.warning(f"Train split files not found, skipping")
    
    # Process val split
    if val_split:
        val_annotation_file = annotations_dir / "captions_val.json"
        val_image_dir = images_dir / "train"  # Note: val images might be in train directory
        
        # Try val directory first, fallback to train
        if not (images_dir / "val").exists():
            val_image_dir = images_dir / "train"
        else:
            val_image_dir = images_dir / "val"
        
        if val_annotation_file.exists() and val_image_dir.exists():
            val_docs, val_stats = process_split(
                annotation_file=val_annotation_file,
                image_dir=val_image_dir,
                split="val",
                normalize_text=normalize_text
            )
            all_documents.extend(val_docs)
            all_stats["val"] = val_stats
        else:
            logger.warning(f"Val split files not found, skipping")
    
    # Save documents
    if all_documents:
        save_documents_jsonl(all_documents, output_file)
        
        # Generate statistics
        report = generate_statistics_report(all_stats, stats_file)
        
        logger.info("=" * 70)
        logger.info("Conversion Complete")
        logger.info("=" * 70)
        logger.info(f"Total documents created: {report['summary']['total_documents']}")
        logger.info(f"Total annotations processed: {report['summary']['total_annotations']}")
        logger.info(f"Documents saved to: {output_file}")
    else:
        logger.error("No documents were created. Please check input files and paths.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MS COCO annotations to pipeline-ready Document JSONL format"
    )
    parser.add_argument(
        "--mscoco-dir",
        type=str,
        default="experiments/datasets/mscoco",
        help="Path to MS COCO dataset directory (default: experiments/datasets/mscoco)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSONL file (default: processed/pipeline/documents.jsonl)"
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip train split processing"
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Skip val split processing"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable text normalization"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Path to save statistics report JSON (optional)"
    )
    
    args = parser.parse_args()
    
    main(
        mscoco_dir=Path(args.mscoco_dir),
        output_file=Path(args.output) if args.output else None,
        train_split=not args.no_train,
        val_split=not args.no_val,
        normalize_text=not args.no_normalize,
        stats_file=Path(args.stats) if args.stats else None
    )

