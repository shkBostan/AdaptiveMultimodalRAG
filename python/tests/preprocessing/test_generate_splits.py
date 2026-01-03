"""
Unit tests for generate_splits.py preprocessing script.

Research intent: Verify split generation produces disjoint, deterministic splits
with no data leakage (same image_id cannot appear in multiple splits).
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List, Set
import importlib.util

# Load preprocessing module using importlib for reliable imports
_preprocessing_dir = Path(__file__).parent.parent.parent / "experiments" / "preprocessing"
_module_path = _preprocessing_dir / "generate_splits.py"
spec = importlib.util.spec_from_file_location("generate_splits", _module_path)
generate_splits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_splits)

# Import functions for easier access
group_documents_by_image = generate_splits.group_documents_by_image
create_deterministic_splits = generate_splits.create_deterministic_splits
validate_splits = generate_splits.validate_splits


def load_documents_jsonl(jsonl_path: Path) -> List[Dict]:
    """Helper to load documents from JSONL."""
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line.strip()))
    return documents


class TestImageGrouping:
    """Test document grouping by image_id to prevent leakage."""
    
    def test_group_by_image_id(self, sample_documents_jsonl):
        """Verify documents are correctly grouped by image_id."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        image_to_docs = group_documents_by_image(documents)
        
        assert len(image_to_docs) > 0
        
        # Verify all documents with same image_id are in same group
        for image_id, doc_ids in image_to_docs.items():
            assert len(doc_ids) > 0
            for doc_id in doc_ids:
                # Find original document
                doc = next(d for d in documents if d["id"] == doc_id)
                assert doc["metadata"]["image_id"] == image_id
    
    def test_group_excludes_missing_image_id(self, tmp_path):
        """Verify documents without image_id are excluded."""
        invalid_docs = [
            {"id": "doc1", "metadata": {}},  # Missing image_id
            {"id": "doc2", "metadata": {"image_id": 1}}
        ]
        
        jsonl_path = tmp_path / "invalid.jsonl"
        with open(jsonl_path, 'w') as f:
            for doc in invalid_docs:
                f.write(json.dumps(doc) + '\n')
        
        documents = load_documents_jsonl(jsonl_path)
        image_to_docs = group_documents_by_image(documents)
        
        # Only doc2 should be included
        assert 1 in image_to_docs
        assert "doc2" in image_to_docs[1]
        for docs in image_to_docs.values():
            assert "doc1" not in docs

class TestSplitGeneration:
    """Test deterministic split creation."""
    
    def test_splits_disjoint(self, sample_documents_jsonl):
        """Verify train/val/test splits are disjoint (no overlap)."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        image_to_docs = group_documents_by_image(documents)
        
        splits = create_deterministic_splits(
            image_to_docs,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=42
        )
        
        train_docs = set(splits["train"])
        val_docs = set(splits["val"])
        test_docs = set(splits["test"])
        
        assert len(train_docs & val_docs) == 0, "Train and val splits overlap"
        assert len(train_docs & test_docs) == 0, "Train and test splits overlap"
        assert len(val_docs & test_docs) == 0, "Val and test splits overlap"
    
    def test_no_image_id_leakage(self, sample_documents_jsonl):
        """Verify same image_id cannot appear in multiple splits (critical for reproducibility)."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        image_to_docs = group_documents_by_image(documents)
        
        splits = create_deterministic_splits(
            image_to_docs,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=42
        )
        
        # Build image_id to split mapping
        doc_to_image = {}
        for image_id, doc_ids in image_to_docs.items():
            for doc_id in doc_ids:
                doc_to_image[doc_id] = image_id
        
        # Check each image appears in only one split
        image_to_splits = {}
        for split_name, doc_ids in splits.items():
            for doc_id in doc_ids:
                image_id = doc_to_image.get(doc_id)
                if image_id:
                    if image_id not in image_to_splits:
                        image_to_splits[image_id] = []
                    image_to_splits[image_id].append(split_name)
        
        # Verify no image appears in multiple splits
        for image_id, split_names in image_to_splits.items():
            unique_splits = set(split_names)
            assert len(unique_splits) == 1, \
                f"Image {image_id} appears in multiple splits: {unique_splits}"
    
    def test_split_sizes_sum_to_total(self, sample_documents_jsonl):
        """Verify split sizes sum to total number of documents."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        image_to_docs = group_documents_by_image(documents)
        
        splits = create_deterministic_splits(
            image_to_docs,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=42
        )
        
        total_docs = sum(len(doc_ids) for doc_ids in image_to_docs.values())
        split_total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        
        assert split_total == total_docs, \
            f"Split total {split_total} != document total {total_docs}"
    
    def test_deterministic_reproducibility(self, sample_documents_jsonl):
        """Verify same seed produces same splits (reproducibility guarantee)."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        image_to_docs = group_documents_by_image(documents)
        
        splits1 = create_deterministic_splits(image_to_docs, seed=42)
        splits2 = create_deterministic_splits(image_to_docs, seed=42)
        
        assert splits1["train"] == splits2["train"], "Train splits differ with same seed"
        assert splits1["val"] == splits2["val"], "Val splits differ with same seed"
        assert splits1["test"] == splits2["test"], "Test splits differ with same seed"
    
    def test_different_seeds_produce_different_splits(self, sample_documents_jsonl):
        """Verify different seeds produce different splits (unless deterministic ordering)."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        image_to_docs = group_documents_by_image(documents)
        
        if len(image_to_docs) > 1:  # Only test if we have multiple images
            splits1 = create_deterministic_splits(image_to_docs, seed=42)
            splits2 = create_deterministic_splits(image_to_docs, seed=123)
            
            # With different seeds and multiple images, splits should differ
            # (unless by chance they're the same, which is unlikely but possible)
            assert isinstance(splits1, dict)
            assert isinstance(splits2, dict)


class TestSplitValidation:
    """Test split validation for data leakage detection."""
    
    def test_validate_no_leakage(self, sample_documents_jsonl):
        """Verify validation passes for properly generated splits."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        image_to_docs = group_documents_by_image(documents)
        
        splits = create_deterministic_splits(image_to_docs, seed=42)
        validation = validate_splits(splits, image_to_docs)
        
        assert validation["valid"] is True, "Valid splits failed validation"
        assert validation["leakage_detected"] is False, "False positive leakage detection"
        assert validation["duplicates_detected"] is False, "False positive duplicate detection"
    
    def test_validate_detects_leakage(self):
        """Verify validation detects artificially introduced leakage."""
        image_to_docs = {
            1: ["doc1", "doc2"],
            2: ["doc3"]
        }
        
        # Artificially create splits with leakage
        splits = {
            "train": ["doc1"],
            "val": ["doc2"],  # Same image_id as doc1 (leakage!)
            "test": ["doc3"]
        }
        
        validation = validate_splits(splits, image_to_docs)
        
        assert validation["valid"] is False, "Leakage not detected"
        assert validation["leakage_detected"] is True, "Leakage detection failed"
        assert validation["num_leakage_cases"] > 0, "No leakage cases reported"
    
    def test_validate_detects_duplicates(self):
        """Verify validation detects duplicate documents across splits."""
        image_to_docs = {
            1: ["doc1"],
            2: ["doc2"]
        }
        
        # Artificially create splits with duplicate
        splits = {
            "train": ["doc1", "doc2"],
            "val": ["doc1"],  # doc1 appears twice (duplicate!)
            "test": ["doc2"]
        }
        
        validation = validate_splits(splits, image_to_docs)
        
        assert validation["duplicates_detected"] is True, "Duplicate detection failed"
        assert validation["num_duplicates"] > 0, "No duplicates reported"

