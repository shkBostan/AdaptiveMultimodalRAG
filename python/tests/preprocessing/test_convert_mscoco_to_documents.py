"""
Unit tests for convert_mscoco_to_documents.py preprocessing script.

Research intent: Verify that MS COCO annotation conversion produces valid,
reproducible documents with correct schema for downstream pipeline consumption.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import importlib.util

# Load preprocessing module using importlib for reliable imports
_preprocessing_dir = Path(__file__).parent.parent.parent / "experiments" / "preprocessing"
_module_path = _preprocessing_dir / "convert_mscoco_to_documents.py"
spec = importlib.util.spec_from_file_location("convert_mscoco_to_documents", _module_path)
convert_mscoco_to_documents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_mscoco_to_documents)

# Import functions for easier access
create_document_id = convert_mscoco_to_documents.create_document_id
validate_image_path = convert_mscoco_to_documents.validate_image_path
create_document = convert_mscoco_to_documents.create_document


def load_documents_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Helper to load documents from JSONL."""
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line.strip()))
    return documents


class TestDocumentIDGeneration:
    """Test deterministic document ID generation for reproducibility."""
    
    def test_document_id_format(self):
        """Verify document IDs follow expected format."""
        doc_id = create_document_id("train", 123, 456)
        assert doc_id == "mscoco_train_000000000123_456"
        assert doc_id.startswith("mscoco_")
        assert "train" in doc_id
    
    def test_document_id_deterministic(self):
        """Ensure same inputs produce same ID (reproducibility)."""
        id1 = create_document_id("val", 789, 10)
        id2 = create_document_id("val", 789, 10)
        assert id1 == id2
    
    def test_document_id_zero_padding(self):
        """Verify image IDs are zero-padded to 12 digits."""
        doc_id = create_document_id("train", 1, 0)
        assert "000000000001" in doc_id


class TestImageValidation:
    """Test image path validation logic."""
    
    def test_validate_existing_image(self, dummy_image_path):
        """Verify valid image paths pass validation."""
        result = validate_image_path(dummy_image_path, "test_doc")
        assert result["valid"] is True
        assert result["exists"] is True
        assert result["readable"] is True
    
    def test_validate_nonexistent_image(self, tmp_path):
        """Verify missing images fail validation."""
        fake_path = tmp_path / "nonexistent.jpg"
        result = validate_image_path(str(fake_path), "test_doc")
        assert result["valid"] is False
        assert result["exists"] is False


class TestDocumentCreation:
    """Test document creation from COCO annotations."""
    
    def test_create_document_valid(self, dummy_image_path):
        """Verify valid document creation with all required fields."""
        image_info = {
            "id": 1,
            "file_name": Path(dummy_image_path).name,
            "width": 640,
            "height": 640
        }
        
        doc = create_document(
            split="train",
            image_id=1,
            caption_id=0,
            caption="A test caption",
            image_info=image_info,
            image_dir=Path(dummy_image_path).parent,
            coco_file="captions_train.json",
            normalize_text=False
        )
        
        assert doc is not None
        assert doc["id"] == "mscoco_train_000000000001_0"
        assert doc["content"] == "A test caption"
        assert "image_path" in doc
        assert doc["metadata"]["split"] == "train"
        assert doc["metadata"]["image_id"] == 1
        assert doc["metadata"]["caption_id"] == 0
        assert "preprocessing_version" in doc["metadata"]
    
    def test_create_document_empty_caption(self, dummy_image_path):
        """Verify empty captions are rejected."""
        image_info = {"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}
        
        doc = create_document(
            split="train",
            image_id=1,
            caption_id=0,
            caption="",
            image_info=image_info,
            image_dir=Path(dummy_image_path).parent,
            coco_file="captions_train.json"
        )
        
        assert doc is None
    
    def test_create_document_missing_image(self, tmp_path):
        """Verify documents with missing images are rejected."""
        image_info = {"id": 1, "file_name": "nonexistent.jpg", "width": 640, "height": 480}
        
        doc = create_document(
            split="train",
            image_id=1,
            caption_id=0,
            caption="Test caption",
            image_info=image_info,
            image_dir=tmp_path,
            coco_file="captions_train.json"
        )
        
        assert doc is None


class TestDocumentsJSONLOutput:
    """Test JSONL output file format and schema."""
    
    def test_output_jsonl_exists(self, sample_documents_jsonl):
        """Verify JSONL file is created."""
        assert sample_documents_jsonl.exists()
        assert sample_documents_jsonl.suffix == ".jsonl"
    
    def test_output_jsonl_valid_format(self, sample_documents_jsonl):
        """Verify JSONL contains valid JSON per line."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        assert len(documents) > 0
        
        for doc in documents:
            assert isinstance(doc, dict)
            assert "id" in doc
            assert "content" in doc
    
    def test_output_schema_compliance(self, sample_documents_jsonl):
        """Verify documents conform to required schema."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        
        required_fields = {"id", "content", "image_path", "metadata"}
        required_metadata = {"split", "image_id", "caption_id", "coco_file"}
        
        for doc in documents:
            assert required_fields.issubset(set(doc.keys())), \
                f"Missing required fields in document {doc.get('id')}"
            
            metadata = doc.get("metadata", {})
            assert required_metadata.issubset(set(metadata.keys())), \
                f"Missing required metadata in document {doc.get('id')}"
    
    def test_no_empty_content(self, sample_documents_jsonl):
        """Verify no documents have empty text content."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        
        for doc in documents:
            content = doc.get("content", "").strip()
            assert len(content) > 0, \
                f"Document {doc.get('id')} has empty content"
    
    def test_no_missing_ids(self, sample_documents_jsonl):
        """Verify all documents have non-empty IDs."""
        documents = load_documents_jsonl(sample_documents_jsonl)
        
        for doc in documents:
            doc_id = doc.get("id", "").strip()
            assert len(doc_id) > 0, "Document has empty or missing ID"

