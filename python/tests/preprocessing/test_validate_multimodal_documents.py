"""
Unit tests for validate_multimodal_documents.py preprocessing script.

Research intent: Verify document validation correctly identifies data quality
issues to prevent downstream pipeline failures and ensure research reproducibility.
"""

import pytest
import json
from pathlib import Path
from typing import List, Dict
import sys
import importlib.util

# Load preprocessing module using importlib for reliable imports
_preprocessing_dir = Path(__file__).parent.parent.parent / "experiments" / "preprocessing"
_module_path = _preprocessing_dir / "validate_multimodal_documents.py"
spec = importlib.util.spec_from_file_location("validate_multimodal_documents", _module_path)
validate_multimodal_documents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_multimodal_documents)

# Import functions for easier access
validate_image_path = validate_multimodal_documents.validate_image_path
validate_text_content = validate_multimodal_documents.validate_text_content
validate_metadata = validate_multimodal_documents.validate_metadata
validate_document = validate_multimodal_documents.validate_document
validate_all_documents = validate_multimodal_documents.validate_all_documents
load_documents = validate_multimodal_documents.load_documents


class TestImagePathValidation:
    """Test image path validation."""
    
    def test_validate_existing_image(self, dummy_image_path):
        """Verify valid image paths pass validation."""
        result = validate_image_path(dummy_image_path, "test_doc")
        assert result["valid"] is True
        assert result["exists"] is True
        assert result["readable"] is True
        assert result["error"] is None
    
    def test_validate_nonexistent_image(self, tmp_path):
        """Verify missing images fail validation."""
        fake_path = tmp_path / "nonexistent.jpg"
        result = validate_image_path(str(fake_path), "test_doc")
        assert result["valid"] is False
        assert result["exists"] is False
        assert result["error"] is not None


class TestTextContentValidation:
    """Test text content validation."""
    
    def test_validate_non_empty_text(self):
        """Verify non-empty text passes validation."""
        result = validate_text_content("Valid caption text", "test_doc")
        assert result["valid"] is True
        assert result["is_string"] is True
        assert result["non_empty"] is True
    
    def test_validate_empty_text(self):
        """Verify empty text fails validation."""
        result = validate_text_content("", "test_doc")
        assert result["valid"] is False
        assert result["non_empty"] is False
        assert result["error"] is not None
    
    def test_validate_whitespace_only_text(self):
        """Verify whitespace-only text fails validation."""
        result = validate_text_content("   \n\t  ", "test_doc")
        assert result["valid"] is False
        assert result["non_empty"] is False
    
    def test_validate_non_string_content(self):
        """Verify non-string content fails validation."""
        result = validate_text_content(123, "test_doc")
        assert result["valid"] is False
        assert result["is_string"] is False
        assert result["error"] is not None


class TestMetadataValidation:
    """Test metadata schema validation."""
    
    def test_validate_complete_metadata(self):
        """Verify complete metadata passes validation."""
        metadata = {
            "split": "train",
            "image_id": 1,
            "caption_id": 0,
            "coco_file": "captions_train.json"
        }
        result = validate_metadata(metadata, "test_doc")
        assert result["valid"] is True
        assert result["is_dict"] is True
        assert len(result["missing_fields"]) == 0
    
    def test_validate_missing_metadata_fields(self):
        """Verify incomplete metadata fails validation."""
        metadata = {
            "split": "train",
            "image_id": 1
            # Missing caption_id and coco_file
        }
        result = validate_metadata(metadata, "test_doc")
        assert result["valid"] is False
        assert len(result["missing_fields"]) > 0
    
    def test_validate_non_dict_metadata(self):
        """Verify non-dictionary metadata fails validation."""
        result = validate_metadata("not a dict", "test_doc")
        assert result["valid"] is False
        assert result["is_dict"] is False


class TestDocumentValidation:
    """Test full document validation."""
    
    def test_valid_document_passes(self, dummy_image_path):
        """Verify valid documents pass validation."""
        doc = {
            "id": "test_doc",
            "content": "Valid caption text",
            "image_path": dummy_image_path,
            "metadata": {
                "split": "train",
                "image_id": 1,
                "caption_id": 0,
                "coco_file": "captions_train.json"
            }
        }
        
        validation = validate_document(doc, strict=True)
        assert validation["valid"] is True
        assert len(validation["errors"]) == 0
    
    def test_invalid_document_fails_strict(self, tmp_path):
        """Verify invalid documents fail in strict mode."""
        doc = {
            "id": "invalid_doc",
            "content": "",  # Empty content
            "image_path": str(tmp_path / "nonexistent.jpg"),
            "metadata": {}
        }
        
        validation = validate_document(doc, strict=True)
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0
    
    def test_missing_image_path_fails_strict(self):
        """Verify missing image_path fails in strict mode."""
        doc = {
            "id": "test_doc",
            "content": "Valid text",
            "metadata": {"split": "train"}
        }
        
        validation = validate_document(doc, strict=True)
        assert validation["valid"] is False
        assert any("image_path" in error.lower() for error in validation["errors"])
    
    def test_strict_vs_non_strict_mode(self):
        """Verify strict mode is more restrictive than non-strict."""
        doc = {
            "id": "test_doc",
            "content": "Valid text"
            # Missing image_path and metadata
        }
        
        strict_validation = validate_document(doc, strict=True)
        non_strict_validation = validate_document(doc, strict=False)
        
        # Strict mode should have more errors
        assert len(strict_validation["errors"]) >= len(non_strict_validation["errors"])


class TestBatchValidation:
    """Test batch document validation."""
    
    def test_validate_all_valid_documents(self, sample_documents_jsonl):
        """Verify all valid documents pass batch validation."""
        documents = load_documents(sample_documents_jsonl)
        report = validate_all_documents(documents, strict=True, sample_size=None)
        
        assert report["validation_summary"]["total_documents"] > 0
        assert report["validation_summary"]["validation_rate"] >= 0.0
    
    def test_validate_all_detects_invalid(self, invalid_documents_jsonl):
        """Verify batch validation correctly identifies invalid documents."""
        documents = load_documents(invalid_documents_jsonl)
        report = validate_all_documents(documents, strict=True, sample_size=None)
        
        assert report["validation_summary"]["invalid_documents"] > 0
        assert report["error_summary"]["missing_images"] > 0 or \
               report["error_summary"]["empty_captions"] > 0
    
    def test_validation_sample_size(self, sample_documents_jsonl):
        """Verify sample_size parameter limits validation scope."""
        documents = load_documents(sample_documents_jsonl)
        report = validate_all_documents(documents, strict=True, sample_size=2)
        
        assert report["validation_summary"]["total_documents"] <= 2


class TestValidationReport:
    """Test validation report generation."""
    
    def test_report_contains_summary(self, sample_documents_jsonl):
        """Verify validation report contains summary statistics."""
        documents = load_documents(sample_documents_jsonl)
        report = validate_all_documents(documents, strict=True)
        
        assert "validation_summary" in report
        assert "error_summary" in report
        assert "generated_at" in report
        
        summary = report["validation_summary"]
        assert "total_documents" in summary
        assert "valid_documents" in summary
        assert "invalid_documents" in summary
        assert "validation_rate" in summary
    
    def test_report_error_counts(self, invalid_documents_jsonl):
        """Verify error summary contains error counts."""
        documents = load_documents(invalid_documents_jsonl)
        report = validate_all_documents(documents, strict=True)
        
        error_summary = report["error_summary"]
        assert "missing_images" in error_summary
        assert "empty_captions" in error_summary
        assert "invalid_metadata" in error_summary
        assert "error_counts" in error_summary

