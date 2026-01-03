"""
Unit tests for generate_mscoco_queries.py preprocessing script.

Research intent: Verify query generation produces valid retrieval queries with
correct ground truth relevance pairs, ensuring evaluation reliability.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any, List
import random
import importlib.util

# Load preprocessing module using importlib for reliable imports
_preprocessing_dir = Path(__file__).parent.parent.parent / "experiments" / "preprocessing"
_module_path = _preprocessing_dir / "generate_mscoco_queries.py"
spec = importlib.util.spec_from_file_location("generate_mscoco_queries", _module_path)
generate_mscoco_queries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_mscoco_queries)

# Import functions for easier access
QueryStrategy = generate_mscoco_queries.QueryStrategy
generate_query_direct = generate_mscoco_queries.generate_query_direct
generate_query_partial = generate_mscoco_queries.generate_query_partial
generate_query_random_words = generate_mscoco_queries.generate_query_random_words
generate_query_text = generate_mscoco_queries.generate_query_text
create_relevance_pairs = generate_mscoco_queries.create_relevance_pairs
load_documents = generate_mscoco_queries.load_documents
build_image_id_to_documents = generate_mscoco_queries.build_image_id_to_documents
generate_queries = generate_mscoco_queries.generate_queries
save_queries_jsonl = generate_mscoco_queries.save_queries_jsonl


def load_queries_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Helper to load queries from JSONL."""
    queries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    return queries


class TestQueryGenerationStrategies:
    """Test query generation strategies for diversity."""
    
    def test_direct_strategy(self):
        """Verify direct strategy uses caption as-is."""
        caption = "A person riding a bicycle on a city street"
        query = generate_query_direct(caption)
        assert query == caption.strip()
    
    def test_partial_strategy(self):
        """Verify partial strategy uses first part of caption."""
        caption = "A person riding a bicycle on a city street"
        query = generate_query_partial(caption, ratio=0.5)
        assert len(query.split()) < len(caption.split())
        assert query in caption
    
    def test_random_words_strategy(self):
        """Verify random words strategy selects subset."""
        random.seed(42)
        caption = "A person riding a bicycle on a city street"
        query = generate_query_random_words(caption, ratio=0.6)
        words = caption.split()
        query_words = query.split()
        assert len(query_words) <= len(words)
        assert all(word in words for word in query_words)
    
    def test_query_strategy_enum(self):
        """Verify strategy enum values are correct."""
        assert QueryStrategy.DIRECT.value == "direct"
        assert QueryStrategy.PARTIAL.value == "partial"
        assert QueryStrategy.RANDOM_WORDS.value == "random_words"


class TestQueryTextGeneration:
    """Test query text generation wrapper."""
    
    def test_generate_query_direct(self):
        """Test direct query generation."""
        caption = "Test caption text"
        query = generate_query_text(caption, QueryStrategy.DIRECT)
        assert query == caption
    
    def test_generate_query_partial(self):
        """Test partial query generation."""
        caption = "A B C D E F"
        query = generate_query_text(caption, QueryStrategy.PARTIAL, {"ratio": 0.5})
        assert len(query.split()) < len(caption.split())
    
    def test_generate_query_invalid_strategy(self):
        """Verify invalid strategy raises error."""
        with pytest.raises(ValueError):
            generate_query_text("test", "invalid_strategy")


class TestRelevancePairCreation:
    """Test ground truth relevance pair generation."""
    
    def test_relevance_same_image_id(self):
        """Verify documents with same image_id are marked as relevant."""
        query_doc = {
            "id": "doc1",
            "metadata": {"image_id": 1, "split": "val"}
        }
        
        all_documents = [
            {"id": "doc1", "metadata": {"image_id": 1, "split": "val"}},
            {"id": "doc2", "metadata": {"image_id": 1, "split": "train"}},
            {"id": "doc3", "metadata": {"image_id": 2, "split": "train"}}
        ]
        
        image_id_index = {1: [all_documents[0], all_documents[1]], 2: [all_documents[2]]}
        
        relevant_ids = create_relevance_pairs(
            query_doc, all_documents, image_id_index, "val"
        )
        
        assert "doc2" in relevant_ids
        assert "doc1" not in relevant_ids  # Exclude query document itself
    
    def test_relevance_cross_split_filtering(self):
        """Verify val queries only map to train documents."""
        query_doc = {
            "id": "val_doc",
            "metadata": {"image_id": 1, "split": "val"}
        }
        
        all_documents = [
            {"id": "val_doc", "metadata": {"image_id": 1, "split": "val"}},
            {"id": "train_doc", "metadata": {"image_id": 1, "split": "train"}},
            {"id": "val_doc2", "metadata": {"image_id": 1, "split": "val"}}
        ]
        
        image_id_index = {1: all_documents}
        
        relevant_ids = create_relevance_pairs(
            query_doc, all_documents, image_id_index, "val"
        )
        
        assert "train_doc" in relevant_ids
        assert "val_doc2" not in relevant_ids  # Same split excluded


class TestQueryJSONLOutput:
    """Test query JSONL output format."""
    
    def test_queries_non_empty(self, sample_documents_jsonl, tmp_path):
        """Verify generated queries have non-empty text."""
        documents = load_documents(sample_documents_jsonl)
        image_id_index = build_image_id_to_documents(documents)
        
        queries = generate_queries(
            documents=documents,
            image_id_index=image_id_index,
            split="val",
            strategy=QueryStrategy.DIRECT,
            max_queries=10
        )
        
        assert len(queries) > 0
        for query in queries:
            assert len(query["query_text"].strip()) > 0, \
                f"Query {query['query_id']} has empty text"
    
    def test_query_schema(self, sample_documents_jsonl, tmp_path):
        """Verify queries conform to required schema."""
        documents = load_documents(sample_documents_jsonl)
        image_id_index = build_image_id_to_documents(documents)
        
        queries = generate_queries(
            documents=documents,
            image_id_index=image_id_index,
            split="val",
            strategy=QueryStrategy.DIRECT,
            max_queries=5
        )
        
        output_file = tmp_path / "queries.jsonl"
        save_queries_jsonl(queries, output_file)
        
        loaded_queries = load_queries_jsonl(output_file)
        
        required_fields = {"query_id", "query_text", "relevant_doc_ids", "metadata"}
        required_metadata = {"split", "source", "generation_strategy"}
        
        for query in loaded_queries:
            assert required_fields.issubset(set(query.keys())), \
                f"Missing required fields in query {query.get('query_id')}"
            
            metadata = query.get("metadata", {})
            assert required_metadata.issubset(set(metadata.keys())), \
                f"Missing required metadata in query {query.get('query_id')}"
    
    def test_relevance_mapping_integrity(self, sample_documents_jsonl, tmp_path):
        """Verify relevance mapping maintains image_id relationships."""
        documents = load_documents(sample_documents_jsonl)
        image_id_index = build_image_id_to_documents(documents)
        
        queries = generate_queries(
            documents=documents,
            image_id_index=image_id_index,
            split="val",
            strategy=QueryStrategy.DIRECT,
            max_queries=5
        )
        
        # Build doc_id to image_id mapping
        doc_to_image = {doc["id"]: doc["metadata"]["image_id"] for doc in documents}
        
        for query in queries:
            query_image_id = query["metadata"].get("image_id")
            for relevant_doc_id in query["relevant_doc_ids"]:
                relevant_image_id = doc_to_image.get(relevant_doc_id)
                assert relevant_image_id == query_image_id, \
                    f"Relevance mismatch: query image_id {query_image_id} != relevant doc image_id {relevant_image_id}"

