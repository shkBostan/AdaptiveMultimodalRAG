"""
Tests for retrieval modules.

Author: s Bostan
Created on: Nov, 2025
"""

import pytest
import numpy as np
from retrieval.retrieval_engine import RetrievalEngine
from retrieval.search_utils import cosine_similarity, euclidean_distance


class TestRetrievalEngine:
    """Test cases for retrieval engine."""
    
    def test_index_building(self):
        """Test index building."""
        engine = RetrievalEngine(embedding_dim=128)
        embeddings = np.random.rand(10, 128).astype('float32')
        documents = [{'id': i, 'content': f'doc {i}'} for i in range(10)]
        engine.build_index(embeddings, documents)
        assert engine.index is not None
        assert len(engine.documents) == 10
    
    def test_search(self):
        """Test search functionality."""
        engine = RetrievalEngine(embedding_dim=128)
        embeddings = np.random.rand(10, 128).astype('float32')
        documents = [{'id': i, 'content': f'doc {i}'} for i in range(10)]
        engine.build_index(embeddings, documents)
        
        query = np.random.rand(128).astype('float32')
        results = engine.search(query, top_k=3)
        assert len(results) == 3
        assert all('score' in r for r in results)


class TestSearchUtils:
    """Test cases for search utilities."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1, 0])
        vec2 = np.array([1, 0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(1.0)
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        vec1 = np.array([0, 0])
        vec2 = np.array([3, 4])
        assert euclidean_distance(vec1, vec2) == pytest.approx(5.0)

