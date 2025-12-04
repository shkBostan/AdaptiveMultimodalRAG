"""
Comprehensive tests for IndexBuilder.

Tests embedding validation, index build/search, and save/load.

Author: s Bostan
Created on: Nov, 2025
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from retrieval.index_builder import IndexBuilder, validate_shape, validate_query
from tests.fixtures import temp_dir, sample_embeddings, sample_query_embedding


class TestIndexBuilder:
    """Test cases for IndexBuilder."""
    
    def test_initialization(self):
        """Test IndexBuilder initialization."""
        builder = IndexBuilder(embedding_dim=128, index_type="flat_l2")
        
        assert builder.embedding_dim == 128
        assert builder.index_type == "flat_l2"
        assert builder.index is None
        assert builder.num_vectors == 0
    
    def test_initialization_ivf(self):
        """Test IndexBuilder initialization with IVF type."""
        builder = IndexBuilder(embedding_dim=128, index_type="ivf_flat", nlist=10)
        
        assert builder.embedding_dim == 128
        assert builder.index_type == "ivf_flat"
        assert builder.nlist == 10
    
    def test_initialization_invalid_dim(self):
        """Test initialization with invalid dimension."""
        with pytest.raises(ValueError):
            IndexBuilder(embedding_dim=0)
    
    def test_initialization_invalid_type(self):
        """Test initialization with invalid index type."""
        with pytest.raises(ValueError):
            IndexBuilder(embedding_dim=128, index_type="invalid")
    
    def test_build_flat_l2(self, sample_embeddings):
        """Test building flat L2 index."""
        builder = IndexBuilder(embedding_dim=128, index_type="flat_l2")
        builder.build(sample_embeddings)
        
        assert builder.index is not None
        assert builder.num_vectors == 10
        assert builder.is_trained
    
    def test_build_cosine(self, sample_embeddings):
        """Test building cosine similarity index."""
        builder = IndexBuilder(embedding_dim=128, index_type="flat_l2", distance_metric="cosine")
        builder.build(sample_embeddings)
        
        assert builder.index is not None
        assert builder.num_vectors == 10
    
    def test_build_ivf_flat(self):
        """Test building IVF flat index."""
        # Need enough vectors for IVF training
        embeddings = np.random.rand(50, 128).astype('float32')
        
        builder = IndexBuilder(embedding_dim=128, index_type="ivf_flat", nlist=10)
        builder.build(embeddings)
        
        assert builder.index is not None
        assert builder.num_vectors == 50
    
    def test_build_ivf_insufficient_vectors(self):
        """Test IVF with insufficient vectors falls back to flat."""
        embeddings = np.random.rand(5, 128).astype('float32')
        
        builder = IndexBuilder(embedding_dim=128, index_type="ivf_flat", nlist=10)
        builder.build(embeddings)
        
        # Should fall back to flat index
        assert builder.index is not None
        assert builder.index_type == "flat_l2"
    
    def test_build_invalid_shape(self):
        """Test building with invalid embedding shape."""
        builder = IndexBuilder(embedding_dim=128)
        
        # Wrong dimension
        wrong_dim = np.random.rand(10, 64).astype('float32')
        with pytest.raises(ValueError):
            builder.build(wrong_dim)
        
        # Wrong number of dimensions
        wrong_ndim = np.random.rand(10).astype('float32')
        with pytest.raises(ValueError):
            builder.build(wrong_ndim)
    
    def test_build_already_built(self, sample_embeddings):
        """Test building when index already exists."""
        builder = IndexBuilder(embedding_dim=128)
        builder.build(sample_embeddings)
        
        with pytest.raises(ValueError, match="already built"):
            builder.build(sample_embeddings)
    
    def test_add_embeddings(self, sample_embeddings):
        """Test adding embeddings incrementally."""
        builder = IndexBuilder(embedding_dim=128)
        builder.build(sample_embeddings)
        
        initial_count = builder.num_vectors
        
        new_embeddings = np.random.rand(5, 128).astype('float32')
        builder.add_embeddings(new_embeddings)
        
        assert builder.num_vectors == initial_count + 5
    
    def test_add_embeddings_not_built(self):
        """Test adding embeddings when index not built."""
        builder = IndexBuilder(embedding_dim=128)
        new_embeddings = np.random.rand(5, 128).astype('float32')
        
        with pytest.raises(ValueError, match="not built"):
            builder.add_embeddings(new_embeddings)
    
    def test_search_flat_l2(self, sample_embeddings, sample_query_embedding):
        """Test search with flat L2 index."""
        builder = IndexBuilder(embedding_dim=128, index_type="flat_l2")
        builder.build(sample_embeddings)
        
        scores, indices = builder.search(sample_query_embedding, top_k=3)
        
        assert len(scores) == 3
        assert len(indices) == 3
        assert all(0 <= idx < 10 for idx in indices)
        assert all(isinstance(score, (float, np.floating)) for score in scores)
    
    def test_search_cosine(self, sample_embeddings, sample_query_embedding):
        """Test search with cosine similarity."""
        builder = IndexBuilder(embedding_dim=128, distance_metric="cosine")
        builder.build(sample_embeddings)
        
        scores, indices = builder.search(sample_query_embedding, top_k=3)
        
        assert len(scores) == 3
        assert len(indices) == 3
    
    def test_search_top_k_larger_than_vectors(self, sample_embeddings, sample_query_embedding):
        """Test search with top_k larger than number of vectors."""
        builder = IndexBuilder(embedding_dim=128)
        builder.build(sample_embeddings)
        
        scores, indices = builder.search(sample_query_embedding, top_k=20)
        
        # Should return at most num_vectors results
        assert len(scores) <= 10
        assert len(indices) <= 10
    
    def test_search_not_built(self, sample_query_embedding):
        """Test search when index not built."""
        builder = IndexBuilder(embedding_dim=128)
        
        with pytest.raises(ValueError, match="not built"):
            builder.search(sample_query_embedding, top_k=5)
    
    def test_search_invalid_query_shape(self, sample_embeddings):
        """Test search with invalid query shape."""
        builder = IndexBuilder(embedding_dim=128)
        builder.build(sample_embeddings)
        
        # Wrong dimension
        wrong_query = np.random.rand(64).astype('float32')
        with pytest.raises(ValueError):
            builder.search(wrong_query, top_k=5)
    
    def test_save_and_load(self, temp_dir, sample_embeddings):
        """Test saving and loading index."""
        # Build and save
        builder1 = IndexBuilder(embedding_dim=128)
        builder1.build(sample_embeddings)
        
        index_path = temp_dir / "test_index.faiss"
        builder1.save(index_path)
        
        assert index_path.exists()
        
        # Load
        builder2 = IndexBuilder(embedding_dim=128)
        builder2.load(index_path)
        
        assert builder2.index is not None
        assert builder2.num_vectors == builder1.num_vectors
        assert builder2.embedding_dim == builder1.embedding_dim
    
    def test_save_not_built(self, temp_dir):
        """Test saving when index not built."""
        builder = IndexBuilder(embedding_dim=128)
        index_path = temp_dir / "test_index.faiss"
        
        with pytest.raises(ValueError, match="not built"):
            builder.save(index_path)
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        builder = IndexBuilder(embedding_dim=128)
        
        with pytest.raises(FileNotFoundError):
            builder.load("nonexistent_index.faiss")
    
    def test_load_dimension_mismatch(self, temp_dir, sample_embeddings):
        """Test loading index with dimension mismatch."""
        # Save with one dimension
        builder1 = IndexBuilder(embedding_dim=128)
        builder1.build(sample_embeddings)
        index_path = temp_dir / "test_index.faiss"
        builder1.save(index_path)
        
        # Try to load with different dimension
        builder2 = IndexBuilder(embedding_dim=256)
        with pytest.raises(ValueError, match="dimension mismatch"):
            builder2.load(index_path)
    
    def test_reset(self, sample_embeddings):
        """Test resetting index builder."""
        builder = IndexBuilder(embedding_dim=128)
        builder.build(sample_embeddings)
        
        assert builder.index is not None
        assert builder.num_vectors > 0
        
        builder.reset()
        
        assert builder.index is None
        assert builder.num_vectors == 0
        assert not builder.is_trained
    
    def test_get_stats(self, sample_embeddings):
        """Test getting index statistics."""
        builder = IndexBuilder(embedding_dim=128, index_type="flat_l2")
        builder.build(sample_embeddings)
        
        stats = builder.get_stats()
        
        assert stats['embedding_dim'] == 128
        assert stats['index_type'] == "flat_l2"
        assert stats['num_vectors'] == 10
        assert stats['is_built'] is True
        assert stats['is_trained'] is True


class TestIndexBuilderUtilities:
    """Test utility functions."""
    
    def test_validate_shape_valid(self):
        """Test validate_shape with valid input."""
        embeddings = np.random.rand(10, 128).astype('float32')
        num, dim = validate_shape(embeddings, expected_dim=128)
        
        assert num == 10
        assert dim == 128
    
    def test_validate_shape_wrong_dim(self):
        """Test validate_shape with wrong dimension."""
        embeddings = np.random.rand(10, 128).astype('float32')
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_shape(embeddings, expected_dim=256)
    
    def test_validate_shape_wrong_ndim(self):
        """Test validate_shape with wrong number of dimensions."""
        embeddings = np.random.rand(10).astype('float32')
        
        with pytest.raises(ValueError):
            validate_shape(embeddings)
    
    def test_validate_query_1d(self):
        """Test validate_query with 1D array."""
        query = np.random.rand(128).astype('float32')
        validated = validate_query(query, expected_dim=128)
        
        assert validated.shape == (1, 128)
        assert validated.dtype == np.float32
    
    def test_validate_query_2d(self):
        """Test validate_query with 2D array."""
        query = np.random.rand(1, 128).astype('float32')
        validated = validate_query(query, expected_dim=128)
        
        assert validated.shape == (1, 128)
    
    def test_validate_query_wrong_dim(self):
        """Test validate_query with wrong dimension."""
        query = np.random.rand(64).astype('float32')
        
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_query(query, expected_dim=128)

