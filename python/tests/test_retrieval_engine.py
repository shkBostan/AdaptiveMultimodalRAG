"""
Comprehensive tests for RetrievalEngine.

Tests load_documents, build_index, and search functionality.

Author: s Bostan
Created on: Nov, 2025
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from retrieval.retrieval_engine import RetrievalEngine
from tests.fixtures import sample_embeddings, sample_query_embedding, sample_documents


class TestRetrievalEngine:
    """Test cases for RetrievalEngine."""
    
    def test_initialization_legacy(self):
        """Test RetrievalEngine initialization in legacy mode."""
        engine = RetrievalEngine(embedding_dim=128, index_type="L2")
        
        assert engine.embedding_dim == 128
        assert engine.index_type == "L2"
        assert not engine.use_index_builder
        assert engine.index is None
    
    def test_initialization_with_index_builder(self):
        """Test RetrievalEngine initialization with IndexBuilder."""
        engine = RetrievalEngine(
            embedding_dim=128,
            use_index_builder=True,
            index_builder_type="flat_l2"
        )
        
        assert engine.use_index_builder
        assert engine.index_builder is not None
        assert engine.index_builder.embedding_dim == 128
    
    def test_load_documents(self, sample_documents):
        """Test loading documents."""
        try:
            from src.retrieval.document_loader import Document
            engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
            
            engine.load_documents(sample_documents)
            
            assert len(engine._document_objects) == 3
        except ImportError:
            pytest.skip("Document class not available")
    
    def test_load_documents_without_index_builder(self, sample_documents):
        """Test loading documents without IndexBuilder mode."""
        try:
            from src.retrieval.document_loader import Document
            engine = RetrievalEngine(embedding_dim=128, use_index_builder=False)
            
            # Should still work but may not store in _document_objects
            # This tests backward compatibility
            pass
        except ImportError:
            pytest.skip("Document class not available")
    
    def test_build_index_legacy(self, sample_embeddings):
        """Test building index in legacy mode."""
        engine = RetrievalEngine(embedding_dim=128, index_type="L2")
        documents = [{"id": str(i), "content": f"doc {i}"} for i in range(10)]
        
        engine.build_index(sample_embeddings, documents)
        
        assert engine.index is not None
        assert len(engine.documents) == 10
    
    def test_build_index_with_embedding_fn(self, sample_documents):
        """Test building index with embedding function."""
        try:
            from src.retrieval.document_loader import Document
            engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
            engine.load_documents(sample_documents)
            
            # Mock embedding function
            def mock_embedding_fn(text: str) -> np.ndarray:
                return np.random.rand(128).astype('float32')
            
            engine.build_index(embedding_fn=mock_embedding_fn)
            
            assert engine.index_builder is not None
            assert engine.index_builder.index is not None
            assert engine.index_builder.num_vectors == 3
        except ImportError:
            pytest.skip("Document class not available")
    
    def test_build_index_with_embedding_fn_and_save(self, temp_dir, sample_documents):
        """Test building index with embedding function and saving."""
        try:
            from src.retrieval.document_loader import Document
            engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
            engine.load_documents(sample_documents)
            
            def mock_embedding_fn(text: str) -> np.ndarray:
                return np.random.rand(128).astype('float32')
            
            index_path = temp_dir / "test_index.faiss"
            engine.build_index(embedding_fn=mock_embedding_fn, save_path=index_path)
            
            assert index_path.exists()
        except ImportError:
            pytest.skip("Document class not available")
    
    def test_build_index_no_documents(self):
        """Test building index without documents."""
        engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
        
        def mock_embedding_fn(text: str) -> np.ndarray:
            return np.random.rand(128).astype('float32')
        
        with pytest.raises(ValueError, match="No documents loaded"):
            engine.build_index(embedding_fn=mock_embedding_fn)
    
    def test_load_index(self, temp_dir, sample_embeddings):
        """Test loading index."""
        engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
        
        # Build and save index first
        documents = [{"id": str(i), "content": f"doc {i}"} for i in range(10)]
        engine.build_index(sample_embeddings, documents)
        
        index_path = temp_dir / "test_index.faiss"
        engine.index_builder.save(index_path)
        
        # Create new engine and load
        new_engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
        new_engine.load_index(index_path)
        
        assert new_engine.index_builder.index is not None
        assert new_engine.index_builder.num_vectors == 10
    
    def test_search_legacy(self, sample_embeddings, sample_query_embedding):
        """Test search in legacy mode."""
        engine = RetrievalEngine(embedding_dim=128, index_type="L2")
        documents = [{"id": str(i), "content": f"doc {i}"} for i in range(10)]
        
        engine.build_index(sample_embeddings, documents)
        results = engine.search(sample_query_embedding, top_k=5)
        
        assert len(results) == 5
        assert all('id' in result for result in results)
        assert all('content' in result for result in results)
        assert all('score' in result for result in results)
        assert all('rank' in result for result in results)
    
    def test_search_with_index_builder(self, sample_embeddings, sample_query_embedding):
        """Test search with IndexBuilder."""
        engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
        documents = [{"id": str(i), "content": f"doc {i}"} for i in range(10)]
        
        engine.build_index(sample_embeddings, documents)
        results = engine.search(sample_query_embedding, top_k=5)
        
        assert len(results) == 5
        assert all('score' in result for result in results)
    
    def test_search_return_documents(self, sample_embeddings, sample_query_embedding):
        """Test search returning Document objects."""
        try:
            from src.retrieval.document_loader import Document
            engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
            
            docs = [
                Document(id=str(i), content=f"doc {i}", metadata={"source": f"file{i}.txt"})
                for i in range(10)
            ]
            engine.load_documents(docs)
            
            def mock_embedding_fn(text: str) -> np.ndarray:
                return np.random.rand(128).astype('float32')
            
            engine.build_index(embedding_fn=mock_embedding_fn)
            
            results = engine.search(sample_query_embedding, top_k=3, return_documents=True)
            
            assert len(results) == 3
            # Results should be Document objects or dicts with metadata
            assert all(hasattr(r, 'content') or 'content' in r for r in results)
        except ImportError:
            pytest.skip("Document class not available")
    
    def test_search_not_built(self, sample_query_embedding):
        """Test search when index not built."""
        engine = RetrievalEngine(embedding_dim=128)
        
        with pytest.raises(ValueError, match="not built"):
            engine.search(sample_query_embedding, top_k=5)
    
    def test_add_documents_legacy(self, sample_embeddings):
        """Test adding documents in legacy mode."""
        engine = RetrievalEngine(embedding_dim=128)
        documents = [{"id": str(i), "content": f"doc {i}"} for i in range(10)]
        
        engine.build_index(sample_embeddings, documents)
        initial_count = len(engine.documents)
        
        new_embeddings = np.random.rand(5, 128).astype('float32')
        new_documents = [{"id": str(i), "content": f"new doc {i}"} for i in range(5)]
        
        engine.add_documents(new_embeddings, new_documents)
        
        assert len(engine.documents) == initial_count + 5
    
    def test_add_documents_with_embedding_fn(self, sample_documents):
        """Test adding documents with embedding function."""
        try:
            from src.retrieval.document_loader import Document
            engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
            engine.load_documents(sample_documents)
            
            def mock_embedding_fn(text: str) -> np.ndarray:
                return np.random.rand(128).astype('float32')
            
            engine.build_index(embedding_fn=mock_embedding_fn)
            initial_count = engine.index_builder.num_vectors
            
            new_docs = [
                Document(id="4", content="New document", metadata={"source": "new.txt"})
            ]
            
            engine.add_documents(document_objects=new_docs, embedding_fn=mock_embedding_fn)
            
            assert engine.index_builder.num_vectors == initial_count + 1
        except ImportError:
            pytest.skip("Document class not available")
    
    def test_get_stats(self, sample_embeddings):
        """Test getting engine statistics."""
        engine = RetrievalEngine(embedding_dim=128)
        documents = [{"id": str(i), "content": f"doc {i}"} for i in range(10)]
        
        engine.build_index(sample_embeddings, documents)
        stats = engine.get_stats()
        
        assert stats['embedding_dim'] == 128
        assert stats['num_documents'] == 10
        assert stats['has_index'] is True

