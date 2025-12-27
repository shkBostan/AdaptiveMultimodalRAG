"""
Integration tests for full AdaptiveMultimodalRAG pipeline.

Simulates complete pipeline: load → embed → index → retrieve → generate.

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

from tests.fixtures import temp_dir, sample_documents


class TestFullPipeline:
    """Integration tests for full RAG pipeline."""
    
    @patch('retrieval.retrieval_engine.BERTEmbedding')
    @patch('generation.rag_module.AutoTokenizer')
    @patch('generation.rag_module.AutoModelForCausalLM')
    def test_full_pipeline_mocked(self, mock_model_class, mock_tokenizer_class, mock_bert_class):
        """Test full pipeline with all components mocked."""
        try:
            from src.retrieval import DocumentLoader, Document
            from src.retrieval import RetrievalEngine
            from src.generation import RAGModule
            
            # Setup mock embedding model
            mock_embedding_model = Mock()
            np.random.seed(42)
            mock_embedding_model.get_embedding = Mock(
                side_effect=lambda text: np.random.rand(128).astype('float32')
            )
            mock_bert_class.return_value = mock_embedding_model
            
            # Setup mock RAG model
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "<eos>"
            mock_tokenizer.eos_token_id = 1
            mock_tokenizer.decode.return_value = "Context:\nTest\n\nQuery:\nWhat?\n\nAnswer: This is a generated response about the query."
            mock_tokenizer.return_value = {"input_ids": Mock()}
            
            mock_model = Mock()
            mock_output = Mock()
            mock_output[0] = Mock()
            mock_model.generate.return_value = mock_output
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Step 1: Load documents
            loader = DocumentLoader()
            docs = [
                Document(id="1", content="RAG is a technique", metadata={"source": "doc1.txt"}),
                Document(id="2", content="Retrieval augmented generation", metadata={"source": "doc2.txt"}),
                Document(id="3", content="Machine learning models", metadata={"source": "doc3.txt"})
            ]
            
            assert len(docs) == 3
            
            # Step 2: Generate embeddings (mocked)
            embedding_fn = lambda text: np.random.rand(128).astype('float32')
            embeddings = np.array([embedding_fn(doc.content) for doc in docs])
            
            assert embeddings.shape == (3, 128)
            
            # Step 3: Build index
            engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
            engine.load_documents(docs)
            engine.build_index(embedding_fn=embedding_fn)
            
            assert engine.index_builder is not None
            assert engine.index_builder.num_vectors == 3
            
            # Step 4: Retrieve documents
            query_embedding = np.random.rand(128).astype('float32')
            results = engine.search(query_embedding, top_k=2)
            
            assert len(results) == 2
            assert all('content' in r or hasattr(r, 'content') for r in results)
            
            # Step 5: Generate response
            rag = RAGModule()
            rag.tokenizer = mock_tokenizer
            rag.model = mock_model
            
            response = rag.generate("What is RAG?", results)
            
            assert isinstance(response, str)
            assert len(response) > 0
            
        except ImportError:
            pytest.skip("Required modules not available")
    
    def test_pipeline_with_synthetic_data(self, temp_dir):
        """Test pipeline with synthetic data and minimal mocks."""
        try:
            from retrieval.retrieval_engine import RetrievalEngine
            from generation.rag_module import RAGModule
            
            # Create synthetic documents
            docs = [
                {"id": "1", "content": "Document about machine learning"},
                {"id": "2", "content": "Document about neural networks"},
                {"id": "3", "content": "Document about deep learning"}
            ]
            
            # Create synthetic embeddings
            np.random.seed(42)
            embeddings = np.random.rand(3, 128).astype('float32')
            
            # Build index
            engine = RetrievalEngine(embedding_dim=128)
            engine.build_index(embeddings, docs)
            
            assert engine.index is not None
            
            # Search
            query_embedding = np.random.rand(128).astype('float32')
            results = engine.search(query_embedding, top_k=2)
            
            assert len(results) == 2
            
            # Prepare context (without actual generation)
            rag = RAGModule()
            context = rag.prepare_context(results)
            
            assert isinstance(context, str)
            assert len(context) > 0
            
        except Exception as e:
            pytest.fail(f"Pipeline test failed: {e}")
    
    def test_pipeline_persistence_integration(self, temp_dir):
        """Test pipeline with persistence integration."""
        try:
            from storage.persistence_manager import PersistenceManager
            from retrieval.retrieval_engine import RetrievalEngine
            from retrieval.index_builder import IndexBuilder
            
            # Create synthetic data
            docs = [
                {"id": "1", "content": "Test document 1"},
                {"id": "2", "content": "Test document 2"}
            ]
            embeddings = np.random.rand(2, 128).astype('float32')
            
            # Build index
            engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
            engine.build_index(embeddings, docs)
            
            # Save state
            manager = PersistenceManager(default_storage_dir=str(temp_dir))
            
            # Save embeddings
            embeddings_path = temp_dir / "test_embeddings.pkl"
            manager.save_embeddings(embeddings, embeddings_path)
            
            # Save index
            index_path = temp_dir / "test_index.faiss"
            manager.save_index(engine.index_builder, index_path)
            
            # Load and verify
            loaded_embeddings = manager.load_embeddings(embeddings_path)
            assert loaded_embeddings.shape == embeddings.shape
            
            new_engine = RetrievalEngine(embedding_dim=128, use_index_builder=True)
            manager.load_index(new_engine.index_builder, index_path)
            assert new_engine.index_builder.num_vectors == 2
            
        except ImportError:
            pytest.skip("PersistenceManager not available")
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        from retrieval.retrieval_engine import RetrievalEngine
        
        # Test with invalid embeddings
        engine = RetrievalEngine(embedding_dim=128)
        
        with pytest.raises((ValueError, AttributeError)):
            # Wrong shape embeddings
            wrong_embeddings = np.random.rand(5, 64).astype('float32')
            docs = [{"id": "1", "content": "Test"}]
            engine.build_index(wrong_embeddings, docs)
    
    def test_pipeline_consistency(self):
        """Test pipeline consistency across components."""
        try:
            from retrieval.retrieval_engine import RetrievalEngine
            from generation.rag_module import RAGModule
            
            # Create consistent test data
            docs = [
                {"id": "1", "content": "Alpha document", "source": "alpha.txt"},
                {"id": "2", "content": "Beta document", "source": "beta.txt"},
                {"id": "3", "content": "Gamma document", "source": "gamma.txt"}
            ]
            
            np.random.seed(42)
            embeddings = np.random.rand(3, 128).astype('float32')
            
            # Build and search
            engine = RetrievalEngine(embedding_dim=128)
            engine.build_index(embeddings, docs)
            
            query_embedding = np.random.rand(128).astype('float32')
            results = engine.search(query_embedding, top_k=3)
            
            # Verify results have expected structure
            assert len(results) == 3
            for result in results:
                assert 'id' in result
                assert 'content' in result
                assert 'score' in result
                assert 'rank' in result
            
            # Prepare context
            rag = RAGModule()
            context = rag.prepare_context(results)
            
            # Verify context contains document content
            assert any(doc['content'] in context for doc in docs)
            
        except Exception as e:
            pytest.fail(f"Consistency test failed: {e}")

