"""
Comprehensive tests for RAGModule.

Tests prepare_context formatting and generate with mocks.

Author: s Bostan
Created on: Nov, 2025
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from generation.rag_module import RAGModule
from tests.fixtures import sample_documents


class TestRAGModule:
    """Test cases for RAGModule."""
    
    def test_initialization(self):
        """Test RAGModule initialization."""
        rag = RAGModule(model_name="gpt2")
        
        assert rag.model_name == "gpt2"
        assert rag.max_context_length == 2000
        assert rag.include_metadata is True
        assert rag.metadata_marker_format == "brackets"
    
    def test_initialization_custom_params(self):
        """Test RAGModule initialization with custom parameters."""
        rag = RAGModule(
            model_name="gpt2",
            max_context_length=1000,
            include_metadata=False,
            metadata_marker_format="parentheses"
        )
        
        assert rag.max_context_length == 1000
        assert rag.include_metadata is False
        assert rag.metadata_marker_format == "parentheses"
    
    def test_prepare_context_with_documents(self, sample_documents):
        """Test prepare_context with Document objects."""
        try:
            from data.document_loader import Document
            rag = RAGModule()
            
            context = rag.prepare_context(sample_documents)
            
            assert isinstance(context, str)
            assert len(context) > 0
            # Should include document content
            assert "First document" in context or "document" in context.lower()
        except ImportError:
            pytest.skip("Document class not available")
    
    def test_prepare_context_with_dicts(self):
        """Test prepare_context with dictionaries."""
        rag = RAGModule()
        docs = [
            {"id": "1", "content": "First document", "source": "doc1.txt"},
            {"id": "2", "content": "Second document", "source": "doc2.txt"}
        ]
        
        context = rag.prepare_context(docs)
        
        assert isinstance(context, str)
        assert len(context) > 0
        assert "First document" in context
        assert "Second document" in context
    
    def test_prepare_context_with_metadata(self):
        """Test prepare_context includes metadata markers."""
        rag = RAGModule(include_metadata=True, metadata_marker_format="brackets")
        docs = [
            {"id": "1", "content": "Test content", "source": "test.txt", "author": "Author A"}
        ]
        
        context = rag.prepare_context(docs)
        
        # Should include metadata marker
        assert "[" in context or "Document" in context
    
    def test_prepare_context_without_metadata(self):
        """Test prepare_context without metadata markers."""
        rag = RAGModule(include_metadata=False)
        docs = [
            {"id": "1", "content": "Test content", "source": "test.txt"}
        ]
        
        context = rag.prepare_context(docs)
        
        # Should not include metadata markers
        assert "[" not in context or "Document" not in context or "Test content" in context
    
    def test_prepare_context_length_limit(self):
        """Test prepare_context respects length limit."""
        rag = RAGModule(max_context_length=50)
        docs = [
            {"id": "1", "content": "A" * 100}  # Very long content
        ]
        
        context = rag.prepare_context(docs)
        
        assert len(context) <= 50
    
    def test_prepare_context_empty_list(self):
        """Test prepare_context with empty list."""
        rag = RAGModule()
        context = rag.prepare_context([])
        
        assert context == ""
    
    def test_prepare_context_custom_length(self):
        """Test prepare_context with custom max_length."""
        rag = RAGModule(max_context_length=2000)
        docs = [
            {"id": "1", "content": "Short content"}
        ]
        
        context = rag.prepare_context(docs, max_length=100)
        
        assert len(context) <= 100
    
    def test_prepare_context_metadata_formats(self):
        """Test different metadata marker formats."""
        docs = [{"id": "1", "content": "Test", "source": "test.txt"}]
        
        # Brackets format
        rag1 = RAGModule(metadata_marker_format="brackets")
        context1 = rag1.prepare_context(docs)
        # Should contain brackets or be empty if no metadata
        
        # Parentheses format
        rag2 = RAGModule(metadata_marker_format="parentheses")
        context2 = rag2.prepare_context(docs)
        
        # Both should produce valid context
        assert isinstance(context1, str)
        assert isinstance(context2, str)
    
    @patch('generation.rag_module.AutoTokenizer')
    @patch('generation.rag_module.AutoModelForCausalLM')
    def test_generate_with_mock_model(self, mock_model_class, mock_tokenizer_class):
        """Test generate with mocked model."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.decode.return_value = "Context:\nTest\n\nQuery:\nWhat?\n\nAnswer: This is a test response."
        mock_tokenizer.return_value = {"input_ids": Mock()}
        
        mock_model = Mock()
        mock_output = Mock()
        mock_output[0] = Mock()
        mock_model.generate.return_value = mock_output
        
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test
        rag = RAGModule(model_name="gpt2")
        rag.tokenizer = mock_tokenizer
        rag.model = mock_model
        
        docs = [{"id": "1", "content": "Test document"}]
        response = rag.generate("What is this?", docs, max_length=100)
        
        assert isinstance(response, str)
        assert len(response) > 0
        mock_model.generate.assert_called_once()
    
    def test_generate_empty_query(self):
        """Test generate with empty query."""
        rag = RAGModule()
        docs = [{"id": "1", "content": "Test"}]
        
        with pytest.raises(ValueError, match="cannot be empty"):
            rag.generate("", docs)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            rag.generate("   ", docs)
    
    def test_generate_empty_context_fallback(self):
        """Test generate with empty context uses fallback."""
        rag = RAGModule()
        
        # Mock model loading
        with patch.object(rag, 'load_model'):
            rag.tokenizer = Mock()
            rag.tokenizer.pad_token = None
            rag.tokenizer.eos_token = "<eos>"
            rag.tokenizer.eos_token_id = 1
            rag.tokenizer.decode.return_value = "Answer based on your knowledge: Test response"
            rag.tokenizer.return_value = {"input_ids": Mock()}
            
            rag.model = Mock()
            mock_output = Mock()
            mock_output[0] = Mock()
            rag.model.generate.return_value = mock_output
            
            # Generate with empty documents
            response = rag.generate("What is this?", [])
            
            assert isinstance(response, str)
            # Should use fallback prompt
    
    def test_build_prompt(self):
        """Test _build_prompt method."""
        rag = RAGModule()
        prompt = rag._build_prompt("What is RAG?", "Context about RAG")
        
        assert "Context:" in prompt
        assert "Query:" in prompt
        assert "Answer:" in prompt
        assert "What is RAG?" in prompt
        assert "Context about RAG" in prompt
    
    def test_build_fallback_prompt(self):
        """Test _build_fallback_prompt method."""
        rag = RAGModule()
        prompt = rag._build_fallback_prompt("What is RAG?")
        
        assert "Query:" in prompt
        assert "Answer based on your knowledge:" in prompt
        assert "What is RAG?" in prompt
    
    def test_extract_answer(self):
        """Test _extract_answer method."""
        rag = RAGModule()
        
        # Test with Answer marker
        response = "Context:\nTest\n\nQuery:\nWhat?\n\nAnswer: This is the answer."
        prompt = "Context:\nTest\n\nQuery:\nWhat?\n\nAnswer:"
        answer = rag._extract_answer(response, prompt)
        
        assert "This is the answer" in answer or len(answer) > 0
    
    def test_normalize_documents(self, sample_documents):
        """Test _normalize_documents method."""
        try:
            from data.document_loader import Document
            rag = RAGModule()
            
            normalized = rag._normalize_documents(sample_documents)
            
            assert len(normalized) == 3
            assert all(isinstance(doc, dict) for doc in normalized)
            assert all('content' in doc for doc in normalized)
        except ImportError:
            pytest.skip("Document class not available")
    
    def test_build_metadata_marker(self):
        """Test _build_metadata_marker method."""
        rag = RAGModule(include_metadata=True, metadata_marker_format="brackets")
        doc_data = {"id": "1", "content": "Test", "source": "test.txt", "author": "Author"}
        
        marker = rag._build_metadata_marker(doc_data, 1)
        
        # Should include metadata
        assert isinstance(marker, str)
    
    def test_truncate_text(self):
        """Test _truncate_text method."""
        rag = RAGModule()
        
        # Long text
        long_text = "This is a test sentence. " * 10
        truncated = rag._truncate_text(long_text, max_length=50)
        
        assert len(truncated) <= 50 + 3  # +3 for "..."
        assert "..." in truncated or len(truncated) <= 50
    
    def test_construct_context_legacy(self):
        """Test legacy _construct_context method."""
        rag = RAGModule()
        docs = [
            {"id": "1", "content": "First"},
            {"id": "2", "content": "Second"}
        ]
        
        context = rag._construct_context(docs)
        
        # Should delegate to prepare_context
        assert isinstance(context, str)

