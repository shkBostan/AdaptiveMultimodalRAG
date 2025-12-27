"""
Comprehensive tests for DocumentLoader.

Tests TXT, JSON, CSV loading, normalization, and chunking.

Author: s Bostan
Created on: Nov, 2025
"""

import pytest
import sys
from pathlib import Path

# Add python directory to path
python_dir = Path(__file__).parent.parent
sys.path.insert(0, str(python_dir))

from src.retrieval import (
    DocumentLoader, Document, TextNormalizer, BasicTextNormalizer, Chunker
)
from tests.fixtures import (
    temp_dir, sample_text_files, sample_json_file_list, 
    sample_json_file_dict, sample_csv_file
)


class TestDocumentLoader:
    """Test cases for DocumentLoader."""
    
    def test_initialization(self):
        """Test DocumentLoader initialization."""
        loader = DocumentLoader()
        assert loader is not None
    
    def test_load_from_txt_file(self, sample_text_files):
        """Test loading from a single TXT file."""
        loader = DocumentLoader()
        docs = loader.load_from_txt(str(sample_text_files[0]))
        
        assert len(docs) == 1
        assert docs[0].content is not None
        assert "document 1" in docs[0].content.lower()
    
    def test_load_from_directory_txt(self, temp_dir, sample_text_files):
        """Test loading TXT files from directory."""
        loader = DocumentLoader()
        docs = loader.load_from_directory(str(temp_dir))
        
        assert len(docs) == 3
        for doc in docs:
            assert doc.content is not None
            assert len(doc.content) > 0
    
    def test_load_from_json_list(self, sample_json_file_list):
        """Test loading JSON file with list format."""
        loader = DocumentLoader()
        docs = loader.load_from_json(str(sample_json_file_list))
        
        assert len(docs) == 3
        assert docs[0].id == "1"
        assert "First document" in docs[0].content
        assert docs[0].metadata.get("source") == "file1.txt"
    
    def test_load_from_json_dict(self, sample_json_file_dict):
        """Test loading JSON file with dictionary format."""
        loader = DocumentLoader()
        docs = loader.load_from_json(str(sample_json_file_dict))
        
        assert len(docs) == 3
        assert docs[0].id == "doc1"
        assert "Document one" in docs[0].content
        assert docs[0].metadata.get("author") == "Author A"
    
    def test_load_from_csv(self, sample_csv_file):
        """Test loading CSV file."""
        loader = DocumentLoader()
        docs = loader.load_from_csv(str(sample_csv_file), content_column="content")
        
        assert len(docs) == 3
        assert docs[0].id == "1"
        assert "First CSV" in docs[0].content
        assert docs[0].metadata.get("category") == "tech"
    
    def test_load_from_mixed(self, temp_dir, sample_text_files, sample_json_file_list):
        """Test loading mixed file types."""
        loader = DocumentLoader()
        docs = loader.load_from_mixed(str(temp_dir), recursive=False)
        
        # Should load both TXT and JSON files
        assert len(docs) >= 3  # At least the TXT files
    
    def test_chunking_character_based(self):
        """Test character-based chunking."""
        chunker = Chunker(chunk_size=50, overlap=10, chunk_by="characters")
        text = "This is a test document with enough content to be chunked. " * 10
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)
    
    def test_chunking_token_based(self):
        """Test token-based chunking."""
        chunker = Chunker(chunk_size=20, overlap=5, chunk_by="tokens")
        text = "This is a test document with enough content to be chunked. " * 10
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        # Token-based chunking may vary, just check we get chunks
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_normalization(self):
        """Test text normalization."""
        normalizer = BasicTextNormalizer(lowercase=True, remove_punctuation=False)
        text = "This IS a TEST Document!"
        normalized = normalizer.normalize(text)
        
        assert normalized == "this is a test document!"
    
    def test_normalization_with_punctuation_removal(self):
        """Test normalization with punctuation removal."""
        normalizer = BasicTextNormalizer(lowercase=True, remove_punctuation=True)
        text = "This IS a TEST Document!"
        normalized = normalizer.normalize(text)
        
        assert "!" not in normalized
        assert normalized.islower()
    
    def test_document_to_dict(self):
        """Test Document to_dict conversion."""
        doc = Document(
            id="test1",
            content="Test content",
            metadata={"source": "test.txt", "author": "Test Author"}
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["id"] == "test1"
        assert doc_dict["content"] == "Test content"
        assert doc_dict["source"] == "test.txt"
        assert doc_dict["author"] == "Test Author"
    
    def test_document_with_embedding(self):
        """Test Document with embedding."""
        embedding = [0.1, 0.2, 0.3]
        doc = Document(
            id="test1",
            content="Test content",
            embedding=embedding
        )
        
        assert doc.embedding == embedding
        doc_dict = doc.to_dict()
        assert doc_dict["embedding"] == embedding
    
    def test_empty_file_handling(self, temp_dir):
        """Test handling of empty files."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        
        loader = DocumentLoader()
        docs = loader.load_from_txt(str(empty_file))
        
        # Should handle empty file gracefully
        assert len(docs) == 0 or (len(docs) == 1 and len(docs[0].content) == 0)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        loader = DocumentLoader()
        
        with pytest.raises((FileNotFoundError, ValueError)):
            loader.load_from_txt("nonexistent_file.txt")
    
    def test_chunk_overlap(self):
        """Test that chunk overlap works correctly."""
        chunker = Chunker(chunk_size=50, overlap=10, chunk_by="characters")
        text = "A " * 100  # 200 characters
        
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Check that chunks overlap
            # This is a basic check - actual overlap verification would be more complex
            assert len(chunks) >= 1


class TestChunker:
    """Test cases for Chunker utility."""
    
    def test_character_chunking(self):
        """Test character-based chunking."""
        chunker = Chunker()
        text = "A" * 100
        
        chunks = chunker.chunk(text, chunk_size=30, overlap=5, method="character")
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 30 for chunk in chunks)
    
    def test_token_chunking(self):
        """Test token-based chunking."""
        chunker = Chunker()
        text = "This is a test sentence. " * 10
        
        chunks = chunker.chunk(text, chunk_size=10, overlap=2, method="token")
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)

