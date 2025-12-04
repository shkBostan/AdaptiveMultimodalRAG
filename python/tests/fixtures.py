"""
Shared fixtures and utilities for AdaptiveMultimodalRAG tests.

Author: s Bostan
Created on: Nov, 2025
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock
import json
import csv


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_text_files(temp_dir):
    """Create sample text files for testing."""
    files = []
    for i in range(3):
        file_path = temp_dir / f"document{i+1}.txt"
        file_path.write_text(f"This is document {i+1}. It contains some text content for testing purposes.")
        files.append(file_path)
    return files


@pytest.fixture
def sample_json_file_list(temp_dir):
    """Create sample JSON file with list format."""
    file_path = temp_dir / "documents_list.json"
    data = [
        {"id": "1", "content": "First document content", "source": "file1.txt"},
        {"id": "2", "content": "Second document content", "source": "file2.txt"},
        {"id": "3", "content": "Third document content", "source": "file3.txt"}
    ]
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def sample_json_file_dict(temp_dir):
    """Create sample JSON file with dictionary format."""
    file_path = temp_dir / "documents_dict.json"
    data = {
        "doc1": {"id": "doc1", "content": "Document one content", "author": "Author A"},
        "doc2": {"id": "doc2", "content": "Document two content", "author": "Author B"},
        "doc3": {"id": "doc3", "content": "Document three content", "author": "Author C"}
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create sample CSV file for testing."""
    file_path = temp_dir / "documents.csv"
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'content', 'category'])
        writer.writerow(['1', 'First CSV document', 'tech'])
        writer.writerow(['2', 'Second CSV document', 'science'])
        writer.writerow(['3', 'Third CSV document', 'tech'])
    return file_path


@pytest.fixture
def sample_embeddings():
    """Generate synthetic embeddings for testing."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(10, 128).astype('float32')


@pytest.fixture
def sample_query_embedding():
    """Generate synthetic query embedding."""
    np.random.seed(42)
    return np.random.rand(128).astype('float32')


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = Mock()
    model.get_embedding = Mock(side_effect=lambda text: np.random.rand(128).astype('float32'))
    model.get_embeddings_batch = Mock(side_effect=lambda texts: np.random.rand(len(texts), 128).astype('float32'))
    return model


@pytest.fixture
def sample_documents():
    """Create sample Document objects."""
    try:
        from data.document_loader import Document
        return [
            Document(id="1", content="First document", metadata={"source": "doc1.txt"}),
            Document(id="2", content="Second document", metadata={"source": "doc2.txt"}),
            Document(id="3", content="Third document", metadata={"source": "doc3.txt"})
        ]
    except ImportError:
        # Fallback to dictionaries
        return [
            {"id": "1", "content": "First document", "source": "doc1.txt"},
            {"id": "2", "content": "Second document", "source": "doc2.txt"},
            {"id": "3", "content": "Third document", "source": "doc3.txt"}
        ]


@pytest.fixture
def mock_faiss_index():
    """Create a mock FAISS index."""
    index = Mock()
    index.ntotal = 10
    index.d = 128
    index.search = Mock(return_value=(
        np.array([[0.1, 0.2, 0.3]]),  # distances
        np.array([[0, 1, 2]])  # indices
    ))
    return index


@pytest.fixture
def mock_rag_model():
    """Create a mock RAG generation model."""
    model = Mock()
    tokenizer = Mock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 1
    tokenizer.decode = Mock(return_value="This is a generated response.")
    tokenizer.return_value = {"input_ids": Mock()}
    
    model.tokenizer = tokenizer
    model.model = Mock()
    model.model.generate = Mock(return_value=Mock())
    
    return model


def create_sample_document_dicts(num: int = 5) -> List[Dict[str, Any]]:
    """Helper to create sample document dictionaries."""
    return [
        {
            "id": str(i),
            "content": f"Document {i} content with some text for testing.",
            "source": f"doc{i}.txt",
            "score": float(1.0 / (i + 1)),
            "rank": i + 1
        }
        for i in range(num)
    ]


def assert_embeddings_valid(embeddings: np.ndarray, expected_shape: tuple = None):
    """Assert embeddings are valid numpy arrays."""
    assert isinstance(embeddings, np.ndarray), "Embeddings must be numpy array"
    assert embeddings.ndim == 2, "Embeddings must be 2D"
    assert embeddings.dtype == np.float32, "Embeddings must be float32"
    if expected_shape:
        assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {embeddings.shape}"


def assert_documents_valid(documents: List, min_count: int = 1):
    """Assert documents list is valid."""
    assert isinstance(documents, list), "Documents must be a list"
    assert len(documents) >= min_count, f"Expected at least {min_count} documents"
    for doc in documents:
        assert doc is not None, "Document cannot be None"

