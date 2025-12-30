"""
Shared pytest fixtures for AdaptiveMultimodalRAG tests.

This module provides reusable fixtures for common test data, configurations,
and mock objects used across test modules.

Author: s Bostan
Created on: Dec, 2025
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging
from PIL import Image
import io

from src.retrieval import Document


@pytest.fixture
def logger():
    """Provide a configured logger for tests."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Add console handler
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    ))
    logger.addHandler(handler)
    
    return logger


@pytest.fixture
def sample_texts():
    """Provide sample text strings for embedding tests."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
        "Retrieval augmented generation combines search with language models."
    ]


@pytest.fixture
def sample_documents():
    """Provide sample Document objects for pipeline tests."""
    return [
        Document(
            id="doc1",
            content="Retrieval augmented generation (RAG) combines vector search with language models.",
            metadata={"source": "test", "topic": "RAG"}
        ),
        Document(
            id="doc2",
            content="BERT is a bidirectional transformer model for understanding language context.",
            metadata={"source": "test", "topic": "NLP"}
        ),
        Document(
            id="doc3",
            content="CLIP is a vision-language model that understands images and text together.",
            metadata={"source": "test", "topic": "multimodal"}
        )
    ]


@pytest.fixture
def sample_documents_with_images(sample_documents):
    """Provide sample Document objects with image paths for multimodal tests."""
    # Create a dummy image in memory for testing
    dummy_image = Image.new('RGB', (224, 224), color='red')
    image_path = Path("tests/data/dummy_test_image.jpg")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_image.save(image_path)
    
    # Add image paths to documents
    docs_with_images = []
    for doc in sample_documents:
        doc_dict = doc.to_dict()
        doc_dict['image_path'] = str(image_path)
        docs_with_images.append(Document(**{k: v for k, v in doc_dict.items() if k != 'embedding'}))
    
    yield docs_with_images
    
    # Cleanup: remove dummy image after tests
    if image_path.exists():
        image_path.unlink()


@pytest.fixture
def dummy_image_path(tmp_path):
    """Create a dummy image file for testing."""
    image_path = tmp_path / "dummy_image.jpg"
    img = Image.new('RGB', (224, 224), color='blue')
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def dummy_pil_image():
    """Provide a PIL Image object for testing."""
    return Image.new('RGB', (224, 224), color='green')


@pytest.fixture
def real_image_path():
    """Provide the path to a real image in your dataset."""
    path = Path("tests/data/image1.jpg")
    return str(path)


@pytest.fixture
def bert_config():
    """Provide BERT embedding configuration for tests."""
    return {
        'embeddings': {
            'model_type': 'bert',
            'model_name': 'bert-base-uncased',
            'pooling_strategy': 'mean',
            'max_length': 512,
            'batch_size': 32
        }
    }


@pytest.fixture
def clip_config():
    """Provide CLIP embedding configuration for tests."""
    return {
        'embeddings': {
            'model_type': 'clip',
            'model_name': 'openai/clip-vit-base-patch32'
        }
    }


@pytest.fixture
def multimodal_config():
    """Provide multimodal fusion configuration for tests."""
    return {
        'embeddings': {
            'fusion_strategy': 'concatenation',
            'text': {
                'model_type': 'bert',
                'model_name': 'bert-base-uncased'
            },
            'image': {
                'model_type': 'clip',
                'model_name': 'openai/clip-vit-base-patch32'
            },
            'batch_size': 16
        }
    }


@pytest.fixture
def full_rag_config():
    """Provide full RAG pipeline configuration for tests."""
    return {
        'embeddings': {
            'model_type': 'bert',
            'model_name': 'bert-base-uncased',
            'batch_size': 2
        },
        'retrieval': {
            'method': 'faiss',
            'similarity_metric': 'cosine',
            'top_k': 3
        },
        'generation': {
            'model_type': 'gpt2',
            'max_length': 64
        }
    }


@pytest.fixture
def sample_embeddings_text():
    """Provide sample text embeddings (768-dim BERT-like vectors)."""
    np.random.seed(42)  # For reproducibility
    return np.random.rand(3, 768).astype(np.float32)


@pytest.fixture
def sample_embeddings_image():
    """Provide sample image embeddings (512-dim CLIP-like vectors)."""
    np.random.seed(43)  # For reproducibility
    return np.random.rand(3, 512).astype(np.float32)


@pytest.fixture
def sample_embeddings_dict(sample_embeddings_text, sample_embeddings_image):
    """Provide dictionary of embeddings for fusion tests."""
    return {
        'text': sample_embeddings_text[0],  # Single sample
        'image': sample_embeddings_image[0]  # Single sample
    }


@pytest.fixture
def sample_embeddings_dict_batch(sample_embeddings_text, sample_embeddings_image):
    """Provide dictionary of batch embeddings for fusion tests."""
    return {
        'text': sample_embeddings_text,  # Batch (3, 768)
        'image': sample_embeddings_image  # Batch (3, 512)
    }


@pytest.fixture
def bert_embedding_model(bert_config, logger):
    """Fixture to provide initialized BERT embedding model with loaded weights."""
    from src.pipeline.rag_pipeline import run_embedding_pipeline
    
    model = run_embedding_pipeline(bert_config, logger)
    # Trigger model loading by generating one embedding
    model.get_embedding("test")
    return model

