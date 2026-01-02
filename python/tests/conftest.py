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
from src.data.mscoco_dataset import MSCOCODataset

# ============================================================================
# LOGGER FIXTURE
# ============================================================================

@pytest.fixture(scope="session")
def logger():
    """
    Session-wide logger for test output.

    Provides a configured logger that outputs INFO level messages to the console.
    Useful for observing test progress and debugging.
    """
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# ============================================================================
# SAMPLE TEXT FIXTURES
# ============================================================================

@pytest.fixture
def sample_texts():
    """
    Provide sample text strings for embedding tests.

    Useful for testing text embeddings, retrieval, and text-based pipeline operations.
    """
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
        "Retrieval augmented generation combines search with language models."
    ]

# ============================================================================
# SAMPLE DOCUMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_documents():
    """
    Provide sample Document objects for pipeline tests.

    Each Document has id, content, and metadata fields.
    Useful for testing retrieval, embedding, and generation pipelines.
    """
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
    """
    Provide sample Document objects with attached image paths for multimodal tests.

    This fixture:
    - Creates a dummy image in memory
    - Saves it to a test path
    - Attaches the image path to each Document object
    - Cleans up the image after the test
    Useful for testing pipelines that combine text and image embeddings.
    """
    dummy_image = Image.new('RGB', (224, 224), color='red')
    image_path = Path("tests/data/dummy_test_image.jpg")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_image.save(image_path)
    
    docs_with_images = []
    for doc in sample_documents:
        doc_dict = doc.to_dict()
        doc_dict['image_path'] = str(image_path)
        docs_with_images.append(Document(**{k: v for k, v in doc_dict.items() if k != 'embedding'}))
    
    yield docs_with_images
    
    # Cleanup: remove dummy image after tests
    if image_path.exists():
        image_path.unlink()

# ============================================================================
# IMAGE FIXTURES
# ============================================================================

@pytest.fixture
def dummy_image_path(tmp_path):
    """
    Create a temporary dummy image file.

    Returns:
        str: Path to the generated dummy image.
    Useful for testing image processing without relying on actual dataset files.
    """
    image_path = tmp_path / "dummy_image.jpg"
    img = Image.new('RGB', (224, 224), color='blue')
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def dummy_pil_image():
    """
    Provide an in-memory PIL Image object.

    Useful for tests that require a PIL.Image but should not read/write files.
    """
    return Image.new('RGB', (224, 224), color='green')


@pytest.fixture
def real_image_path():
    """
    Provide path to an actual image in the test dataset.

    Returns:
        str: Path to a real image.
    This is useful for integration tests that require real image inputs.
    """
    path = Path("tests/data/image1.jpg")
    return str(path)

# ============================================================================
# EMBEDDING CONFIG FIXTURES
# ============================================================================

@pytest.fixture
def bert_config():
    """
    Return configuration dictionary for BERT embeddings.

    Contains model type, model name, pooling strategy, max sequence length, and batch size.
    Useful for initializing embedding pipelines in tests.
    """
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
    """
    Return configuration dictionary for CLIP embeddings.

    Useful for initializing image-text embedding pipelines in tests.
    """
    return {
        'embeddings': {
            'model_type': 'clip',
            'model_name': 'openai/clip-vit-base-patch32'
        }
    }


@pytest.fixture
def multimodal_config():
    """
    Return configuration dictionary for multimodal fusion embeddings.

    Combines BERT text embeddings with CLIP image embeddings using a concatenation fusion strategy.
    Useful for testing pipelines with multimodal inputs.
    """
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
    """
    Return configuration for full RAG (Retrieval-Augmented Generation) pipeline.

    Includes:
    - Text embedding model
    - Retrieval configuration (FAISS index, top-k results)
    - Generation model configuration
    Useful for end-to-end pipeline tests.
    """
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

# ============================================================================
# SAMPLE EMBEDDINGS FIXTURES
# ============================================================================

@pytest.fixture
def sample_embeddings_text():
    """
    Provide sample 768-dimensional text embeddings (BERT-like).

    Useful for testing embedding fusion, similarity computation, or other text-based operations.
    """
    np.random.seed(42)  # For reproducibility
    return np.random.rand(3, 768).astype(np.float32)


@pytest.fixture
def sample_embeddings_image():
    """
    Provide sample 512-dimensional image embeddings (CLIP-like).

    Useful for testing image embedding pipelines and multimodal fusion.
    """
    np.random.seed(43)  # For reproducibility
    return np.random.rand(3, 512).astype(np.float32)


@pytest.fixture
def sample_embeddings_dict(sample_embeddings_text, sample_embeddings_image):
    """
    Provide dictionary of single sample embeddings for fusion tests.

    Keys: 'text' and 'image'.
    Useful for testing concatenation or other fusion strategies on a single example.
    """
    return {
        'text': sample_embeddings_text[0],
        'image': sample_embeddings_image[0]
    }


@pytest.fixture
def sample_embeddings_dict_batch(sample_embeddings_text, sample_embeddings_image):
    """
    Provide batch embeddings dictionary for multimodal fusion tests.

    Each key contains a batch of embeddings (shape [3, dim]).
    Useful for testing vectorized fusion operations.
    """
    return {
        'text': sample_embeddings_text,
        'image': sample_embeddings_image
    }

# ============================================================================
# EMBEDDING MODEL FIXTURE
# ============================================================================

@pytest.fixture
def bert_embedding_model(bert_config, logger):
    """
    Initialize a BERT embedding model and return it.

    Loads model weights and generates one dummy embedding to ensure readiness.
    Useful for testing embedding pipelines without repeated initialization overhead.
    """
    from src.pipeline.rag_pipeline import run_embedding_pipeline
    
    model = run_embedding_pipeline(bert_config, logger)
    # Trigger model loading
    model.get_embedding("test")
    return model

# ============================================================================
# DATASET PATH FIXTURE
# ============================================================================

@pytest.fixture(scope="session")
def mscoco_root():
    """
    Return the root directory of processed MSCOCO dataset.

    Expected structure:
    experiments/datasets/mscoco/processed/

    Raises:
        FileNotFoundError: If dataset directory does not exist.
    """
    root = Path("experiments/datasets/mscoco/processed")
    if not root.exists():
        raise FileNotFoundError(f"MSCOCO processed dataset not found at: {root}")
    return root

# ============================================================================
# DATASET FIXTURE
# ============================================================================

@pytest.fixture(scope="session")
def mscoco_dataset(mscoco_root, logger):
    """
    Provide a session-wide MSCOCO dataset instance.

    Loads the dataset once per test session to speed up tests.
    Useful for any test requiring real dataset samples.
    """
    logger.info("Initializing MSCOCODataset (session fixture)")
    dataset = MSCOCODataset(
        root_dir=mscoco_root,
        split="train"
    )
    logger.info(f"✓ MSCOCODataset loaded ({len(dataset)} samples)")
    return dataset
