"""
Test suite for embedding models (BERT, CLIP, MultimodalFusion).

This module tests the core embedding components of the AdaptiveMultimodalRAG system:
- BERTEmbedding: Text embedding model with various pooling strategies
- CLIPImageEmbedding: Image embedding model using CLIP
- MultimodalFusion: Strategies for combining multimodal embeddings

Each component is tested independently with:
- Initialization and configuration
- Embedding generation (single and batch)
- Output shape and type validation
- Error handling and edge cases

Tests are designed to be:
- Fast: Use small models and minimal data
- Isolated: Each test function is independent
- Informative: Include logging and detailed assertions

Author: s Bostan
Created on: Dec, 2025
"""

import pytest
import numpy as np
import logging
from pathlib import Path
from PIL import Image

from src.embeddings import (
    BERTEmbedding,
    CLIPImageEmbedding,
    MultimodalFusion,
    PoolingStrategy,
    FusionStrategy
)


# ============================================================================
# BERT EMBEDDING TESTS
# ============================================================================

def test_bert_embedding_initialization(logger):
    """Test BERTEmbedding initialization with default and custom parameters."""
    logger.info("Testing BERTEmbedding initialization...")
    
    # Test default initialization
    model = BERTEmbedding()
    assert model.model_name == "bert-base-uncased"
    assert model.pooling_strategy == PoolingStrategy.CLS
    assert model.max_length == 512
    assert isinstance(model.embedding_dim, int)
    assert model.embedding_dim > 0 
    # assert model.embedding_dim is None  # Not loaded yet (lazy loading)
    logger.info(f"✓ Default BERTEmbedding initialized: {model.model_name}")
    
    # Test custom initialization
    model_custom = BERTEmbedding(
        model_name="distilbert-base-uncased",
        pooling_strategy=PoolingStrategy.MEAN,
        max_length=256,
        normalize_embeddings=True
    )
    assert model_custom.model_name == "distilbert-base-uncased"
    assert model_custom.pooling_strategy == PoolingStrategy.MEAN
    assert model_custom.max_length == 256
    assert model_custom.normalize_embeddings is True
    logger.info(f"✓ Custom BERTEmbedding initialized: {model_custom.model_name}")


def test_bert_embedding_single_text(logger, sample_texts):
    """Test BERTEmbedding with single text input."""
    logger.info("Testing BERTEmbedding single text embedding...")
    
    model = BERTEmbedding(
        model_name="bert-base-uncased",
        pooling_strategy="mean"
    )
    
    text = sample_texts[0]
    logger.info(f"  Input text: '{text[:50]}...'")
    
    embedding = model.get_embedding(text)
    
    # Validate output
    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert embedding.dtype == np.float32, "Embedding should be float32"
    assert embedding.ndim == 1, "Single embedding should be 1D"
    assert embedding.shape[0] > 0, "Embedding dimension should be positive"
    
    # BERT-base produces 768-dim embeddings
    assert embedding.shape[0] == 768, f"Expected 768-dim embedding, got {embedding.shape[0]}"
    
    logger.info(f"✓ Single embedding generated: shape={embedding.shape}, dtype={embedding.dtype}")
    logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    logger.info(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")


def test_bert_embedding_batch(logger, sample_texts):
    """Test BERTEmbedding with batch text input."""
    logger.info("Testing BERTEmbedding batch processing...")
    
    model = BERTEmbedding(
        model_name="bert-base-uncased",
        pooling_strategy="mean",
        batch_size=2
    )
    
    texts = sample_texts[:3]  # Use first 3 texts
    logger.info(f"  Processing batch of {len(texts)} texts")
    
    embeddings = model.get_embeddings_batch(texts)
    
    # Validate output
    assert isinstance(embeddings, np.ndarray), "Embeddings should be numpy array"
    assert embeddings.dtype == np.float32, "Embeddings should be float32"
    assert embeddings.ndim == 2, "Batch embeddings should be 2D"
    assert embeddings.shape[0] == len(texts), f"Batch size mismatch: {embeddings.shape[0]} != {len(texts)}"
    assert embeddings.shape[1] == 768, f"Expected 768-dim embeddings, got {embeddings.shape[1]}"
    
    logger.info(f"✓ Batch embeddings generated: shape={embeddings.shape}")
    logger.info(f"  Mean embedding norm: {np.mean([np.linalg.norm(e) for e in embeddings]):.4f}")


def test_bert_embedding_pooling_strategies(logger):
    """Test different pooling strategies for BERTEmbedding."""
    logger.info("Testing BERTEmbedding pooling strategies...")
    
    text = "This is a test sentence for pooling strategies."
    
    strategies_to_test = [
        PoolingStrategy.CLS,
        PoolingStrategy.MEAN,
        PoolingStrategy.MAX
    ]
    
    embeddings = {}
    for strategy in strategies_to_test:
        logger.info(f"  Testing strategy: {strategy.value}")
        model = BERTEmbedding(
            model_name="bert-base-uncased",
            pooling_strategy=strategy
        )
        embedding = model.get_embedding(text)
        embeddings[strategy.value] = embedding
        
        # All strategies should produce 768-dim embeddings (except MEAN_MAX)
        if strategy != PoolingStrategy.MEAN_MAX:
            assert embedding.shape[0] == 768, f"{strategy.value} should produce 768-dim embedding"
        logger.info(f"    Shape: {embedding.shape}, Norm: {np.linalg.norm(embedding):.4f}")
    
    # Verify different strategies produce different embeddings
    assert not np.allclose(embeddings['cls'], embeddings['mean']), "CLS and MEAN should differ"
    logger.info("✓ All pooling strategies tested successfully")


def test_bert_embedding_normalization(logger):
    """Test L2 normalization option for BERTEmbedding."""
    logger.info("Testing BERTEmbedding normalization...")
    
    text = "Test text for normalization"
    
    # Without normalization
    model_no_norm = BERTEmbedding(normalize_embeddings=False)
    embedding_no_norm = model_no_norm.get_embedding(text)
    norm_no_norm = np.linalg.norm(embedding_no_norm)
    
    # With normalization
    model_norm = BERTEmbedding(normalize_embeddings=True)
    embedding_norm = model_norm.get_embedding(text)
    norm_norm = np.linalg.norm(embedding_norm)
    
    # Normalized embedding should have norm ≈ 1.0
    assert abs(norm_norm - 1.0) < 0.01, f"Normalized embedding norm should be ≈1.0, got {norm_norm}"
    
    logger.info(f"✓ Normalization test: norm without={norm_no_norm:.4f}, with={norm_norm:.4f}")


# ============================================================================
# CLIP IMAGE EMBEDDING TESTS
# ============================================================================

def test_clip_embedding_initialization(logger):
    """Test CLIPImageEmbedding initialization."""
    logger.info("Testing CLIPImageEmbedding initialization...")
    
    model = CLIPImageEmbedding()
    assert model.model_name == "openai/clip-vit-base-patch32"
    assert isinstance(model.embedding_dim, int)
    assert model.embedding_dim > 0
    # assert model.embedding_dim is None  # Not loaded yet (lazy loading)
    logger.info(f"✓ CLIPImageEmbedding initialized: {model.model_name}")


def test_clip_embedding_from_path(logger, dummy_image_path):
    """Test CLIPImageEmbedding with image file path."""
    logger.info("Testing CLIPImageEmbedding from file path...")
    
    model = CLIPImageEmbedding(
        model_name="openai/clip-vit-base-patch32"
    )
    
    logger.info(f"  Image path: {dummy_image_path}")
    embedding = model.get_embedding(dummy_image_path)
    
    # Validate output
    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert embedding.dtype == np.float32, "Embedding should be float32"
    assert embedding.ndim == 1, "Single embedding should be 1D"
    
    # CLIP-base produces 512-dim embeddings
    assert embedding.shape[0] == 512, f"Expected 512-dim embedding, got {embedding.shape[0]}"
    
    logger.info(f"✓ Image embedding generated: shape={embedding.shape}, dtype={embedding.dtype}")
    logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")


def test_clip_embedding_from_pil(logger, dummy_pil_image):
    """Test CLIPImageEmbedding with PIL Image object."""
    logger.info("Testing CLIPImageEmbedding from PIL Image...")
    
    model = CLIPImageEmbedding()
    embedding = model.get_embedding(dummy_pil_image)
    
    assert isinstance(embedding, np.ndarray), "Embedding should be numpy array"
    assert embedding.shape[0] == 512, f"Expected 512-dim embedding, got {embedding.shape[0]}"
    
    logger.info(f"✓ PIL Image embedding generated: shape={embedding.shape}")


def test_clip_embedding_from_real_image(logger, real_image_path):
    logger.info("Testing CLIPImageEmbedding from real image...")
    model = CLIPImageEmbedding()
    logger.info(f"  Real image path: {real_image_path}")
    # Load image via PIL
    img = Image.open(real_image_path)
    
    embedding = model.get_embedding(img)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 512
    logger.info(f"  Embedding shape: {embedding.shape}")
    logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    logger.info(f"✓ Real image embedding generated: shape={embedding.shape}")


@pytest.mark.parametrize("image_color", ["red", "green", "blue"])
def test_clip_embedding_various_colors(logger, image_color):
    logger.info(f"Testing CLIPImageEmbedding with color: {image_color}")
    img = Image.new('RGB', (224, 224), color=image_color)
    model = CLIPImageEmbedding()
    embedding = model.get_embedding(img)
    
    assert embedding.shape[0] == 512
    logger.info(f"  Embedding shape: {embedding.shape}")
    logger.info(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    logger.info(f"✓ Various colors embedding generated: shape={embedding.shape}")


def test_multimodal_pipeline_with_real_images(logger, real_image_path):
    logger.info("Testing multimodal pipeline with real images...")
    docs = [
        {"text": "This is a test doc", "image_path": real_image_path}
    ]

    fusion_input = {}
    # Single document embedding
    fusion_input['text'] = BERTEmbedding().get_embedding(docs[0]['text']) 
    fusion_input['image'] = CLIPImageEmbedding().get_embedding(docs[0]['image_path']) 
    
    embeddings = MultimodalFusion().fuse(fusion_input)

    # embeddings is a single ndarray of shape (text_dim + image_dim)
    assert isinstance(embeddings, np.ndarray)
    expected_dim = fusion_input['text'].shape[0] + fusion_input['image'].shape[0]
    assert embeddings.shape[0] == expected_dim
    logger.info(f"  Fused embedding norm: {np.linalg.norm(embeddings):.4f}")
    logger.info(f"✓ Multimodal pipeline with real images successful: shape={embeddings.shape[0]}")


    



def test_clip_embedding_batch(logger, tmp_path):
    """Test CLIPImageEmbedding with batch of images."""
    logger.info("Testing CLIPImageEmbedding batch processing...")
    
    # Create multiple dummy images
    image_paths = []
    for i in range(3):
        img = Image.new('RGB', (224, 224), color=(i*80, 100, 150))
        path = tmp_path / f"test_image_{i}.jpg"
        img.save(path)
        image_paths.append(str(path))
    
    model = CLIPImageEmbedding()
    logger.info(f"  Processing batch of {len(image_paths)} images")
    
    embeddings = model.get_embeddings_batch(image_paths)
    
    # Validate output
    assert isinstance(embeddings, np.ndarray), "Embeddings should be numpy array"
    assert embeddings.ndim == 2, "Batch embeddings should be 2D"
    assert embeddings.shape[0] == len(image_paths), "Batch size mismatch"
    assert embeddings.shape[1] == 512, f"Expected 512-dim embeddings, got {embeddings.shape[1]}"
    
    logger.info(f"✓ Batch embeddings generated: shape={embeddings.shape}")


def test_clip_embedding_normalization(logger, dummy_image_path):
    """Test L2 normalization option for CLIPImageEmbedding."""
    logger.info("Testing CLIPImageEmbedding normalization...")
    
    # Without normalization
    model_no_norm = CLIPImageEmbedding(normalize_embeddings=False)
    embedding_no_norm = model_no_norm.get_embedding(dummy_image_path)
    norm_no_norm = np.linalg.norm(embedding_no_norm)
    
    # With normalization
    model_norm = CLIPImageEmbedding(normalize_embeddings=True)
    embedding_norm = model_norm.get_embedding(dummy_image_path)
    norm_norm = np.linalg.norm(embedding_norm)
    
    assert abs(norm_norm - 1.0) < 0.01, f"Normalized embedding norm should be ≈1.0, got {norm_norm}"
    
    logger.info(f"✓ Normalization test: norm without={norm_no_norm:.4f}, with={norm_norm:.4f}")


# ============================================================================
# MULTIMODAL FUSION TESTS
# ============================================================================

def test_multimodal_fusion_initialization(logger):
    """Test MultimodalFusion initialization with different strategies."""
    logger.info("Testing MultimodalFusion initialization...")
    
    # Test concatenation strategy
    fusion_concat = MultimodalFusion(strategy=FusionStrategy.CONCATENATION)
    assert fusion_concat.strategy == FusionStrategy.CONCATENATION
    logger.info(f"✓ Concatenation fusion initialized")
    
    # Test weighted sum strategy
    fusion_weighted = MultimodalFusion(
        strategy=FusionStrategy.WEIGHTED_SUM,
        weights={'text': 0.6, 'image': 0.4}
    )
    assert fusion_weighted.strategy == FusionStrategy.WEIGHTED_SUM
    assert fusion_weighted.weights is not None
    logger.info(f"✓ Weighted sum fusion initialized: {fusion_weighted.weights}")
    
    # Test attention strategy
    fusion_attention = MultimodalFusion(strategy=FusionStrategy.ATTENTION)
    assert fusion_attention.strategy == FusionStrategy.ATTENTION
    logger.info(f"✓ Attention fusion initialized")


def test_multimodal_fusion_concatenation(logger, sample_embeddings_dict):
    """Test concatenation fusion strategy."""
    logger.info("Testing MultimodalFusion concatenation strategy...")
    
    fusion = MultimodalFusion(strategy=FusionStrategy.CONCATENATION)
    
    text_emb = sample_embeddings_dict['text']
    image_emb = sample_embeddings_dict['image']
    
    logger.info(f"  Text embedding shape: {text_emb.shape}")
    logger.info(f"  Image embedding shape: {image_emb.shape}")
    
    fused = fusion.fuse(sample_embeddings_dict)
    
    # Validate output
    assert isinstance(fused, np.ndarray), "Fused embedding should be numpy array"
    assert fused.ndim == 1, "Single fused embedding should be 1D"
    
    # Concatenation: dimension should be sum of input dimensions
    expected_dim = text_emb.shape[0] + image_emb.shape[0]
    assert fused.shape[0] == expected_dim, f"Expected {expected_dim}-dim, got {fused.shape[0]}"
    
    logger.info(f"✓ Concatenation fusion successful: shape={fused.shape}")
    logger.info(f"  Expected dimension: {expected_dim}, Got: {fused.shape[0]}")


def test_multimodal_fusion_weighted_sum(logger):
    """Test weighted sum fusion strategy."""
    logger.info("Testing MultimodalFusion weighted sum strategy...")
    
    # Create embeddings with same dimension for weighted sum
    text_emb = np.random.rand(512).astype(np.float32)
    image_emb = np.random.rand(512).astype(np.float32)
    
    fusion = MultimodalFusion(
        strategy=FusionStrategy.WEIGHTED_SUM,
        weights={'text': 0.6, 'image': 0.4}
    )
    
    embeddings = {'text': text_emb, 'image': image_emb}
    fused = fusion.fuse(embeddings)
    
    # Validate output
    assert isinstance(fused, np.ndarray), "Fused embedding should be numpy array"
    assert fused.shape == text_emb.shape, "Weighted sum should preserve dimension"
    
    # Manual verification: weighted sum should be 0.6 * text + 0.4 * image
    expected = 0.6 * text_emb + 0.4 * image_emb
    assert np.allclose(fused, expected), "Weighted sum computation incorrect"
    
    logger.info(f"✓ Weighted sum fusion successful: shape={fused.shape}")


def test_multimodal_fusion_attention(logger):
    """Test attention fusion strategy."""
    logger.info("Testing MultimodalFusion attention strategy...")
    
    # Create embeddings with same dimension for attention
    text_emb = np.random.rand(512).astype(np.float32)
    image_emb = np.random.rand(512).astype(np.float32)
    
    fusion = MultimodalFusion(strategy=FusionStrategy.ATTENTION)
    
    embeddings = {'text': text_emb, 'image': image_emb}
    fused = fusion.fuse(embeddings)
    
    # Validate output
    assert isinstance(fused, np.ndarray), "Fused embedding should be numpy array"
    assert fused.shape == text_emb.shape, "Attention fusion should preserve dimension"
    
    logger.info(f"✓ Attention fusion successful: shape={fused.shape}")
    logger.info(f"  Fused embedding norm: {np.linalg.norm(fused):.4f}")


def test_multimodal_fusion_batch(logger, sample_embeddings_dict_batch):
    """Test multimodal fusion with batch inputs."""
    logger.info("Testing MultimodalFusion with batch inputs...")
    
    fusion = MultimodalFusion(strategy=FusionStrategy.CONCATENATION)
    
    text_batch = sample_embeddings_dict_batch['text']  # (3, 768)
    image_batch = sample_embeddings_dict_batch['image']  # (3, 512)
    
    logger.info(f"  Text batch shape: {text_batch.shape}")
    logger.info(f"  Image batch shape: {image_batch.shape}")
    
    fused_batch = fusion.fuse(sample_embeddings_dict_batch)
    
    # Validate output
    assert isinstance(fused_batch, np.ndarray), "Fused embeddings should be numpy array"
    assert fused_batch.ndim == 2, "Batch fused embeddings should be 2D"
    assert fused_batch.shape[0] == text_batch.shape[0], "Batch size should match"
    assert fused_batch.shape[1] == text_batch.shape[1] + image_batch.shape[1], "Dimension should be sum"
    
    logger.info(f"✓ Batch fusion successful: shape={fused_batch.shape}")


def test_multimodal_fusion_error_handling(logger):
    """Test error handling in MultimodalFusion."""
    logger.info("Testing MultimodalFusion error handling...")
    
    fusion = MultimodalFusion(strategy=FusionStrategy.WEIGHTED_SUM)
    
    # Test with mismatched dimensions (should raise error)
    text_emb = np.random.rand(768).astype(np.float32)
    image_emb = np.random.rand(512).astype(np.float32)  # Different dimension
    
    with pytest.raises(ValueError):
        fusion.fuse({'text': text_emb, 'image': image_emb})
    
    logger.info("✓ Error handling test passed: mismatched dimensions correctly detected")
    
    # Test with empty embeddings dict
    with pytest.raises(ValueError):
        fusion.fuse({})
    
    logger.info("✓ Error handling test passed: empty embeddings dict correctly detected")

