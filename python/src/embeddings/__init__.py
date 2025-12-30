"""
Embeddings module for multimodal RAG.

This module provides embedding models for different modalities (text, image)
and fusion strategies for combining multimodal embeddings.

Text Embeddings:
- BERTEmbedding: BERT-based text embeddings with multiple pooling strategies
- Word2VecModel: Word2Vec-based text embeddings with aggregation strategies

Image Embeddings:
- CLIPImageEmbedding: CLIP-based image embeddings

Fusion:
- MultimodalFusion: Strategies for combining embeddings from different modalities

Author: s Bostan
Created on: Nov, 2025
"""

from .word2vec_model import Word2VecModel, AggregationStrategy
from .bert_embedding import BERTEmbedding, PoolingStrategy
from .clip_image_embedding import CLIPImageEmbedding
from .multimodal_fusion import MultimodalFusion, FusionStrategy

__all__ = [
    # Text embedding models
    'Word2VecModel',
    'BERTEmbedding',
    # Image embedding models
    'CLIPImageEmbedding',
    # Fusion
    'MultimodalFusion',
    # Strategy enums
    'FusionStrategy',
    'PoolingStrategy',
    'AggregationStrategy'
]

