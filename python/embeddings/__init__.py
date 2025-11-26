"""
Embeddings module for multimodal RAG.

Author: s Bostan
Created on: Nov, 2025
"""

from .word2vec_model import Word2VecModel
from .bert_embedding import BERTEmbedding
from .multimodal_fusion import MultimodalFusion, FusionStrategy

__all__ = ['Word2VecModel', 'BERTEmbedding', 'MultimodalFusion', 'FusionStrategy']

