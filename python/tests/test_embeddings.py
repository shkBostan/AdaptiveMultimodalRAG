"""
Tests for embedding modules.

Author: s Bostan
Created on: Nov, 2025
"""

import pytest
import numpy as np
from embeddings.bert_embedding import BERTEmbedding
from embeddings.multimodal_fusion import MultimodalFusion, FusionStrategy


class TestBERTEmbedding:
    """Test cases for BERT embedding."""
    
    def test_embedding_shape(self):
        """Test that embeddings have correct shape."""
        model = BERTEmbedding()
        model.load_model()
        embedding = model.get_embedding("test text")
        assert embedding.shape == (768,)
    
    def test_batch_embeddings(self):
        """Test batch embedding generation."""
        model = BERTEmbedding()
        model.load_model()
        texts = ["text 1", "text 2", "text 3"]
        embeddings = model.get_embeddings_batch(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 768


class TestMultimodalFusion:
    """Test cases for multimodal fusion."""
    
    def test_concatenation_fusion(self):
        """Test concatenation fusion strategy."""
        fusion = MultimodalFusion(strategy=FusionStrategy.CONCATENATION)
        embeddings = {
            'text': np.array([1, 2, 3]),
            'image': np.array([4, 5, 6]),
            'audio': np.array([7, 8, 9])
        }
        fused = fusion.fuse(embeddings)
        assert len(fused) == 9
        assert np.array_equal(fused, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

