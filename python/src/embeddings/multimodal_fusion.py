"""
Multimodal fusion for combining text, image, and audio embeddings.

This module provides various strategies for fusing embeddings from different
modalities (text, image, audio) into unified representations suitable for
multimodal retrieval and generation tasks.

Design principles:
- Encoder-agnostic: Works with any embedding vectors, regardless of source encoder
- Deterministic: Same inputs always produce same outputs
- Research-oriented: Clear, interpretable fusion strategies suitable for publication
- Extensible: Easy to add new fusion strategies

Author: s Bostan
Created on: Nov, 2025
"""

import numpy as np
from typing import Dict, Optional, List, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """
    Fusion strategies for combining multimodal embeddings.
    
    - CONCATENATION: Concatenate all modality embeddings (deterministic order)
                     Result dimension = sum of all modality dimensions
                     Suitable when modalities complement each other
    
    - WEIGHTED_SUM: Weighted sum of embeddings (requires same dimension)
                    Result dimension = input dimension (all must match)
                    Suitable when modalities represent same semantic space
    
    - ATTENTION: Modality-level attention mechanism (learns to weight modalities)
                 Result dimension = input dimension (all must match)
                 Research-oriented, uses simple attention computation
                 Suitable for adaptive modality weighting
    """
    CONCATENATION = "concatenation"
    WEIGHTED_SUM = "weighted_sum"
    ATTENTION = "attention"


class MultimodalFusion:
    """
    Fusion module for combining embeddings from different modalities.
    
    This class is encoder-agnostic and operates purely on embedding vectors.
    It does not depend on specific embedding classes (BERT, CLIP, etc.),
    making it suitable for downstream fusion in multimodal RAG pipelines.
    
    Example:
        >>> # Concatenation fusion
        >>> fusion = MultimodalFusion(strategy=FusionStrategy.CONCATENATION)
        >>> embeddings = {
        ...     'text': np.array([0.1, 0.2, 0.3]),      # dim=3
        ...     'image': np.array([0.4, 0.5, 0.6])      # dim=3
        ... }
        >>> fused = fusion.fuse(embeddings)  # Result: dim=6
    
        >>> # Weighted sum fusion
        >>> fusion = MultimodalFusion(
        ...     strategy=FusionStrategy.WEIGHTED_SUM,
        ...     weights={'text': 0.6, 'image': 0.4}
        ... )
        >>> embeddings = {
        ...     'text': np.array([0.1, 0.2, 0.3]),      # dim=3
        ...     'image': np.array([0.4, 0.5, 0.6])      # dim=3
        ... }
        >>> fused = fusion.fuse(embeddings)  # Result: dim=3
    
        >>> # Attention fusion
        >>> fusion = MultimodalFusion(strategy=FusionStrategy.ATTENTION)
        >>> fused = fusion.fuse(embeddings)  # Adaptive weighting
    """
    
    # Deterministic modality order for concatenation
    MODALITY_ORDER = ['text', 'image', 'audio']
    
    def __init__(
        self,
        strategy: Union[FusionStrategy, str] = FusionStrategy.CONCATENATION,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multimodal fusion module.
        
        Args:
            strategy: Fusion strategy to use.
                     Options: 'concatenation', 'weighted_sum', 'attention'
            weights: Optional modality weights for WEIGHTED_SUM strategy.
                    Dictionary with modality names as keys and weights as values.
                    If None, uses equal weights.
                    Example: {'text': 0.6, 'image': 0.4}
                    Weights are automatically normalized to sum to 1.0
        
        Raises:
            ValueError: If invalid strategy is provided
        """
        # Parse strategy
        if isinstance(strategy, str):
            try:
                self.strategy = FusionStrategy(strategy.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown fusion strategy: {strategy}. "
                    f"Choose from: {[s.value for s in FusionStrategy]}"
                )
        else:
            self.strategy = strategy
        
        # Process weights
        self.weights = weights
        if self.weights is not None:
            # Normalize weights to sum to 1.0
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {k: v / total_weight for k, v in self.weights.items()}
            else:
                raise ValueError("Weights must sum to a positive value")
        
        logger.info(
            f"Initialized MultimodalFusion with strategy={self.strategy.value}, "
            f"weights={self.weights}"
        )
    
    def fuse(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Fuse embeddings from different modalities.
        
        Args:
            embeddings: Dictionary mapping modality names to embedding vectors.
                       Keys should be: 'text', 'image', 'audio' (or subset).
                       Values are 1D numpy arrays (for single samples) or
                       2D numpy arrays (for batches).
                       Example:
                       {
                           'text': np.array([0.1, 0.2, 0.3]),      # Single sample
                           'image': np.array([0.4, 0.5, 0.6])      # Single sample
                       }
                       Or for batches:
                       {
                           'text': np.array([[0.1, 0.2], [0.3, 0.4]]),    # (2, 2)
                           'image': np.array([[0.5, 0.6], [0.7, 0.8]])    # (2, 2)
                       }
        
        Returns:
            Fused embedding vector(s).
            - Single sample: 1D array of shape (fused_dim,)
            - Batch: 2D array of shape (batch_size, fused_dim)
        
        Raises:
            ValueError: If embeddings dict is empty, or dimensional compatibility fails
            NotImplementedError: For unsupported fusion strategies
        """
        if not embeddings:
            raise ValueError("Embeddings dictionary cannot be empty")
        
        # Filter out None values
        embeddings = {k: v for k, v in embeddings.items() if v is not None}
        if not embeddings:
            raise ValueError("All embeddings are None")
        
        # Route to appropriate fusion method
        if self.strategy == FusionStrategy.CONCATENATION:
            return self._concatenate(embeddings)
        elif self.strategy == FusionStrategy.WEIGHTED_SUM:
            return self._weighted_sum(embeddings)
        elif self.strategy == FusionStrategy.ATTENTION:
            return self._attention_fusion(embeddings)
        else:
            raise NotImplementedError(
                f"Fusion strategy '{self.strategy.value}' is not implemented. "
                f"Available strategies: {[s.value for s in FusionStrategy]}"
            )
    
    def _concatenate(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Concatenate embeddings from all modalities in deterministic order.
        
        Order: text → image → audio (as defined by MODALITY_ORDER)
        
        Args:
            embeddings: Dictionary of modality → embedding mappings
        
        Returns:
            Concatenated embedding vector(s)
        """
        vectors = []
        
        # Use deterministic order
        for modality in self.MODALITY_ORDER:
            if modality in embeddings:
                vectors.append(embeddings[modality])
        
        if not vectors:
            raise ValueError("No valid embeddings found in deterministic order")
        
        # Handle both single samples (1D) and batches (2D)
        # Check if inputs are 1D or 2D
        is_batch = any(v.ndim == 2 for v in vectors)
        
        if is_batch:
            # Batch mode: ensure all have same batch size
            batch_sizes = [v.shape[0] for v in vectors if v.ndim == 2]
            if len(set(batch_sizes)) > 1:
                raise ValueError(
                    f"Inconsistent batch sizes in concatenation: {batch_sizes}"
                )
            # Ensure all are 2D
            vectors_2d = [
                v if v.ndim == 2 else v.reshape(1, -1) for v in vectors
            ]
            result = np.concatenate(vectors_2d, axis=1)
            # If original was 1D, squeeze back
            if all(v.ndim == 1 for v in vectors):
                result = result[0]
        else:
            # Single sample mode: all should be 1D
            vectors_1d = [v.flatten() for v in vectors]
            result = np.concatenate(vectors_1d, axis=0)
        
        return result
    
    def _weighted_sum(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Weighted sum of embeddings (requires same dimension).
        
        Args:
            embeddings: Dictionary of modality → embedding mappings
        
        Returns:
            Weighted sum embedding vector(s)
        
        Raises:
            ValueError: If embeddings have incompatible dimensions,
                       or if weights are not provided and strategy requires them
        """
        if not embeddings:
            raise ValueError("Cannot compute weighted sum with empty embeddings")
        
        # Get embedding dimensions
        embedding_list = list(embeddings.values())
        dims = [v.shape[-1] for v in embedding_list]  # Last dimension
        
        # Check dimensional compatibility
        if len(set(dims)) > 1:
            dim_info = {k: v.shape for k, v in embeddings.items()}
            raise ValueError(
                f"All embeddings must have the same dimension for weighted_sum. "
                f"Got dimensions: {dim_info}"
            )
        
        embedding_dim = dims[0]
        
        # Get weights (default to equal weights if not provided)
        if self.weights is None:
            n_modalities = len(embeddings)
            weights = {k: 1.0 / n_modalities for k in embeddings.keys()}
        else:
            weights = self.weights
        
        # Validate weights cover all modalities
        missing_weights = set(embeddings.keys()) - set(weights.keys())
        if missing_weights:
            raise ValueError(
                f"Weights missing for modalities: {missing_weights}. "
                f"Required modalities: {list(embeddings.keys())}"
            )
        
        # Compute weighted sum
        result = None
        is_batch = any(v.ndim == 2 for v in embedding_list)
        
        for modality, embedding in embeddings.items():
            weight = weights.get(modality, 0.0)
            
            # Ensure correct shape
            if is_batch and embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif not is_batch and embedding.ndim == 2:
                embedding = embedding[0]
            
            weighted_embedding = embedding * weight
            
            if result is None:
                result = weighted_embedding
            else:
                result = result + weighted_embedding
        
        if result is None:
            raise ValueError("Failed to compute weighted sum")
        
        return result
    
    def _attention_fusion(self, embeddings: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Attention-based fusion of embeddings.
        
        Implements a simple modality-level attention mechanism that learns
        to weight different modalities based on their embedding magnitudes.
        This is a lightweight, inference-time fusion approach suitable for
        research applications without requiring training.
        
        Algorithm:
        1. Compute attention scores for each modality based on embedding norms
        2. Normalize scores using softmax
        3. Weighted sum of embeddings using attention weights
        
        Assumptions:
        - All embeddings must have the same dimension
        - Attention is computed based on L2 norm (magnitude) of embeddings
        - This is a simple heuristic; for learned attention, use trained models
        
        Args:
            embeddings: Dictionary of modality → embedding mappings
        
        Returns:
            Attention-weighted fusion embedding vector(s)
        
        Raises:
            ValueError: If embeddings have incompatible dimensions
        """
        if not embeddings:
            raise ValueError("Cannot compute attention fusion with empty embeddings")
        
        # Get embedding dimensions
        embedding_list = list(embeddings.values())
        dims = [v.shape[-1] for v in embedding_list]
        
        # Check dimensional compatibility
        if len(set(dims)) > 1:
            dim_info = {k: v.shape for k, v in embeddings.items()}
            raise ValueError(
                f"All embeddings must have the same dimension for attention fusion. "
                f"Got dimensions: {dim_info}"
            )
        
        # Compute attention scores based on embedding norms (L2 norm)
        # Higher norm → more "important" modality (heuristic)
        attention_scores = {}
        for modality, embedding in embeddings.items():
            # Handle both 1D and 2D arrays
            if embedding.ndim == 1:
                norm = np.linalg.norm(embedding)
            else:  # 2D batch
                # Average norm across batch (or use per-sample if needed)
                norm = np.mean([np.linalg.norm(embedding[i]) for i in range(embedding.shape[0])])
            attention_scores[modality] = norm
        
        # Convert to numpy array for softmax
        modalities = list(embeddings.keys())
        scores = np.array([attention_scores[m] for m in modalities])
        
        # Apply softmax to get attention weights
        # Add small epsilon to avoid numerical issues
        scores_exp = np.exp(scores - np.max(scores))  # Numerical stability
        attention_weights = scores_exp / (np.sum(scores_exp) + 1e-8)
        
        # Create weights dictionary
        attention_weights_dict = {
            modalities[i]: attention_weights[i] for i in range(len(modalities))
        }
        
        logger.debug(
            f"Attention fusion weights: {attention_weights_dict} "
            f"(computed from embedding norms)"
        )
        
        # Compute weighted sum directly (don't modify self.weights)
        result = None
        is_batch = any(v.ndim == 2 for v in embedding_list)
        
        for modality, embedding in embeddings.items():
            weight = attention_weights_dict[modality]
            
            # Ensure correct shape
            if is_batch and embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif not is_batch and embedding.ndim == 2:
                embedding = embedding[0]
            
            weighted_embedding = embedding * weight
            
            if result is None:
                result = weighted_embedding
            else:
                result = result + weighted_embedding
        
        if result is None:
            raise ValueError("Failed to compute attention fusion")
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the fusion module."""
        return (
            f"MultimodalFusion(strategy={self.strategy.value}, "
            f"weights={self.weights})"
        )
