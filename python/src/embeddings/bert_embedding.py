"""
BERT embedding model implementation.

This module provides BERT-based embeddings with support for various BERT model
variants, multiple pooling strategies, and flexible configuration options.
Designed for research-grade applications with comprehensive error handling
and production-ready features.

Author: s Bostan
Created on: Nov, 2025
"""

import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from typing import List, Optional, Literal, Union
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class PoolingStrategy(Enum):
    """
    Pooling strategies for BERT embeddings.
    
    - CLS: Use [CLS] token embedding (default for BERT, single vector)
    - MEAN: Mean pooling over all token embeddings (better for longer texts)
    - MAX: Max pooling over all token embeddings (captures strongest features)
    - MEAN_MAX: Concatenation of mean and max pooling (captures both statistics)
    - POOLER: Use BERT's pooler output (if available, trained on next sentence prediction)
    - FIRST_LAST_MEAN: Average of first and last hidden layers (captures shallow + deep features)
    """
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    MEAN_MAX = "mean_max"
    POOLER = "pooler"
    FIRST_LAST_MEAN = "first_last_mean"


class BERTEmbedding:
    """
    BERT-based embedding model for text processing.
    
    Supports various BERT model variants (bert-base-uncased, distilbert, roberta, etc.)
    and provides multiple pooling strategies for converting token-level embeddings
    to sentence/document-level embeddings.
    
    Features:
    - Lazy loading (models loaded on first use)
    - Multiple pooling strategies for different use cases
    - Batch processing support
    - GPU/CPU automatic device selection
    - Comprehensive error handling
    - Configurable tokenization parameters
    
    Example:
        >>> # Basic usage with default settings
        >>> model = BERTEmbedding(model_name="bert-base-uncased")
        >>> embedding = model.get_embedding("example text")
        
        >>> # Using mean pooling for longer documents
        >>> model = BERTEmbedding(
        ...     model_name="bert-base-uncased",
        ...     pooling_strategy=PoolingStrategy.MEAN
        ... )
        >>> embedding = model.get_embedding("long document text...")
        
        >>> # Batch processing
        >>> embeddings = model.get_embeddings_batch(["text1", "text2", "text3"])
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pooling_strategy: Union[PoolingStrategy, str] = PoolingStrategy.CLS,
        max_length: int = 512,
        batch_size: int = 32,
        device: Optional[Union[torch.device, str]] = None,
        use_fast_tokenizer: bool = True,
        normalize_embeddings: bool = False,
        trust_remote_code: bool = False
    ):
        """
        Initialize BERT embedding model.
        
        Args:
            model_name: Name or path of the BERT model to use.
                       Supports any HuggingFace model compatible with AutoModel:
                       - "bert-base-uncased", "bert-large-uncased"
                       - "distilbert-base-uncased"
                       - "roberta-base", "roberta-large"
                       - Custom model paths
            pooling_strategy: Strategy for pooling token embeddings into sentence embeddings.
                             Options: 'cls', 'mean', 'max', 'mean_max', 'pooler', 'first_last_mean'
            max_length: Maximum sequence length for tokenization (BERT limit: 512)
            batch_size: Batch size for batch processing (not used in single embedding calls)
            device: PyTorch device ('cuda', 'cpu', or torch.device). 
                   If None, automatically selects CUDA if available
            use_fast_tokenizer: Whether to use fast tokenizer (faster, requires tokenizers library)
            normalize_embeddings: Whether to L2-normalize embeddings (useful for cosine similarity)
            trust_remote_code: Whether to trust remote code when loading models from HuggingFace
        
        Raises:
            ImportError: If transformers or torch is not installed
            ValueError: If invalid pooling strategy is provided
        """
        self.model_name = model_name
        self.tokenizer: Optional[Union[AutoTokenizer, BertTokenizer]] = None
        self.model: Optional[Union[AutoModel, BertModel]] = None
        
        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Pooling strategy
        if isinstance(pooling_strategy, str):
            try:
                self.pooling_strategy = PoolingStrategy(pooling_strategy.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown pooling strategy: {pooling_strategy}. "
                    f"Choose from: {[s.value for s in PoolingStrategy]}"
                )
        else:
            self.pooling_strategy = pooling_strategy
        
        # Tokenization and processing parameters
        self.max_length = min(max_length, 512)  # BERT's hard limit
        self.batch_size = batch_size
        self.use_fast_tokenizer = use_fast_tokenizer
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        
        # Lazy loading tracking
        self._model_loaded = False
        self._embedding_dim: Optional[int] = None
        
        logger.info(
            f"Initialized BERTEmbedding with model={model_name}, "
            f"pooling={self.pooling_strategy.value}, device={self.device}"
        )
    
    def load_model(self) -> None:
        """
        Load BERT tokenizer and model.
        
        Uses AutoTokenizer and AutoModel for maximum compatibility with various
        BERT variants and custom models.
        
        Raises:
            OSError: If model cannot be loaded from HuggingFace Hub
            ValueError: If model configuration is invalid
        """
        if self._model_loaded and self.tokenizer is not None and self.model is not None:
            return
        
        try:
            logger.info(f"Loading BERT model: {self.model_name}")
            
            # Load tokenizer (try fast first, fallback to slow)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=self.use_fast_tokenizer,
                    trust_remote_code=self.trust_remote_code
                )
            except Exception as e:
                if self.use_fast_tokenizer:
                    logger.warning(
                        f"Fast tokenizer failed, falling back to slow tokenizer: {e}"
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        use_fast=False,
                        trust_remote_code=self.trust_remote_code
                    )
                else:
                    raise
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Determine embedding dimension
            config = self.model.config
            if hasattr(config, 'hidden_size'):
                self._embedding_dim = config.hidden_size
            elif hasattr(config, 'dim'):
                self._embedding_dim = config.dim
            else:
                # Fallback: infer from a dummy forward pass
                with torch.no_grad():
                    dummy_input = self.tokenizer(
                        "test", return_tensors="pt", padding=True
                    )
                    dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
                    dummy_output = self.model(**dummy_input)
                    if hasattr(dummy_output, 'last_hidden_state'):
                        self._embedding_dim = dummy_output.last_hidden_state.shape[-1]
                    else:
                        self._embedding_dim = 768  # Default BERT dimension
            
            # Adjust embedding dim for pooling strategies that concatenate
            if self.pooling_strategy == PoolingStrategy.MEAN_MAX:
                # Mean + Max concatenation doubles the dimension
                self._embedding_dim = self._embedding_dim * 2
            elif self.pooling_strategy == PoolingStrategy.FIRST_LAST_MEAN:
                # First + Last mean doesn't change dimension (it's a mean, not concat)
                pass  # Dimension stays the same
            
            self._model_loaded = True
            logger.info(
                f"Model loaded successfully. Embedding dimension: {self._embedding_dim}, "
                f"Device: {self.device}"
            )
            
        except Exception as e:
            raise OSError(
                f"Failed to load BERT model '{self.model_name}'. "
                f"Error: {str(e)}. "
                f"Ensure the model name is correct and you have internet access "
                f"to download from HuggingFace Hub."
            ) from e
    
    def _pool_embeddings(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooler_output: Optional[torch.Tensor] = None,
        hidden_states: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Pool token embeddings into sentence embeddings using the configured strategy.
        
        Args:
            last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            pooler_output: Optional pooler output (if available)
            hidden_states: Optional tuple of all hidden states (for first_last_mean)
        
        Returns:
            Pooled embeddings of shape (batch_size, embedding_dim)
        """
        batch_size = last_hidden_state.shape[0]
        
        if self.pooling_strategy == PoolingStrategy.CLS:
            # Use [CLS] token (first token)
            embeddings = last_hidden_state[:, 0, :]
        
        elif self.pooling_strategy == PoolingStrategy.MEAN:
            # Mean pooling over sequence (excluding padding)
            if attention_mask is not None:
                # Expand attention mask for broadcasting
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                # Sum embeddings, masked by attention
                sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                # Sum of attention mask (sequence lengths)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            else:
                embeddings = torch.mean(last_hidden_state, dim=1)
        
        elif self.pooling_strategy == PoolingStrategy.MAX:
            # Max pooling over sequence
            if attention_mask is not None:
                # Set padding tokens to very negative value before max
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                masked_embeddings = last_hidden_state * attention_mask_expanded + (
                    (1 - attention_mask_expanded) * -1e9
                )
                embeddings = torch.max(masked_embeddings, dim=1)[0]
            else:
                embeddings = torch.max(last_hidden_state, dim=1)[0]
        
        elif self.pooling_strategy == PoolingStrategy.MEAN_MAX:
            # Concatenate mean and max pooling
            if attention_mask is not None:
                attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                # Mean
                sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                mean_emb = sum_embeddings / sum_mask
                # Max
                masked_embeddings = last_hidden_state * attention_mask_expanded + (
                    (1 - attention_mask_expanded) * -1e9
                )
                max_emb = torch.max(masked_embeddings, dim=1)[0]
            else:
                mean_emb = torch.mean(last_hidden_state, dim=1)
                max_emb = torch.max(last_hidden_state, dim=1)[0]
            embeddings = torch.cat([mean_emb, max_emb], dim=1)
        
        elif self.pooling_strategy == PoolingStrategy.POOLER:
            # Use BERT's pooler output if available
            if pooler_output is not None:
                embeddings = pooler_output
            else:
                logger.warning(
                    "Pooler output not available, falling back to CLS token embedding"
                )
                embeddings = last_hidden_state[:, 0, :]
        
        elif self.pooling_strategy == PoolingStrategy.FIRST_LAST_MEAN:
            # Average of first and last hidden layers
            if hidden_states is not None and len(hidden_states) > 1:
                first_hidden = hidden_states[0]  # First layer
                last_hidden = hidden_states[-1]  # Last layer
                # Average the two layers
                combined = (first_hidden + last_hidden) / 2.0
                # Then use CLS token or mean pooling
                if attention_mask is not None:
                    attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                    sum_embeddings = torch.sum(combined * attention_mask_expanded, dim=1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = combined[:, 0, :]  # CLS token
            else:
                logger.warning(
                    "Hidden states not available, falling back to last hidden state mean"
                )
                if attention_mask is not None:
                    attention_mask_expanded = attention_mask.unsqueeze(-1).float()
                    sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = torch.mean(last_hidden_state, dim=1)
        
        else:
            # Fallback to CLS
            embeddings = last_hidden_state[:, 0, :]
        
        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for input text.
        
        Args:
            text: Input text string
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If embedding generation fails
        """
        if not self._model_loaded:
            self.load_model()
        
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                # Request hidden states if needed for first_last_mean pooling
                output_hidden_states = (
                    self.pooling_strategy == PoolingStrategy.FIRST_LAST_MEAN
                )
                
                outputs = self.model(**inputs, output_hidden_states=output_hidden_states)
                
                # Extract embeddings using pooling strategy
                embedding = self._pool_embeddings(
                    last_hidden_state=outputs.last_hidden_state,
                    attention_mask=inputs.get('attention_mask'),
                    pooler_output=getattr(outputs, 'pooler_output', None),
                    hidden_states=getattr(outputs, 'hidden_states', None)
                )
                
                # Convert to numpy (single sample, so squeeze batch dimension)
                embedding = embedding.cpu().numpy()[0]
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate embedding for text. Error: {str(e)}"
            ) from e
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input text strings
        
        Returns:
            Array of embedding vectors of shape (n_texts, embedding_dim)
        
        Raises:
            ValueError: If model is not loaded or texts list is empty
            RuntimeError: If embedding generation fails
        """
        if not texts:
            return np.array([])
        
        if not self._model_loaded:
            self.load_model()
        
        if self.tokenizer is None or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                # Request hidden states if needed
                output_hidden_states = (
                    self.pooling_strategy == PoolingStrategy.FIRST_LAST_MEAN
                )
                
                outputs = self.model(**inputs, output_hidden_states=output_hidden_states)
                
                # Extract embeddings using pooling strategy
                embeddings = self._pool_embeddings(
                    last_hidden_state=outputs.last_hidden_state,
                    attention_mask=inputs.get('attention_mask'),
                    pooler_output=getattr(outputs, 'pooler_output', None),
                    hidden_states=getattr(outputs, 'hidden_states', None)
                )
                
                # Convert to numpy
                embeddings = embeddings.cpu().numpy()
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate batch embeddings. Error: {str(e)}"
            ) from e
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        # If not loaded yet, try to infer from model config (estimate)
        # For BERT-base: 768, BERT-large: 1024
        if "large" in self.model_name.lower():
            return 1024
        elif "distil" in self.model_name.lower():
            return 768
        else:
            return 768  # Default for BERT-base
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"BERTEmbedding(model_name='{self.model_name}', "
            f"pooling='{self.pooling_strategy.value}', "
            f"device={self.device}, loaded={self._model_loaded})"
        )
