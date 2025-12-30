"""
Word2Vec embedding model implementation.

This module provides Word2Vec-based embeddings with support for pre-trained models,
custom training, and multiple aggregation strategies for sentence/document-level
embeddings. Designed for research-grade applications with comprehensive error
handling and flexible configuration.

Author: s Bostan
Created on: Nov, 2025
"""

import numpy as np
from typing import List, Optional, Literal, Union, Tuple
from pathlib import Path
import logging
from enum import Enum

try:
    from gensim.models import Word2Vec, KeyedVectors  # type: ignore
    from gensim.models.word2vec import LineSentence  # type: ignore
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    Word2Vec = None  # type: ignore
    KeyedVectors = None  # type: ignore
    LineSentence = None  # type: ignore
    # Don't log warning here - it will be raised when class is instantiated

from ..utils.preprocessing import tokenize_text, preprocess_text


class AggregationStrategy(Enum):
    """
    Aggregation strategies for combining word embeddings into sentence/document embeddings.
    
    - MEAN: Simple average of word vectors (fast, good baseline)
    - WEIGHTED_MEAN: TF-IDF weighted average (better for longer documents)
    - MAX_POOLING: Element-wise maximum (captures strongest features)
    - MIN_POOLING: Element-wise minimum (captures weakest features)
    """
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    MAX_POOLING = "max_pooling"
    MIN_POOLING = "min_pooling"


class Word2VecModel:
    """
    Word2Vec embedding model for text processing.
    
    Supports both pre-trained models (from file or Gensim KeyedVectors) and
    training custom models. Provides multiple aggregation strategies for
    converting word-level embeddings to sentence/document-level embeddings.
    
    Example:
        >>> # Using pre-trained model
        >>> model = Word2VecModel(model_path="path/to/model.bin")
        >>> embedding = model.get_embedding("example text")
        
        >>> # Training new model
        >>> model = Word2VecModel(vector_size=300)
        >>> model.train(["sentence 1", "sentence 2"], save_path="model.bin")
        >>> embedding = model.get_embedding("new text")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vector_size: int = 300,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        sg: int = 0,
        aggregation_strategy: Union[AggregationStrategy, str] = AggregationStrategy.MEAN,
        handle_oov: bool = True,
        oov_vector: Optional[np.ndarray] = None,
        preprocess: bool = True,
        lowercase: bool = True
    ):
        """
        Initialize Word2Vec model.
        
        Args:
            model_path: Path to pre-trained Word2Vec model (.bin, .model, or KeyedVectors format)
                       If None, model must be trained before use
            vector_size: Dimensionality of word vectors (used for training new models)
            window: Maximum distance between current and predicted word (training)
            min_count: Minimum frequency threshold for words (training)
            workers: Number of worker threads (training)
            sg: Training algorithm: 0=CBOW, 1=Skip-gram (training)
            aggregation_strategy: Strategy for combining word embeddings into sentence embeddings
                                 Options: 'mean', 'weighted_mean', 'max_pooling', 'min_pooling'
            handle_oov: Whether to handle out-of-vocabulary words
                       If True, uses zero vector or oov_vector for unknown words
            oov_vector: Custom vector for OOV words (if None, uses zero vector)
            preprocess: Whether to preprocess input text (lowercase, normalize)
            lowercase: Convert text to lowercase during preprocessing
        
        Raises:
            ImportError: If gensim is not installed
            FileNotFoundError: If model_path is specified but file doesn't exist
        """
        if not GENSIM_AVAILABLE:
            raise ImportError(
                "Gensim library is required for Word2VecModel. "
                "Install it with: pip install gensim"
            )
        
        self.model_path = Path(model_path) if model_path else None
        self.model: Optional[Union[Word2Vec, KeyedVectors]] = None
        
        # Training parameters (used when training new models)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg  # 0=CBOW, 1=Skip-gram
        
        # Aggregation and preprocessing settings
        if isinstance(aggregation_strategy, str):
            try:
                self.aggregation_strategy = AggregationStrategy(aggregation_strategy.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown aggregation strategy: {aggregation_strategy}. "
                    f"Choose from: {[s.value for s in AggregationStrategy]}"
                )
        else:
            self.aggregation_strategy = aggregation_strategy
        
        self.handle_oov = handle_oov
        self.oov_vector = oov_vector
        self.preprocess_input = preprocess
        self.lowercase = lowercase
        
        # Lazy loading: only load model when needed
        self._model_loaded = False
        self._vocab_size: Optional[int] = None
        
        # Load model if path is provided
        if self.model_path and self.model_path.exists():
            self.load_model()
    
    def load_model(self) -> None:
        """
        Load pre-trained Word2Vec model from file.
        
        Supports multiple formats:
        - Binary Word2Vec format (.bin)
        - Gensim Word2Vec model (.model)
        - KeyedVectors format (.kv, .wv)
        - Text format (.txt)
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file format is not supported
        """
        if self._model_loaded and self.model is not None:
            return
        
        if self.model_path is None:
            raise ValueError(
                "No model path specified. Either provide model_path in __init__ "
                "or train a new model using train() method."
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        file_ext = self.model_path.suffix.lower()
        file_path_str = str(self.model_path)
        
        try:
            # Try loading as KeyedVectors (binary format)
            if file_ext in ['.bin', '.kv', '.wv']:
                self.model = KeyedVectors.load_word2vec_format(file_path_str, binary=True)
                logging.info(f"Loaded Word2Vec model from binary file: {self.model_path}")
            
            # Try loading as Gensim Word2Vec model
            elif file_ext == '.model':
                loaded = Word2Vec.load(file_path_str)
                # Extract KeyedVectors if it's a full Word2Vec model
                self.model = loaded.wv if hasattr(loaded, 'wv') else loaded
                logging.info(f"Loaded Word2Vec model from Gensim format: {self.model_path}")
            
            # Try loading as text format
            elif file_ext == '.txt':
                self.model = KeyedVectors.load_word2vec_format(file_path_str, binary=False)
                logging.info(f"Loaded Word2Vec model from text file: {self.model_path}")
            
            # Try generic loading (auto-detect format)
            else:
                try:
                    self.model = KeyedVectors.load_word2vec_format(file_path_str, binary=True)
                except (UnicodeDecodeError, ValueError):
                    try:
                        loaded = Word2Vec.load(file_path_str)
                        self.model = loaded.wv if hasattr(loaded, 'wv') else loaded
                    except Exception:
                        self.model = KeyedVectors.load_word2vec_format(file_path_str, binary=False)
                logging.info(f"Loaded Word2Vec model (auto-detected format): {self.model_path}")
            
            # Handle both old and new Gensim API
            if hasattr(self.model, 'key_to_index'):
                # Gensim 4.0+ uses key_to_index
                self._vocab_size = len(self.model.key_to_index)
            elif hasattr(self.model, 'vocab'):
                # Older Gensim versions use vocab dict
                self._vocab_size = len(self.model.vocab)
            else:
                # Fallback: try to get length from index
                self._vocab_size = len(self.model.index_to_key) if hasattr(self.model, 'index_to_key') else 0
            self.vector_size = self.model.vector_size
            self._model_loaded = True
            
        except Exception as e:
            raise ValueError(
                f"Failed to load Word2Vec model from {self.model_path}. "
                f"Error: {str(e)}. "
                f"Supported formats: .bin, .model, .txt, .kv, .wv"
            ) from e
    
    def train(
        self,
        sentences: Union[List[str], str, Path],
        save_path: Optional[Union[str, Path]] = None,
        epochs: int = 5,
        **kwargs
    ) -> None:
        """
        Train a new Word2Vec model on provided sentences.
        
        Args:
            sentences: Training data. Can be:
                      - List of strings (sentences)
                      - Path to text file (one sentence per line)
                      - Path object to text file
            save_path: Optional path to save trained model
            epochs: Number of training epochs (iterations)
            **kwargs: Additional training parameters passed to Word2Vec
        
        Raises:
            ValueError: If sentences is empty or invalid
        """
        # Prepare training data
        if isinstance(sentences, (str, Path)):
            sentences_path = Path(sentences)
            if not sentences_path.exists():
                raise FileNotFoundError(f"Training file not found: {sentences_path}")
            # Use LineSentence for memory-efficient reading
            training_data = LineSentence(str(sentences_path))
        elif isinstance(sentences, list):
            if not sentences:
                raise ValueError("Training sentences list cannot be empty")
            # Tokenize sentences
            training_data = [
                tokenize_text(preprocess_text(s, lower=self.lowercase) if self.preprocess_input else s)
                for s in sentences
            ]
        else:
            raise ValueError(f"Unsupported sentences type: {type(sentences)}")
        
        # Train model
        logging.info(f"Training Word2Vec model with {epochs} epochs...")
        self.model = Word2Vec(
            sentences=training_data,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=epochs,
            **kwargs
        )
        
        # Extract KeyedVectors for embedding lookup
        self.model = self.model.wv
        self._vocab_size = len(self.model.key_to_index)
        self._model_loaded = True
        
        logging.info(
            f"Training completed. Vocabulary size: {self._vocab_size}, "
            f"Vector size: {self.vector_size}"
        )
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save trained model to file.
        
        Args:
            save_path: Path to save the model (supports .bin, .model, .kv formats)
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_ext = save_path.suffix.lower()
        
        if file_ext in ['.bin', '.kv', '.wv']:
            self.model.save_word2vec_format(str(save_path), binary=True)
        elif file_ext == '.model':
            # For .model format, we need to wrap in Word2Vec
            # This is a limitation of Gensim - we'll save as binary instead
            logging.warning(
                ".model format requires full Word2Vec object. Saving as .bin instead."
            )
            save_path = save_path.with_suffix('.bin')
            self.model.save_word2vec_format(str(save_path), binary=True)
        else:
            self.model.save_word2vec_format(str(save_path), binary=True)
        
        logging.info(f"Model saved to: {save_path}")
    
    def _get_word_vectors(
        self,
        words: List[str]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get word vectors for a list of words, handling OOV words.
        
        Args:
            words: List of words to get vectors for
        
        Returns:
            Tuple of (word_vectors array, valid_indices list)
            word_vectors: Array of shape (n_valid_words, vector_size)
            valid_indices: Indices of words that were found in vocabulary
        """
        if not self._model_loaded:
            self.load_model()
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        word_vectors = []
        valid_indices = []
        
        for idx, word in enumerate(words):
            if word in self.model:
                word_vectors.append(self.model[word])
                valid_indices.append(idx)
            elif self.handle_oov:
                # Handle OOV word
                if self.oov_vector is not None:
                    word_vectors.append(self.oov_vector)
                    valid_indices.append(idx)
                # Otherwise skip (will result in valid_indices not containing this idx)
        
        if not word_vectors:
            # All words were OOV and handle_oov=False, return zero vector
            return np.zeros((1, self.vector_size)), []
        
        return np.array(word_vectors), valid_indices
    
    def _aggregate_vectors(
        self,
        word_vectors: np.ndarray,
        words: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Aggregate word vectors into a single sentence/document vector.
        
        Args:
            word_vectors: Array of word vectors, shape (n_words, vector_size)
            words: Optional list of words (used for weighted_mean strategy)
        
        Returns:
            Aggregated vector of shape (vector_size,)
        """
        if len(word_vectors) == 0:
            return np.zeros(self.vector_size)
        
        if self.aggregation_strategy == AggregationStrategy.MEAN:
            return np.mean(word_vectors, axis=0)
        
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_MEAN:
            # Simple IDF-like weighting (inverse document frequency approximation)
            # In practice, you might want to use actual IDF values
            if words is None or len(words) == 0:
                return np.mean(word_vectors, axis=0)
            
            # Use word frequencies as weights (simpler than full TF-IDF)
            # Higher frequency words get lower weight (more common = less informative)
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Inverse frequency weights
            max_count = max(word_counts.values()) if word_counts else 1
            weights = np.array([
                1.0 / (word_counts.get(word, 1) / max_count + 1e-6)
                for word in words[:len(word_vectors)]
            ])
            weights = weights / (np.sum(weights) + 1e-6)  # Normalize
            
            return np.average(word_vectors, axis=0, weights=weights)
        
        elif self.aggregation_strategy == AggregationStrategy.MAX_POOLING:
            return np.max(word_vectors, axis=0)
        
        elif self.aggregation_strategy == AggregationStrategy.MIN_POOLING:
            return np.min(word_vectors, axis=0)
        
        else:
            # Fallback to mean
            return np.mean(word_vectors, axis=0)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for input text.
        
        The method:
        1. Preprocesses the text (if enabled)
        2. Tokenizes into words
        3. Retrieves word vectors (handling OOV words)
        4. Aggregates word vectors using the specified strategy
        
        Args:
            text: Input text string
        
        Returns:
            Embedding vector of shape (vector_size,)
        
        Raises:
            ValueError: If model is not loaded/trained
        """
        if not self._model_loaded:
            self.load_model()
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        # Preprocess text if enabled
        if self.preprocess_input:
            text = preprocess_text(text, lower=self.lowercase)
        
        # Tokenize
        words = tokenize_text(text)
        
        if not words:
            # Empty text - return zero vector or OOV vector
            if self.handle_oov and self.oov_vector is not None:
                return self.oov_vector.copy()
            return np.zeros(self.vector_size)
        
        # Get word vectors
        word_vectors, valid_indices = self._get_word_vectors(words)
        
        # Aggregate vectors
        valid_words = [words[i] for i in valid_indices] if valid_indices else None
        embedding = self._aggregate_vectors(word_vectors, valid_words)
        
        return embedding.astype(np.float32)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input text strings
        
        Returns:
            Array of embedding vectors, shape (n_texts, vector_size)
        
        Raises:
            ValueError: If model is not loaded/trained
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    @property
    def vocab_size(self) -> Optional[int]:
        """Get vocabulary size of the loaded model."""
        if self._model_loaded and self.model is not None:
            return self._vocab_size
        return None
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        if self.model is not None and self._model_loaded:
            return self.vector_size
        return self.vector_size  # Return configured size if model not loaded
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size if self.vocab_size is not None else 0
