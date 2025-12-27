"""
FAISS-based index builder for efficient vector similarity search.

This module provides a complete indexing solution with support for multiple
FAISS index types, persistence, incremental updates, and validation.

Author: s Bostan
Created on: Nov, 2025
"""

from pathlib import Path
from typing import Tuple, Optional, Literal, Union
import numpy as np
import faiss


# Type aliases for clarity
IndexType = Literal["flat_l2", "ivf_flat"]
DistanceMetric = Literal["L2", "cosine"]


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
        
    Returns:
        Path object to the directory
        
    Raises:
        OSError: If directory cannot be created
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def validate_shape(embeddings: np.ndarray, expected_dim: Optional[int] = None) -> Tuple[int, int]:
    """
    Validate embedding array shape and return dimensions.
    
    Args:
        embeddings: Array of embeddings, shape [N, D]
        expected_dim: Expected embedding dimension (optional)
        
    Returns:
        Tuple of (num_vectors, embedding_dim)
        
    Raises:
        ValueError: If shape is invalid or dimension mismatch
    """
    if not isinstance(embeddings, np.ndarray):
        raise ValueError(f"Embeddings must be numpy array, got {type(embeddings)}")
    
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array [N, D], got shape {embeddings.shape}")
    
    num_vectors, embedding_dim = embeddings.shape
    
    if num_vectors == 0:
        raise ValueError("Embeddings array is empty (N=0)")
    
    if embedding_dim == 0:
        raise ValueError("Embedding dimension is zero (D=0)")
    
    if expected_dim is not None and embedding_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {embedding_dim}"
        )
    
    return num_vectors, embedding_dim


def validate_query(query_embedding: np.ndarray, expected_dim: int) -> np.ndarray:
    """
    Validate and prepare query embedding for search.
    
    Args:
        query_embedding: Query embedding vector, shape [D] or [1, D]
        expected_dim: Expected embedding dimension
        
    Returns:
        Reshaped query embedding as [1, D] float32 array
        
    Raises:
        ValueError: If shape or dimension is invalid
    """
    if not isinstance(query_embedding, np.ndarray):
        raise ValueError(f"Query embedding must be numpy array, got {type(query_embedding)}")
    
    # Reshape to 2D if needed
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    elif query_embedding.ndim == 2:
        if query_embedding.shape[0] != 1:
            raise ValueError(f"Query embedding must be single vector, got shape {query_embedding.shape}")
    else:
        raise ValueError(f"Query embedding must be 1D or 2D, got {query_embedding.ndim}D")
    
    actual_dim = query_embedding.shape[1]
    if actual_dim != expected_dim:
        raise ValueError(
            f"Query dimension mismatch: expected {expected_dim}, got {actual_dim}"
        )
    
    return query_embedding.astype('float32')


class IndexBuilder:
    """
    FAISS-based index builder for efficient vector similarity search.
    
    Supports multiple index types (Flat L2, IVF Flat) with persistence,
    incremental updates, and comprehensive validation.
    
    Attributes:
        embedding_dim: Dimension of embedding vectors
        index_type: Type of FAISS index ("flat_l2" or "ivf_flat")
        distance_metric: Distance metric ("L2" or "cosine")
        index: FAISS index object (None if not built)
        is_trained: Whether the index has been trained (for IVF indices)
        num_vectors: Number of vectors currently in the index
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: IndexType = "flat_l2",
        distance_metric: DistanceMetric = "L2",
        nlist: int = 100
    ):
        """
        Initialize index builder.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index
                - "flat_l2": Exact search with L2 distance (slower, exact)
                - "ivf_flat": Inverted file index (faster, approximate)
            distance_metric: Distance metric ("L2" or "cosine")
            nlist: Number of clusters for IVF index (only used for "ivf_flat")
                Higher values = better accuracy but slower training
        
        Raises:
            ValueError: If parameters are invalid
        """
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        
        if index_type not in ["flat_l2", "ivf_flat"]:
            raise ValueError(f"index_type must be 'flat_l2' or 'ivf_flat', got {index_type}")
        
        if distance_metric not in ["L2", "cosine"]:
            raise ValueError(f"distance_metric must be 'L2' or 'cosine', got {distance_metric}")
        
        if index_type == "ivf_flat" and nlist < 1:
            raise ValueError(f"nlist must be at least 1 for IVF index, got {nlist}")
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.distance_metric = distance_metric
        self.nlist = nlist
        
        # Index state
        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        self.num_vectors = 0
    
    def build(self, embeddings: np.ndarray) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings, shape [N, D]
                Must match embedding_dim specified in __init__
        
        Raises:
            ValueError: If embeddings are invalid or index already built
        """
        if self.index is not None:
            raise ValueError("Index already built. Use add_embeddings() for incremental updates or reset() first.")
        
        # Validate embeddings
        num_vectors, embedding_dim = validate_shape(embeddings, self.embedding_dim)
        
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity if needed
        if self.distance_metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Create index based on type
        if self.index_type == "flat_l2":
            if self.distance_metric == "L2":
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:  # cosine
                self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        elif self.index_type == "ivf_flat":
            # Create quantizer (flat index for clustering)
            if self.distance_metric == "L2":
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
            else:  # cosine
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                # For cosine, we normalize embeddings, so use IP quantizer
            
            # Create IVF index
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
            
            # Train IVF index (requires at least nlist vectors)
            if num_vectors >= self.nlist:
                self.index.train(embeddings)
                self.is_trained = True
            else:
                # If not enough vectors, fall back to flat index
                print(f"Warning: Only {num_vectors} vectors provided, need at least {self.nlist} for IVF training.")
                print("Falling back to flat index.")
                if self.distance_metric == "L2":
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                else:
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index_type = "flat_l2"  # Update type
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.num_vectors = num_vectors
    
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Add new embeddings to existing index (incremental update).
        
        Args:
            embeddings: Array of new embeddings, shape [N, D]
                Must match embedding_dim
        
        Raises:
            ValueError: If index not built, embeddings invalid, or IVF index not trained
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Validate embeddings
        num_new, embedding_dim = validate_shape(embeddings, self.embedding_dim)
        
        # Convert to float32
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity if needed
        if self.distance_metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Check if IVF index is trained
        if self.index_type == "ivf_flat":
            if not self.is_trained:
                raise ValueError(
                    "IVF index not trained. Train with build() using at least "
                    f"{self.nlist} vectors before adding embeddings."
                )
        
        # Add to index
        self.index.add(embeddings)
        self.num_vectors += num_new
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        nprobe: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector, shape [D] or [1, D]
            top_k: Number of top results to return
            nprobe: Number of clusters to probe (only for IVF index)
                Higher values = better accuracy but slower search
                Must be <= nlist
        
        Returns:
            Tuple of (scores, indices):
                - scores: Array of distances/similarities, shape [top_k]
                - indices: Array of vector indices, shape [top_k]
        
        Raises:
            ValueError: If index not built, query invalid, or parameters invalid
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        if top_k > self.num_vectors:
            top_k = self.num_vectors
            print(f"Warning: top_k reduced to {top_k} (number of vectors in index)")
        
        # Validate and prepare query
        query = validate_query(query_embedding, self.embedding_dim)
        
        # Normalize query for cosine similarity
        if self.distance_metric == "cosine":
            faiss.normalize_L2(query)
        
        # Set nprobe for IVF index
        if self.index_type == "ivf_flat":
            if nprobe < 1:
                raise ValueError(f"nprobe must be at least 1, got {nprobe}")
            if nprobe > self.nlist:
                raise ValueError(f"nprobe ({nprobe}) cannot exceed nlist ({self.nlist})")
            self.index.nprobe = nprobe
        
        # Perform search
        distances, indices = self.index.search(query, top_k)
        
        # Return first row (single query)
        return distances[0], indices[0]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save index to disk.
        
        Args:
            filepath: Path to save index file
        
        Raises:
            ValueError: If index not built
            OSError: If file cannot be written
        """
        if self.index is None:
            raise ValueError("Index not built. Cannot save empty index.")
        
        filepath = Path(filepath)
        
        # Ensure directory exists
        ensure_dir(filepath.parent)
        
        # Save index
        faiss.write_index(self.index, str(filepath))
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load index from disk.
        
        Args:
            filepath: Path to index file
        
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If loaded index dimension doesn't match
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        # Load index
        self.index = faiss.read_index(str(filepath))
        
        # Validate dimension
        loaded_dim = self.index.d
        if loaded_dim != self.embedding_dim:
            raise ValueError(
                f"Index dimension mismatch: expected {self.embedding_dim}, "
                f"got {loaded_dim} from file"
            )
        
        # Update state
        self.num_vectors = self.index.ntotal
        
        # Check if IVF index and if trained
        if isinstance(self.index, faiss.IndexIVF):
            self.index_type = "ivf_flat"
            self.is_trained = self.index.is_trained
            self.nlist = self.index.nlist
        else:
            self.index_type = "flat_l2"
            self.is_trained = True  # Flat indices are always "trained"
    
    def reset(self) -> None:
        """
        Reset index builder to initial state.
        
        Clears the index and resets all state variables.
        """
        self.index = None
        self.is_trained = False
        self.num_vectors = 0
    
    def get_stats(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics:
                - embedding_dim: Embedding dimension
                - index_type: Type of index
                - distance_metric: Distance metric
                - num_vectors: Number of vectors in index
                - is_trained: Whether index is trained
                - is_built: Whether index is built
        """
        return {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'distance_metric': self.distance_metric,
            'num_vectors': self.num_vectors,
            'is_trained': self.is_trained,
            'is_built': self.index is not None,
            'nlist': self.nlist if self.index_type == "ivf_flat" else None
        }
    
    def __repr__(self) -> str:
        """String representation of IndexBuilder."""
        status = "built" if self.index is not None else "not built"
        return (
            f"IndexBuilder(embedding_dim={self.embedding_dim}, "
            f"index_type={self.index_type}, distance_metric={self.distance_metric}, "
            f"num_vectors={self.num_vectors}, status={status})"
        )

