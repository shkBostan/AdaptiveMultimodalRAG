"""
Retrieval engine for multimodal RAG system.

This module provides a cohesive retrieval pipeline integrating DocumentLoader
and IndexBuilder for end-to-end document indexing and search.

Author: s Bostan
Created on: Nov, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Callable
import numpy as np
import faiss

from .index_builder import IndexBuilder, IndexType, DistanceMetric
from ..logging import get_logger
from .document_loader import Document


class RetrievalEngine:
    """
    Retrieval engine for finding relevant documents.
    
    Integrates DocumentLoader and IndexBuilder to provide a complete retrieval pipeline:
    - Document ingestion (via DocumentLoader)
    - Embedding generation (via embedding function)
    - Index building (via IndexBuilder)
    - Search and retrieval
    
    Maintains backward compatibility with existing API while providing enhanced features.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = "L2",
        use_index_builder: bool = False,
        index_builder_type: IndexType = "flat_l2",
        nlist: int = 100
    ):
        """
        Initialize retrieval engine.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ("L2" or "cosine")
                Used for legacy mode (when use_index_builder=False)
            use_index_builder: Whether to use IndexBuilder (new mode) or legacy FAISS directly
            index_builder_type: Type of IndexBuilder index ("flat_l2" or "ivf_flat")
                Only used when use_index_builder=True
            nlist: Number of clusters for IVF index (only used for "ivf_flat")
        
        Note:
            Legacy mode (use_index_builder=False) maintains backward compatibility.
            New mode (use_index_builder=True) provides enhanced features with IndexBuilder.
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_index_builder = use_index_builder
        
        # Legacy mode: direct FAISS index
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []
        
        # New mode: IndexBuilder
        self.index_builder: Optional[IndexBuilder] = None
        if use_index_builder:
            distance_metric: DistanceMetric = "L2" if index_type == "L2" else "cosine"
            self.index_builder = IndexBuilder(
                embedding_dim=embedding_dim,
                index_type=index_builder_type,
                distance_metric=distance_metric,
                nlist=nlist
            )
        
        # Document storage (for new mode)
        self._document_objects: List[Document] = []
        
        # Logger
        self.logger = get_logger(__name__)
        
    def load_documents(self, documents: List[Document]) -> None:
        """
        Load and store Document objects internally.
        
        This method stores Document objects for use with the new indexing pipeline.
        Documents are used when building index with embedding function.
        
        Args:
            documents: List of Document objects from DocumentLoader
        
        Raises:
            ImportError: If Document class is not available
        """
        self.logger.info(
            f"Loading {len(documents)} documents",
            extra={"num_documents": len(documents)}
        )
        
        self._document_objects = documents
        
        self.logger.debug(
            "Documents loaded successfully",
            extra={
                "num_documents": len(documents),
                "sample_ids": [doc.id for doc in documents[:5]]
            }
        )
    
    def build_index(
        self,
        embeddings: Optional[np.ndarray] = None,
        documents: Optional[List[Dict]] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Build FAISS index from embeddings and documents.
        
        This method supports two modes:
        1. Legacy mode: Provide embeddings and documents directly
        2. New mode: Use embedding function with loaded Document objects
        
        Args:
            embeddings: Array of document embeddings (legacy mode)
            documents: List of document dictionaries (legacy mode)
            embedding_fn: Function that takes text and returns embedding vector (new mode)
                Signature: embedding_fn(text: str) -> np.ndarray
            save_path: Optional path to save index (only in new mode with IndexBuilder)
        
        Raises:
            ValueError: If parameters are invalid or mode is ambiguous
        """
        # Determine mode
        if embeddings is not None and documents is not None:
            # Legacy mode
            self._build_index_legacy(embeddings, documents)
        elif embedding_fn is not None:
            # New mode: use embedding function with Document objects
            self._build_index_with_embedding_fn(embedding_fn, save_path)
        else:
            raise ValueError(
                "Either provide (embeddings, documents) for legacy mode "
                "or embedding_fn for new mode"
            )
    
    def _build_index_legacy(self, embeddings: np.ndarray, documents: List[Dict]) -> None:
        """
        Build index using legacy method (direct FAISS).
        
        Args:
            embeddings: Array of document embeddings
            documents: List of document dictionaries
        """
        self.logger.info(
            "Building index (legacy mode)",
            extra={
                "num_documents": len(documents),
                "embedding_shape": embeddings.shape,
                "index_type": self.index_type
            }
        )
        
        if self.use_index_builder:
            # If IndexBuilder is enabled, use it even in legacy mode
            if self.index_builder is None:
                raise ValueError("IndexBuilder not initialized")
            
            self.index_builder.build(embeddings)
            self.documents = documents
            
            self.logger.info(
                "Index built successfully (legacy mode with IndexBuilder)",
                extra={"num_vectors": self.index_builder.num_vectors}
            )
        else:
            # Original legacy implementation
            if self.index_type == "L2":
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                # Cosine similarity
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings.astype('float32'))
            self.documents = documents
            
            self.logger.info(
                "Index built successfully (legacy mode)",
                extra={"num_vectors": len(documents)}
            )
    
    def _build_index_with_embedding_fn(
        self,
        embedding_fn: Callable[[str], np.ndarray],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Build index using embedding function with loaded Document objects.
        
        Args:
            embedding_fn: Function to generate embeddings from text
            save_path: Optional path to save index
        """
        if not self._document_objects:
            raise ValueError(
                "No documents loaded. Call load_documents() first or provide "
                "embeddings and documents directly."
            )
        
        if not self.use_index_builder or self.index_builder is None:
            raise ValueError(
                "IndexBuilder mode required. Initialize with use_index_builder=True"
            )
        
        self.logger.info(
            "Building index with embedding function",
            extra={
                "num_documents": len(self._document_objects),
                "embedding_dim": self.embedding_dim
            }
        )
        
        # Generate embeddings for all documents
        self.logger.debug("Generating embeddings for documents")
        embeddings_list = []
        
        for i, doc in enumerate(self._document_objects):
            if (i + 1) % 100 == 0:
                self.logger.debug(
                    f"Processed {i + 1}/{len(self._document_objects)} documents"
                )
            
            try:
                embedding = embedding_fn(doc.content)
                embeddings_list.append(embedding)
                
                # Store embedding in document object
                doc.embedding = embedding.tolist()
            except Exception as e:
                self.logger.error(
                    f"Failed to generate embedding for document {doc.id}",
                    extra={"document_id": doc.id, "error": str(e)},
                    exc_info=True
                )
                raise
        
        embeddings = np.array(embeddings_list)
        self.logger.info(
            "Embeddings generated",
            extra={"embedding_shape": embeddings.shape}
        )
        
        # Build index using IndexBuilder
        self.logger.debug("Building FAISS index")
        self.index_builder.build(embeddings)
        
        self.logger.info(
            "Index built successfully",
            extra={
                "num_vectors": self.index_builder.num_vectors,
                "index_type": self.index_builder.index_type,
                "is_trained": self.index_builder.is_trained
            }
        )
        
        # Save index if path provided
        if save_path:
            self.logger.info(f"Saving index to {save_path}")
            self.index_builder.save(save_path)
            self.logger.info("Index saved successfully")
    
    def load_index(self, filepath: Union[str, Path]) -> None:
        """
        Load FAISS index from disk using IndexBuilder.
        
        Args:
            filepath: Path to saved index file
        
        Raises:
            ValueError: If IndexBuilder mode not enabled
            FileNotFoundError: If index file not found
        """
        if not self.use_index_builder or self.index_builder is None:
            raise ValueError(
                "IndexBuilder mode required. Initialize with use_index_builder=True"
            )
        
        self.logger.info(f"Loading index from {filepath}")
        
        try:
            self.index_builder.load(filepath)
            
            self.logger.info(
                "Index loaded successfully",
                extra={
                    "num_vectors": self.index_builder.num_vectors,
                    "index_type": self.index_builder.index_type,
                    "is_trained": self.index_builder.is_trained
                }
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load index from {filepath}",
                extra={"filepath": str(filepath), "error": str(e)},
                exc_info=True
            )
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        return_documents: bool = False,
        nprobe: int = 1
    ) -> Union[List[Dict], List[Document]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector, shape [D] or [1, D]
            top_k: Number of top results to return
            return_documents: If True, return Document objects (new mode).
                             If False, return dictionaries (legacy mode)
            nprobe: Number of clusters to probe for IVF index (only for IndexBuilder mode)
        
        Returns:
            List of retrieved documents with scores.
            Returns Document objects if return_documents=True and documents are loaded.
            Returns dictionaries otherwise (legacy mode).
        
        Raises:
            ValueError: If index not built
        """
        self.logger.debug(
            "Searching for similar documents",
            extra={"top_k": top_k, "return_documents": return_documents}
        )
        
        if self.use_index_builder and self.index_builder is not None:
            return self._search_with_index_builder(query_embedding, top_k, return_documents, nprobe)
        else:
            return self._search_legacy(query_embedding, top_k)
    
    def _search_with_index_builder(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        return_documents: bool,
        nprobe: int
    ) -> Union[List[Dict], List[Document]]:
        """
        Search using IndexBuilder.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            return_documents: Whether to return Document objects
            nprobe: Number of clusters to probe
        
        Returns:
            List of results (Document objects or dictionaries)
        """
        if self.index_builder is None or self.index_builder.index is None:
            raise ValueError("Index not built. Call build_index() or load_index() first.")
        
        # Perform search using IndexBuilder
        scores, indices = self.index_builder.search(query_embedding, top_k, nprobe)
        
        self.logger.debug(
            "Search completed",
            extra={"num_results": len(scores), "top_score": float(scores[0]) if len(scores) > 0 else None}
        )
        
        # Build results
        if return_documents and self._document_objects:
            # Return Document objects
            results = []
            for i, (score, idx) in enumerate(zip(scores, indices)):
                if 0 <= idx < len(self._document_objects):
                    doc = self._document_objects[int(idx)]
                    # Create a copy-like result with score
                    # Import Document here to ensure it's available
                    from .document_loader import Document as DocClass
                    result_doc = DocClass(
                        id=doc.id,
                        content=doc.content,
                        metadata={**doc.metadata, 'score': float(score), 'rank': i + 1},
                        embedding=doc.embedding,
                        image_path=doc.image_path,
                        audio_path=doc.audio_path
                    )
                    results.append(result_doc)
            return results
        else:
            # Return dictionaries (legacy format)
            results = []
            if self.documents:
                doc_dicts = self.documents
            elif self._document_objects:
                doc_dicts = [doc.to_dict() for doc in self._document_objects]
            else:
                doc_dicts = []
            
            for i, (score, idx) in enumerate(zip(scores, indices)):
                if 0 <= idx < len(doc_dicts):
                    doc = doc_dicts[int(idx)].copy()
                    doc['score'] = float(score)
                    doc['rank'] = i + 1
                    results.append(doc)
            return results
    
    def _search_legacy(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """
        Search using legacy FAISS index.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
        
        Returns:
            List of document dictionaries
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        if self.index_type == "cosine":
            faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(dist)
                doc['rank'] = i + 1
                results.append(doc)
        
        return results
    
    def add_documents(
        self,
        embeddings: Optional[np.ndarray] = None,
        documents: Optional[List[Dict]] = None,
        document_objects: Optional[List[Document]] = None,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None
    ) -> None:
        """
        Add new documents to existing index (incremental update).
        
        Supports both legacy mode (embeddings + documents) and new mode
        (document_objects + embedding_fn).
        
        Args:
            embeddings: Array of new document embeddings (legacy mode)
            documents: List of new document dictionaries (legacy mode)
            document_objects: List of new Document objects (new mode)
            embedding_fn: Function to generate embeddings (new mode)
        
        Raises:
            ValueError: If parameters are invalid or mode is ambiguous
        """
        if embeddings is not None and documents is not None:
            # Legacy mode
            self._add_documents_legacy(embeddings, documents)
        elif document_objects is not None and embedding_fn is not None:
            # New mode
            self._add_documents_with_embedding_fn(document_objects, embedding_fn)
        else:
            raise ValueError(
                "Either provide (embeddings, documents) for legacy mode "
                "or (document_objects, embedding_fn) for new mode"
            )
    
    def _add_documents_legacy(self, embeddings: np.ndarray, documents: List[Dict]) -> None:
        """
        Add documents using legacy method.
        
        Args:
            embeddings: Array of new document embeddings
            documents: List of new document dictionaries
        """
        self.logger.info(
            "Adding documents (legacy mode)",
            extra={"num_documents": len(documents)}
        )
        
        if self.use_index_builder and self.index_builder is not None:
            self.index_builder.add_embeddings(embeddings)
            self.documents.extend(documents)
            self.logger.info("Documents added successfully (IndexBuilder mode)")
        else:
            if self.index is None:
                self.build_index(embeddings, documents)
            else:
                self.index.add(embeddings.astype('float32'))
                self.documents.extend(documents)
            self.logger.info("Documents added successfully (legacy mode)")
    
    def _add_documents_with_embedding_fn(
        self,
        document_objects: List[Document],
        embedding_fn: Callable[[str], np.ndarray]
    ) -> None:
        """
        Add documents using embedding function.
        
        Args:
            document_objects: List of new Document objects
            embedding_fn: Function to generate embeddings
        """
        if not self.use_index_builder or self.index_builder is None:
            raise ValueError(
                "IndexBuilder mode required. Initialize with use_index_builder=True"
            )
        
        self.logger.info(
            "Adding documents with embedding function",
            extra={"num_documents": len(document_objects)}
        )
        
        # Generate embeddings
        embeddings_list = []
        for doc in document_objects:
            embedding = embedding_fn(doc.content)
            embeddings_list.append(embedding)
            doc.embedding = embedding.tolist()
        
        embeddings = np.array(embeddings_list)
        
        # Add to index
        self.index_builder.add_embeddings(embeddings)
        self._document_objects.extend(document_objects)
        
        self.logger.info(
            "Documents added successfully",
            extra={"total_documents": len(self._document_objects)}
        )
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the current index and documents.
        
        Returns:
            Dictionary with statistics about index and documents
        """
        stats = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "use_index_builder": self.use_index_builder,
            "num_documents": len(self.documents) if self.documents else len(self._document_objects),
            "has_index": False
        }
        
        if self.use_index_builder and self.index_builder is not None:
            stats.update(self.index_builder.get_stats())
            stats["has_index"] = self.index_builder.index is not None
        else:
            stats["has_index"] = self.index is not None
            if self.index is not None:
                stats["num_vectors"] = self.index.ntotal
        
        return stats

