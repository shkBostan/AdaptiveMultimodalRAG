"""
Retrieval engine for multimodal RAG system.

Author: s Bostan
Created on: Nov, 2025
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import faiss


class RetrievalEngine:
    """Retrieval engine for finding relevant documents."""
    
    def __init__(self, embedding_dim: int = 768, index_type: str = "L2"):
        """
        Initialize retrieval engine.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ("L2" or "cosine")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.documents = []
        
    def build_index(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Build FAISS index from embeddings and documents.
        
        Args:
            embeddings: Array of document embeddings
            documents: List of document dictionaries with metadata
        """
        if self.index_type == "L2":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            # Cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        self.documents = documents
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with scores
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
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Add new documents to existing index.
        
        Args:
            embeddings: Array of new document embeddings
            documents: List of new document dictionaries
        """
        if self.index is None:
            self.build_index(embeddings, documents)
        else:
            self.index.add(embeddings.astype('float32'))
            self.documents.extend(documents)

