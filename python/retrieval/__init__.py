"""
Retrieval module for multimodal RAG.

Author: s Bostan
Created on: Nov, 2025
"""

from .retrieval_engine import RetrievalEngine
from .search_utils import cosine_similarity, euclidean_distance, filter_by_threshold, rerank_results

__all__ = ['RetrievalEngine', 'cosine_similarity', 'euclidean_distance', 
           'filter_by_threshold', 'rerank_results']

