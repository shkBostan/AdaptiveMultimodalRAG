"""
Utility functions for search and retrieval operations.

Author: s Bostan
Created on: Nov, 2025
"""

import numpy as np
from typing import List, Dict, Callable


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    return float(np.linalg.norm(vec1 - vec2))


def filter_by_threshold(results: List[Dict], threshold: float, 
                       score_key: str = 'score') -> List[Dict]:
    """
    Filter search results by score threshold.
    
    Args:
        results: List of search results
        threshold: Minimum score threshold
        score_key: Key in result dict containing the score
        
    Returns:
        Filtered list of results
    """
    return [r for r in results if r.get(score_key, 0) >= threshold]


def rerank_results(results: List[Dict], reranker: Callable) -> List[Dict]:
    """
    Rerank search results using a custom reranker function.
    
    Args:
        results: List of search results
        reranker: Function that takes results and returns reranked results
        
    Returns:
        Reranked list of results
    """
    return reranker(results)

