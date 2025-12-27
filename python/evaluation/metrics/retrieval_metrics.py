"""
Retrieval evaluation metrics.

Author: s Bostan
Created on: Nov, 2025
"""

import numpy as np
from typing import List, Dict, Any


def precision_at_k(relevant_items: List[int], retrieved_items: List[int], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        relevant_items: List of relevant item indices
        retrieved_items: List of retrieved item indices
        k: Number of top items to consider
        
    Returns:
        Precision@K score
    """
    if k == 0 or len(retrieved_items) == 0:
        return 0.0
    
    top_k = retrieved_items[:k]
    relevant_set = set(relevant_items)
    relevant_retrieved = sum(1 for item in top_k if item in relevant_set)
    
    return relevant_retrieved / k


def recall_at_k(relevant_items: List[int], retrieved_items: List[int], k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        relevant_items: List of relevant item indices
        retrieved_items: List of retrieved item indices
        k: Number of top items to consider
        
    Returns:
        Recall@K score
    """
    if len(relevant_items) == 0:
        return 0.0
    
    top_k = retrieved_items[:k]
    relevant_set = set(relevant_items)
    relevant_retrieved = sum(1 for item in top_k if item in relevant_set)
    
    return relevant_retrieved / len(relevant_items)


def mean_reciprocal_rank(relevant_items: List[List[int]], retrieved_items: List[List[int]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        relevant_items: List of lists, each containing relevant item indices for a query
        retrieved_items: List of lists, each containing retrieved item indices for a query
        
    Returns:
        MRR score
    """
    if len(relevant_items) == 0:
        return 0.0
    
    reciprocal_ranks = []
    for rel_items, ret_items in zip(relevant_items, retrieved_items):
        relevant_set = set(rel_items)
        for rank, item in enumerate(ret_items, start=1):
            if item in relevant_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def normalized_dcg(relevant_items: List[int], retrieved_items: List[int], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        relevant_items: List of relevant item indices
        retrieved_items: List of retrieved item indices
        k: Number of top items to consider (None for all)
        
    Returns:
        NDCG score
    """
    if k is None:
        k = len(retrieved_items)
    
    if len(relevant_items) == 0:
        return 0.0
    
    top_k = retrieved_items[:k]
    relevant_set = set(relevant_items)
    
    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 1)
    
    # Calculate IDCG (ideal DCG)
    ideal_relevant = sorted(relevant_items[:k], reverse=True)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(ideal_relevant), k) + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

