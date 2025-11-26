"""
Evaluation utilities for RAG system.

Author: s Bostan
Created on: Nov, 2025
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_retrieval_metrics(retrieved_docs: List[Dict], 
                               relevant_docs: List[str]) -> Dict[str, float]:
    """
    Calculate retrieval metrics (precision, recall, F1).
    
    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        
    Returns:
        Dictionary with metrics
    """
    retrieved_ids = [doc.get('id', doc.get('doc_id')) for doc in retrieved_docs]
    relevant_set = set(relevant_docs)
    retrieved_set = set(retrieved_ids)
    
    if len(retrieved_set) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Calculate metrics
    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
        
    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def evaluate_generation(generated: str, reference: str) -> Dict[str, float]:
    """
    Evaluate generated text against reference.
    
    Args:
        generated: Generated text
        reference: Reference text
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Simple word overlap metrics
    gen_words = set(generated.lower().split())
    ref_words = set(reference.lower().split())
    
    if len(ref_words) == 0:
        return {'word_overlap': 0.0, 'coverage': 0.0}
    
    overlap = len(gen_words & ref_words)
    word_overlap = overlap / len(ref_words) if len(ref_words) > 0 else 0.0
    coverage = overlap / len(gen_words) if len(gen_words) > 0 else 0.0
    
    return {
        'word_overlap': word_overlap,
        'coverage': coverage
    }

