"""
Generation evaluation metrics.

Author: s Bostan
Created on: Nov, 2025
"""

from typing import List, Optional


def bleu_score(reference: str, candidate: str, n: int = 4) -> float:
    """
    Calculate BLEU score for text generation.
    
    Args:
        reference: Reference text
        candidate: Generated text
        n: Maximum n-gram order (default: 4)
        
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        
        # BLEU score calculation
        score = sentence_bleu([ref_tokens], cand_tokens)
        return score
    except ImportError:
        # Fallback simple implementation
        ref_words = reference.lower().split()
        cand_words = candidate.lower().split()
        
        if len(cand_words) == 0:
            return 0.0
        
        # Simple unigram precision
        ref_set = set(ref_words)
        matches = sum(1 for word in cand_words if word in ref_set)
        return matches / len(cand_words)


def rouge_score(reference: str, candidate: str, rouge_type: str = 'rouge-l') -> float:
    """
    Calculate ROUGE score for text generation.
    
    Args:
        reference: Reference text
        candidate: Generated text
        rouge_type: Type of ROUGE metric ('rouge-l', 'rouge-1', 'rouge-2')
        
    Returns:
        ROUGE score
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores[rouge_type].fmeasure
    except ImportError:
        # Fallback simple F1-based implementation
        ref_words = set(reference.lower().split())
        cand_words = set(candidate.lower().split())
        
        if len(ref_words) == 0 or len(cand_words) == 0:
            return 0.0
        
        intersection = ref_words & cand_words
        precision = len(intersection) / len(cand_words) if cand_words else 0.0
        recall = len(intersection) / len(ref_words) if ref_words else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


def bert_score(reference: str, candidate: str, model_type: str = 'bert-base-uncased') -> float:
    """
    Calculate BERTScore for text generation.
    
    Args:
        reference: Reference text
        candidate: Generated text
        model_type: BERT model type to use
        
    Returns:
        BERTScore F1
    """
    try:
        from bert_score import score
        
        P, R, F1 = score([candidate], [reference], model_type=model_type, verbose=False)
        return F1.item()
    except ImportError:
        # Fallback to simple similarity
        return rouge_score(reference, candidate, 'rouge-l')

