"""
Pipeline module for AdaptiveMultimodalRAG.

Contains core RAG pipeline logic and orchestration.
"""

from .rag_pipeline import (
    run_embedding_pipeline,
    run_full_embedding_pipeline,
    run_retrieval_pipeline,
    run_generation_pipeline,
    execute_full_rag_pipeline,
    execute_rag_pipeline  # Backward compatibility alias
)

__all__ = [
    'run_embedding_pipeline',
    'run_full_embedding_pipeline',
    'run_retrieval_pipeline',
    'run_generation_pipeline',
    'execute_full_rag_pipeline',
    'execute_rag_pipeline'  # Backward compatibility
]

