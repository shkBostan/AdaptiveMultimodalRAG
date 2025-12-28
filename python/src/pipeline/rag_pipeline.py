"""
Core RAG pipeline module for AdaptiveMultimodalRAG.

This module defines how RAG works in this project, containing the logical
execution order and pipeline stages (Embedding → Retrieval → Generation).

It is importable and unit-testable, with no experiment-specific logic,
filesystem paths, or result saving.

Author: s Bostan
Created on: Nov, 2025
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import logging

from ..embeddings import BERTEmbedding, MultimodalFusion
from ..retrieval import RetrievalEngine, Document
from ..generation import RAGModule


def run_embedding_pipeline(config: Dict[str, Any], logger: logging.Logger):
    """
    Run embedding generation pipeline.
    
    Initializes and returns the embedding model based on configuration.
    
    Args:
        config: Configuration dictionary containing 'embeddings' section
        logger: Logger instance for pipeline logging
        
    Returns:
        Embedding model instance (BERTEmbedding or MultimodalFusion)
    """
    logger.info("Starting embedding pipeline...")
    
    embedding_config = config.get('embeddings', {})
    
    if 'fusion_strategy' in embedding_config:
        # Multimodal fusion
        fusion = MultimodalFusion(
            strategy=embedding_config.get('fusion_strategy', 'weighted_average'),
            weights=embedding_config.get('fusion_weights', {})
        )
        logger.info("Initialized multimodal fusion")
        return fusion
    else:
        # Single modality (text)
        bert = BERTEmbedding(
            model_name=embedding_config.get('model_name', 'bert-base-uncased')
        )
        logger.info(f"Initialized BERT embedding model: {embedding_config.get('model_name', 'bert-base-uncased')}")
        return bert


def run_retrieval_pipeline(
    config: Dict[str, Any],
    embedding_model: Any,
    logger: logging.Logger
) -> RetrievalEngine:
    """
    Run retrieval pipeline.
    
    Initializes and returns the retrieval engine based on configuration.
    
    Args:
        config: Configuration dictionary containing 'retrieval' section
        embedding_model: Embedding model instance (used to determine embedding_dim)
        logger: Logger instance for pipeline logging
        
    Returns:
        Initialized RetrievalEngine instance
    """
    logger.info("Starting retrieval pipeline...")
    
    retrieval_config = config.get('retrieval', {})
    
    # RetrievalEngine parameters
    embedding_dim = 768  # Default BERT embedding dimension
    index_type = "L2" if retrieval_config.get('similarity_metric', 'cosine') == 'L2' else "cosine"
    
    engine = RetrievalEngine(
        embedding_dim=embedding_dim,
        index_type=index_type,
        use_index_builder=False  # Use legacy mode for simplicity
    )
    
    logger.info(f"Initialized retrieval engine with method: {retrieval_config.get('method', 'faiss')}, index_type: {index_type}")
    return engine


def run_generation_pipeline(
    config: Dict[str, Any],
    retrieval_engine: RetrievalEngine,
    logger: logging.Logger
) -> RAGModule:
    """
    Run generation pipeline.
    
    Initializes and returns the RAG module for generation.
    
    Args:
        config: Configuration dictionary containing 'generation' section
        retrieval_engine: Initialized RetrievalEngine instance
        logger: Logger instance for pipeline logging
        
    Returns:
        Initialized RAGModule instance
    """
    logger.info("Starting generation pipeline...")
    
    generation_config = config.get('generation', {})
    
    rag = RAGModule(
        model_name=generation_config.get('model_type', 'gpt2'),
        max_context_length=generation_config.get('max_length', 512)
    )
    
    logger.info(f"Initialized RAG module with model: {generation_config.get('model_type', 'gpt2')}")
    return rag


def execute_rag_pipeline(
    config: Dict[str, Any],
    documents: List[Document],
    logger: logging.Logger
) -> Tuple[Any, RetrievalEngine, RAGModule, np.ndarray]:
    """
    Execute the full RAG pipeline: Embedding → Retrieval → Generation.
    
    This is the core orchestration function that defines how RAG works in this project.
    It processes documents through the complete pipeline and returns all components.
    
    Args:
        config: Configuration dictionary with 'embeddings', 'retrieval', and 'generation' sections
        documents: List of Document objects to process
        logger: Logger instance for pipeline logging
        
    Returns:
        Tuple of (embedding_model, retrieval_engine, rag_module, embeddings_array)
        
    Raises:
        ValueError: If documents list is empty
        RuntimeError: If pipeline execution fails
    """
    if not documents:
        raise ValueError("Cannot execute RAG pipeline with empty documents list")
    
    # Step 1: Embedding
    embedding_model = run_embedding_pipeline(config, logger)
    
    # Step 2: Generate embeddings for documents
    logger.info("Generating document embeddings...")
    doc_embeddings = []
    for i, doc in enumerate(documents):
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{len(documents)} documents...")
        embedding = embedding_model.get_embedding(doc.content)
        doc_embeddings.append(embedding)
    
    embeddings_array = np.array(doc_embeddings)
    logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
    
    # Step 3: Retrieval
    retrieval_engine = run_retrieval_pipeline(config, embedding_model, logger)
    
    # Build index
    logger.info("Building retrieval index...")
    doc_dicts = [doc.to_dict() for doc in documents]
    retrieval_engine.build_index(embeddings_array, doc_dicts)
    logger.info("Index built successfully")
    
    # Step 4: Generation
    rag_module = run_generation_pipeline(config, retrieval_engine, logger)
    
    logger.info("RAG pipeline execution completed successfully")
    
    return embedding_model, retrieval_engine, rag_module, embeddings_array

