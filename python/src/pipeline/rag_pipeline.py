"""
Core RAG pipeline module for AdaptiveMultimodalRAG.

This module defines the complete RAG pipeline architecture with support for
multimodal embeddings (text, image) and flexible fusion strategies. Designed
for research-grade applications with modular, testable components.

Pipeline Architecture:
    Embedding Stage → Retrieval Stage → Generation Stage
    
The embedding stage supports:
    - Single modality models (BERT, Word2Vec, CLIP)
    - Multimodal fusion (concatenation, weighted_sum, attention)
    - Batch processing for efficiency
    - Lazy loading of models

Author: s Bostan
Created on: Nov, 2025
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import logging

from ..embeddings import (
    BERTEmbedding,
    Word2VecModel,
    CLIPImageEmbedding,
    MultimodalFusion,
    PoolingStrategy,
    AggregationStrategy,
    FusionStrategy
)
from ..retrieval import RetrievalEngine, Document
from ..generation import RAGModule


logger = logging.getLogger(__name__)


def run_embedding_pipeline(
    config: Dict[str, Any],
    logger: logging.Logger
) -> Union[BERTEmbedding, Word2VecModel, CLIPImageEmbedding, MultimodalFusion]:
    """
    Initialize and return embedding model(s) based on configuration.
    
    Supports three modes:
    1. Single text model (BERT or Word2Vec)
    2. Single image model (CLIP)
    3. Multimodal fusion (combining text + image models)
    
    Args:
        config: Configuration dictionary containing 'embeddings' section.
                Example configs:
                
                Single text model:
                {
                    'embeddings': {
                        'model_type': 'bert',  # or 'word2vec'
                        'model_name': 'bert-base-uncased'
                    }
                }
                
                Multimodal fusion:
                {
                    'embeddings': {
                        'fusion_strategy': 'concatenation',
                        'text': {
                            'model_type': 'bert',
                            'model_name': 'bert-base-uncased'
                        },
                        'image': {
                            'model_type': 'clip',
                            'model_name': 'openai/clip-vit-base-patch32'
                        }
                    }
                }
        logger: Logger instance for pipeline logging
    
    Returns:
        Initialized embedding model or MultimodalFusion object.
        The returned object must have:
        - get_embedding() method (for single embeddings)
        - get_embeddings_batch() method (for batch embeddings)
        - embedding_dim property (for dimension queries)
    
    Raises:
        ValueError: If configuration is invalid or incomplete
        ImportError: If required dependencies are missing
    """
    logger.info("=" * 60)
    logger.info("Starting embedding pipeline initialization...")
    logger.info("=" * 60)
    
    embedding_config = config.get('embeddings', {})
    
    if not embedding_config:
        logger.warning("No embedding configuration found, using defaults")
        embedding_config = {}
    
    # Check if multimodal fusion is requested
    if 'fusion_strategy' in embedding_config:
        return _initialize_multimodal_fusion(embedding_config, logger)
    
    # Single modality model
    model_type = embedding_config.get('model_type', 'bert').lower()
    
    if model_type == 'bert':
        return _initialize_bert_model(embedding_config, logger)
    elif model_type == 'word2vec':
        return _initialize_word2vec_model(embedding_config, logger)
    elif model_type == 'clip':
        return _initialize_clip_model(embedding_config, logger)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'bert', 'word2vec', 'clip', or use 'fusion_strategy' for multimodal"
        )


def _initialize_bert_model(
    config: Dict[str, Any],
    logger: logging.Logger
) -> BERTEmbedding:
    """Initialize BERT embedding model from configuration."""
    model_name = config.get('model_name', 'bert-base-uncased')
    pooling_strategy = config.get('pooling_strategy', 'cls')
    max_length = config.get('max_length', 512)
    normalize = config.get('normalize_embeddings', False)
    
    logger.info(f"Initializing BERT model: {model_name}")
    logger.info(f"  Pooling strategy: {pooling_strategy}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Normalize embeddings: {normalize}")
    
    model = BERTEmbedding(
        model_name=model_name,
        pooling_strategy=pooling_strategy,
        max_length=max_length,
        normalize_embeddings=normalize
    )
    
    logger.info("BERT model initialized successfully (lazy loading enabled)")
    return model


def _initialize_word2vec_model(
    config: Dict[str, Any],
    logger: logging.Logger
) -> Word2VecModel:
    """Initialize Word2Vec embedding model from configuration."""
    model_path = config.get('model_path', None)
    vector_size = config.get('vector_size', 300)
    aggregation_strategy = config.get('aggregation_strategy', 'mean')
    
    logger.info(f"Initializing Word2Vec model")
    if model_path:
        logger.info(f"  Pre-trained model path: {model_path}")
    else:
        logger.info(f"  Vector size: {vector_size} (will require training)")
    logger.info(f"  Aggregation strategy: {aggregation_strategy}")
    
    model = Word2VecModel(
        model_path=model_path,
        vector_size=vector_size,
        aggregation_strategy=aggregation_strategy
    )
    
    logger.info("Word2Vec model initialized successfully")
    return model


def _initialize_clip_model(
    config: Dict[str, Any],
    logger: logging.Logger
) -> CLIPImageEmbedding:
    """Initialize CLIP image embedding model from configuration."""
    model_name = config.get('model_name', 'openai/clip-vit-base-patch32')
    normalize = config.get('normalize_embeddings', False)
    
    logger.info(f"Initializing CLIP model: {model_name}")
    logger.info(f"  Normalize embeddings: {normalize}")
    
    model = CLIPImageEmbedding(
        model_name=model_name,
        normalize_embeddings=normalize
    )
    
    logger.info("CLIP model initialized successfully (lazy loading enabled)")
    return model


def _initialize_multimodal_fusion(
    config: Dict[str, Any],
    logger: logging.Logger
) -> MultimodalFusion:
    """Initialize multimodal fusion with multiple embedding models."""
    fusion_strategy = config.get('fusion_strategy', 'concatenation')
    fusion_weights = config.get('fusion_weights', None)
    
    logger.info(f"Initializing multimodal fusion")
    logger.info(f"  Fusion strategy: {fusion_strategy}")
    if fusion_weights:
        logger.info(f"  Fusion weights: {fusion_weights}")
    
    # Note: MultimodalFusion operates on embeddings, not models
    # Individual models are initialized in run_full_embedding_pipeline
    fusion = MultimodalFusion(
        strategy=fusion_strategy,
        weights=fusion_weights
    )
    
    logger.info("Multimodal fusion initialized successfully")
    logger.info("Note: Individual embedding models will be initialized during embedding generation")
    return fusion


def run_full_embedding_pipeline(
    config: Dict[str, Any],
    documents: List[Document],
    logger: logging.Logger,
    batch_size: Optional[int] = None
) -> Tuple[
    Union[BERTEmbedding, Word2VecModel, CLIPImageEmbedding, MultimodalFusion],
    np.ndarray
]:
    """
    Generate embeddings for documents using configured embedding model(s).
    
    This function handles:
    - Single modality embeddings (text or image)
    - Multimodal embeddings with fusion (text + image)
    - Batch processing for efficiency
    - Proper handling of documents with optional image paths
    
    Args:
        config: Configuration dictionary with 'embeddings' section
        documents: List of Document objects to embed.
                  For multimodal fusion, documents should have:
                  - 'content' field (text)
                  - 'image_path' field (optional, for image embeddings)
        logger: Logger instance for pipeline logging
        batch_size: Optional batch size for embedding generation.
                   If None, uses config value or processes all at once.
                   Only applies to models that support batch processing.
    
    Returns:
        Tuple of (embedding_model_or_fusion, embeddings_array)
        - embedding_model_or_fusion: The initialized embedding model or fusion object
        - embeddings_array: NumPy array of shape (n_documents, embedding_dim)
    
    Raises:
        ValueError: If documents list is empty or configuration is invalid
        RuntimeError: If embedding generation fails
    """
    logger.info("=" * 60)
    logger.info("Starting full embedding pipeline...")
    logger.info(f"Processing {len(documents)} documents")
    logger.info("=" * 60)
    
    if not documents:
        raise ValueError("Cannot generate embeddings for empty documents list")
    
    embedding_config = config.get('embeddings', {})
    use_fusion = 'fusion_strategy' in embedding_config
    
    if use_fusion:
        return _generate_multimodal_embeddings(
            config, documents, logger, batch_size
        )
    else:
        return _generate_single_modality_embeddings(
            config, documents, logger, batch_size
        )


def _generate_single_modality_embeddings(
    config: Dict[str, Any],
    documents: List[Document],
    logger: logging.Logger,
    batch_size: Optional[int]
) -> Tuple[Union[BERTEmbedding, Word2VecModel, CLIPImageEmbedding], np.ndarray]:
    """Generate embeddings using a single modality model."""
    # Initialize model
    embedding_model = run_embedding_pipeline(config, logger)
    
    model_type = config.get('embeddings', {}).get('model_type', 'bert').lower()
    
    # Determine batch size
    if batch_size is None:
        batch_size = config.get('embeddings', {}).get('batch_size', 32)
    
    logger.info(f"Generating embeddings using {model_type} model")
    logger.info(f"Batch size: {batch_size}")
    
    embeddings_list = []
    
    if model_type == 'clip':
        # CLIP: process images
        image_paths = []
        for doc in documents:
            if doc.image_path:
                image_paths.append(doc.image_path)
            else:
                raise ValueError(
                    f"Document {doc.id} has no image_path. "
                    f"CLIP model requires image paths."
                )
        
        # Use batch processing if supported
        try:
            embeddings = embedding_model.get_embeddings_batch(image_paths)
            embeddings_list = list(embeddings)
        except Exception:
            # Fallback to single processing
            logger.warning("Batch processing failed, falling back to single processing")
            for img_path in image_paths:
                emb = embedding_model.get_embedding(img_path)
                embeddings_list.append(emb)
    
    else:
        # Text models: process text content
        texts = [doc.content for doc in documents]
        
        # Use batch processing if batch_size > 1 and model supports it
        if batch_size > 1 and hasattr(embedding_model, 'get_embeddings_batch'):
            logger.info(f"Processing in batches of {batch_size}...")
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embedding_model.get_embeddings_batch(batch_texts)
                embeddings_list.extend(batch_embeddings)
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} documents...")
        else:
            # Single processing
            for i, text in enumerate(texts):
                emb = embedding_model.get_embedding(text)
                embeddings_list.append(emb)
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(texts)} documents...")
    
    embeddings_array = np.array(embeddings_list)
    logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
    logger.info(f"Embedding dimension: {embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 'scalar'}")
    
    return embedding_model, embeddings_array


def _generate_multimodal_embeddings(
    config: Dict[str, Any],
    documents: List[Document],
    logger: logging.Logger,
    batch_size: Optional[int]
) -> Tuple[MultimodalFusion, np.ndarray]:
    """Generate embeddings using multimodal fusion."""
    embedding_config = config.get('embeddings', {})
    
    logger.info("Generating multimodal embeddings with fusion")
    
    # Initialize fusion object
    fusion = _initialize_multimodal_fusion(embedding_config, logger)
    
    # Initialize individual models
    text_config = embedding_config.get('text', {})
    image_config = embedding_config.get('image', {})
    
    text_model = None
    image_model = None
    
    if text_config:
        text_model_type = text_config.get('model_type', 'bert').lower()
        if text_model_type == 'bert':
            text_model = _initialize_bert_model(text_config, logger)
        elif text_model_type == 'word2vec':
            text_model = _initialize_word2vec_model(text_config, logger)
        else:
            raise ValueError(f"Unsupported text model type for fusion: {text_model_type}")
    
    if image_config:
        image_model_type = image_config.get('model_type', 'clip').lower()
        if image_model_type == 'clip':
            image_model = _initialize_clip_model(image_config, logger)
        else:
            raise ValueError(f"Unsupported image model type for fusion: {image_model_type}")
    
    # Determine batch size
    if batch_size is None:
        batch_size = embedding_config.get('batch_size', 32)
    
    logger.info(f"Batch size: {batch_size}")
    
    # Generate embeddings for each modality
    fused_embeddings_list = []
    
    n_docs = len(documents)
    for i in range(0, n_docs, batch_size):
        batch_docs = documents[i:i + batch_size]
        
        # Prepare modality embeddings dict
        modality_embeddings = {}
        
        # Text embeddings
        if text_model:
            texts = [doc.content for doc in batch_docs]
            if batch_size > 1 and hasattr(text_model, 'get_embeddings_batch'):
                text_embeddings = text_model.get_embeddings_batch(texts)
            else:
                text_embeddings = np.array([
                    text_model.get_embedding(text) for text in texts
                ])
            modality_embeddings['text'] = text_embeddings
        
        # Image embeddings
        if image_model:
            image_paths = []
            for doc in batch_docs:
                if doc.image_path:
                    image_paths.append(doc.image_path)
                else:
                    raise ValueError(
                        f"Document {doc.id} has no image_path. "
                        f"Multimodal fusion with image model requires image_path."
                    )
            
            if batch_size > 1:
                image_embeddings = image_model.get_embeddings_batch(image_paths)
            else:
                image_embeddings = np.array([
                    image_model.get_embedding(img_path) for img_path in image_paths
                ])
            modality_embeddings['image'] = image_embeddings
        
        # Fuse embeddings
        if len(batch_docs) == 1:
            # Single document: create dict with 1D arrays
            fused_dict = {
                mod: emb[0] if emb.ndim > 1 else emb
                for mod, emb in modality_embeddings.items()
            }
            fused_emb = fusion.fuse(fused_dict)
            fused_embeddings_list.append(fused_emb)
        else:
            # Batch: fuse each document separately
            for j in range(len(batch_docs)):
                fused_dict = {
                    mod: emb[j] for mod, emb in modality_embeddings.items()
                }
                fused_emb = fusion.fuse(fused_dict)
                fused_embeddings_list.append(fused_emb)
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {min(i + batch_size, n_docs)}/{n_docs} documents...")
    
    embeddings_array = np.array(fused_embeddings_list)
    logger.info(f"Generated fused embeddings shape: {embeddings_array.shape}")
    logger.info(f"Fused embedding dimension: {embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 'scalar'}")
    
    # Store models in fusion object for later use (hack for compatibility)
    # In practice, fusion is stateless and models are separate
    return fusion, embeddings_array


def run_retrieval_pipeline(
    config: Dict[str, Any],
    embedding_model_or_fusion: Union[
        BERTEmbedding, Word2VecModel, CLIPImageEmbedding, MultimodalFusion
    ],
    logger: logging.Logger
) -> RetrievalEngine:
    """
    Initialize and return retrieval engine based on configuration.
    
    The retrieval engine's embedding dimension is automatically determined
    from the embedding model or fusion object.
    
    Args:
        config: Configuration dictionary containing 'retrieval' section.
                Example:
                {
                    'retrieval': {
                        'method': 'faiss',
                        'similarity_metric': 'cosine',  # or 'L2'
                        'index_type': 'flat'
                    }
                }
        embedding_model_or_fusion: Initialized embedding model or MultimodalFusion.
                                  Used to determine embedding dimension.
        logger: Logger instance for pipeline logging
    
    Returns:
        Initialized RetrievalEngine instance
    
    Raises:
        ValueError: If embedding dimension cannot be determined
        AttributeError: If embedding model doesn't have embedding_dim property
    """
    logger.info("=" * 60)
    logger.info("Starting retrieval pipeline initialization...")
    logger.info("=" * 60)
    
    retrieval_config = config.get('retrieval', {})
    
    # Determine embedding dimension
    if hasattr(embedding_model_or_fusion, 'embedding_dim'):
        embedding_dim = embedding_model_or_fusion.embedding_dim
        logger.info(f"Embedding dimension from model: {embedding_dim}")
    elif isinstance(embedding_model_or_fusion, MultimodalFusion):
        # For fusion, we need to infer from strategy
        # This is a limitation - fusion dimension depends on input embeddings
        # We'll use a default and let build_index handle actual dimension
        logger.warning(
            "Cannot determine embedding dimension from MultimodalFusion. "
            "Using default 768. Dimension will be validated during index building."
        )
        embedding_dim = 768  # Will be validated later
    else:
        raise ValueError(
            f"Embedding model {type(embedding_model_or_fusion)} does not have "
            f"embedding_dim property. Cannot initialize retrieval engine."
        )
    
    # Get retrieval parameters
    similarity_metric = retrieval_config.get('similarity_metric', 'cosine').lower()
    index_type = "L2" if similarity_metric == 'l2' else "cosine"
    method = retrieval_config.get('method', 'faiss')
    
    logger.info(f"Retrieval method: {method}")
    logger.info(f"Similarity metric: {similarity_metric} ({index_type} index)")
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    engine = RetrievalEngine(
        embedding_dim=embedding_dim,
        index_type=index_type,
        use_index_builder=False  # Use legacy mode for simplicity
    )
    
    logger.info("Retrieval engine initialized successfully")
    return engine


def run_generation_pipeline(
    config: Dict[str, Any],
    retrieval_engine: RetrievalEngine,
    logger: logging.Logger
) -> RAGModule:
    """
    Initialize and return RAG generation module based on configuration.
    
    Args:
        config: Configuration dictionary containing 'generation' section.
                Example:
                {
                    'generation': {
                        'model_type': 'gpt2',
                        'max_length': 512,
                        'temperature': 0.7
                    }
                }
        retrieval_engine: Initialized RetrievalEngine instance.
                         Note: RAGModule uses retrieval_engine in generate()
                         method, not in initialization.
        logger: Logger instance for pipeline logging
    
    Returns:
        Initialized RAGModule instance
    
    Raises:
        ValueError: If configuration is invalid
    """
    logger.info("=" * 60)
    logger.info("Starting generation pipeline initialization...")
    logger.info("=" * 60)
    
    generation_config = config.get('generation', {})
    
    model_type = generation_config.get('model_type', 'gpt2')
    max_context_length = generation_config.get('max_length', 512)
    max_length = generation_config.get('max_length', 512)
    
    logger.info(f"RAG model type: {model_type}")
    logger.info(f"Max context length: {max_context_length}")
    logger.info(f"Max generation length: {max_length}")
    
    rag_module = RAGModule(
        model_name=model_type,
        max_context_length=max_context_length
    )
    
    logger.info("RAG module initialized successfully (lazy loading enabled)")
    logger.info("Note: Model will be loaded on first generate() call")
    
    return rag_module


def execute_full_rag_pipeline(
    config: Dict[str, Any],
    documents: List[Document],
    logger: logging.Logger,
    batch_size: Optional[int] = None
) -> Tuple[
    Union[BERTEmbedding, Word2VecModel, CLIPImageEmbedding, MultimodalFusion],
    RetrievalEngine,
    RAGModule,
    np.ndarray
]:
    """
    Execute the complete RAG pipeline: Embedding → Retrieval → Generation.
    
    This is the core orchestration function that defines how RAG works in this project.
    It processes documents through all pipeline stages and returns all initialized
    components along with the generated embeddings.
    
    Pipeline Stages:
    1. Embedding Stage: Initialize embedding model(s) and generate document embeddings
    2. Retrieval Stage: Initialize retrieval engine and build search index
    3. Generation Stage: Initialize RAG module for context-aware generation
    
    Args:
        config: Configuration dictionary with required sections:
                - 'embeddings': Embedding model configuration
                - 'retrieval': Retrieval engine configuration
                - 'generation': RAG module configuration
        documents: List of Document objects to process.
                  For multimodal:
                  - Must have 'content' (text)
                  - Optionally 'image_path' (for image embeddings)
        logger: Logger instance for pipeline logging
        batch_size: Optional batch size for embedding generation.
                   If None, uses config value or processes all at once.
    
    Returns:
        Tuple of (embedding_model_or_fusion, retrieval_engine, rag_module, embeddings_array)
        - embedding_model_or_fusion: Initialized embedding model or MultimodalFusion
        - retrieval_engine: Initialized RetrievalEngine with built index
        - rag_module: Initialized RAGModule ready for generation
        - embeddings_array: NumPy array of document embeddings, shape (n_documents, embedding_dim)
    
    Raises:
        ValueError: If documents list is empty or configuration is invalid
        RuntimeError: If any pipeline stage fails
    
    Example:
        >>> from src.pipeline.rag_pipeline import execute_full_rag_pipeline
        >>> from src.retrieval import DocumentLoader
        >>> import logging
        >>> 
        >>> logger = logging.getLogger(__name__)
        >>> config = {
        ...     'embeddings': {'model_type': 'bert', 'model_name': 'bert-base-uncased'},
        ...     'retrieval': {'similarity_metric': 'cosine'},
        ...     'generation': {'model_type': 'gpt2'}
        ... }
        >>> 
        >>> loader = DocumentLoader()
        >>> documents = loader.load_from_directory('data/text')
        >>> 
        >>> embedding_model, retrieval_engine, rag_module, embeddings = \\
        ...     execute_full_rag_pipeline(config, documents, logger)
    """
    logger.info("=" * 70)
    logger.info("EXECUTING FULL RAG PIPELINE")
    logger.info("=" * 70)
    
    if not documents:
        raise ValueError("Cannot execute RAG pipeline with empty documents list")
    
    logger.info(f"Pipeline configuration:")
    logger.info(f"  - Documents: {len(documents)}")
    logger.info(f"  - Batch size: {batch_size or 'default'}")
    
    try:
        # Stage 1: Embedding Pipeline
        logger.info("")
        logger.info("STAGE 1: EMBEDDING")
        logger.info("-" * 70)
        embedding_model_or_fusion, embeddings_array = run_full_embedding_pipeline(
            config=config,
            documents=documents,
            logger=logger,
            batch_size=batch_size
        )
        
        logger.info(f"✓ Embedding stage completed")
        logger.info(f"  Embeddings shape: {embeddings_array.shape}")
        logger.info(f"  Embedding dimension: {embeddings_array.shape[1]}")
        
        # Stage 2: Retrieval Pipeline
        logger.info("")
        logger.info("STAGE 2: RETRIEVAL")
        logger.info("-" * 70)
        retrieval_engine = run_retrieval_pipeline(
            config=config,
            embedding_model_or_fusion=embedding_model_or_fusion,
            logger=logger
        )
        
        # Build index
        logger.info("Building retrieval index...")
        doc_dicts = [doc.to_dict() for doc in documents]
        
        # Validate embedding dimension matches retrieval engine
        actual_embedding_dim = embeddings_array.shape[1]
        if retrieval_engine.embedding_dim != actual_embedding_dim:
            logger.warning(
                f"Embedding dimension mismatch: "
                f"retrieval_engine expects {retrieval_engine.embedding_dim}, "
                f"but embeddings have {actual_embedding_dim}. "
                f"Updating retrieval engine embedding_dim."
            )
            # Update retrieval engine's embedding dimension
            retrieval_engine.embedding_dim = actual_embedding_dim
        
        retrieval_engine.build_index(embeddings_array, doc_dicts)
        logger.info(f"✓ Retrieval stage completed")
        logger.info(f"  Index built with {len(documents)} documents")
        
        # Stage 3: Generation Pipeline
        logger.info("")
        logger.info("STAGE 3: GENERATION")
        logger.info("-" * 70)
        rag_module = run_generation_pipeline(
            config=config,
            retrieval_engine=retrieval_engine,
            logger=logger
        )
        
        logger.info(f"✓ Generation stage completed")
        logger.info("")
        logger.info("=" * 70)
        logger.info("FULL RAG PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Pipeline components ready for use:")
        logger.info(f"  - Embedding model: {type(embedding_model_or_fusion).__name__}")
        logger.info(f"  - Retrieval engine: Initialized with {len(documents)} documents")
        logger.info(f"  - RAG module: {rag_module.model_name}")
        logger.info(f"  - Embeddings: {embeddings_array.shape}")
        logger.info("")
        
        return embedding_model_or_fusion, retrieval_engine, rag_module, embeddings_array
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error("RAG PIPELINE EXECUTION FAILED")
        logger.error("=" * 70)
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise RuntimeError(
            f"RAG pipeline execution failed at stage. "
            f"Original error: {str(e)}"
        ) from e


# Backward compatibility alias
execute_rag_pipeline = execute_full_rag_pipeline
