"""
Test suite for RAG pipeline components and full pipeline execution.

This module tests the RAG pipeline components:
- Embedding pipeline stage (run_embedding_pipeline, run_full_embedding_pipeline)
- Retrieval pipeline stage (run_retrieval_pipeline)
- Generation pipeline stage (run_generation_pipeline)
- Full pipeline orchestration (execute_full_rag_pipeline)

Each stage is tested independently and together, ensuring:
- Component initialization works correctly
- Data flows properly between stages
- Output shapes and types are validated
- Integration works end-to-end

Tests are designed to be:
- Modular: Each stage can be tested independently
- Fast: Use minimal data and lightweight models
- Informative: Include detailed logging and assertions

Author: s Bostan
Created on: Dec, 2025
"""

import pytest
import numpy as np
import logging

from src.pipeline.rag_pipeline import (
    run_embedding_pipeline,
    run_full_embedding_pipeline,
    run_retrieval_pipeline,
    run_generation_pipeline,
    execute_full_rag_pipeline
)
from src.retrieval import Document
from src.embeddings import BERTEmbedding, MultimodalFusion


# ============================================================================
# EMBEDDING PIPELINE STAGE TESTS
# ============================================================================

def test_run_embedding_pipeline_bert(logger, bert_config):
    """Test run_embedding_pipeline for BERT embedding model initialization."""
    logger.info("Testing run_embedding_pipeline (BERT)...")
    
    embedding_model = run_embedding_pipeline(bert_config, logger)
    
    # Validate returned model
    assert isinstance(embedding_model, BERTEmbedding), "Should return BERTEmbedding instance"
    assert embedding_model.model_name == "bert-base-uncased"
    assert embedding_model.pooling_strategy.value == "mean"
    
    logger.info(f"✓ Embedding pipeline initialized: {type(embedding_model).__name__}")
    logger.info(f"  Model: {embedding_model.model_name}")
    logger.info(f"  Embedding dim: {embedding_model.embedding_dim}")


def test_run_embedding_pipeline_multimodal(logger, multimodal_config):
    """Test run_embedding_pipeline for multimodal fusion initialization."""
    logger.info("Testing run_embedding_pipeline (Multimodal Fusion)...")
    
    fusion = run_embedding_pipeline(multimodal_config, logger)
    
    # Validate returned fusion object
    assert isinstance(fusion, MultimodalFusion), "Should return MultimodalFusion instance"
    assert fusion.strategy.value == "concatenation"
    
    logger.info(f"✓ Multimodal fusion pipeline initialized: {type(fusion).__name__}")
    logger.info(f"  Strategy: {fusion.strategy.value}")


def test_run_full_embedding_pipeline_bert(logger, bert_config, sample_documents):
    """Test run_full_embedding_pipeline for BERT embeddings generation."""
    logger.info("Testing run_full_embedding_pipeline (BERT)...")
    
    embedding_model, embeddings_array = run_full_embedding_pipeline(
        config=bert_config,
        documents=sample_documents,
        logger=logger,
        batch_size=2
    )
    
    # Validate returned components
    assert isinstance(embedding_model, BERTEmbedding), "Should return BERTEmbedding"
    assert isinstance(embeddings_array, np.ndarray), "Should return numpy array"
    
    # Validate embeddings shape
    assert embeddings_array.ndim == 2, "Embeddings should be 2D array"
    assert embeddings_array.shape[0] == len(sample_documents), "Number of embeddings should match documents"
    assert embeddings_array.shape[1] == 768, "BERT-base produces 768-dim embeddings"
    assert embeddings_array.dtype == np.float32, "Embeddings should be float32"
    
    logger.info(f"✓ Full embedding pipeline completed")
    logger.info(f"  Embedding model: {type(embedding_model).__name__}")
    logger.info(f"  Embeddings shape: {embeddings_array.shape}")
    logger.info(f"  Embeddings dtype: {embeddings_array.dtype}")
    logger.info(f"  Mean embedding norm: {np.mean([np.linalg.norm(e) for e in embeddings_array]):.4f}")


def test_run_full_embedding_pipeline_batch_processing(logger, bert_config):
    """Test run_full_embedding_pipeline with batch processing."""
    logger.info("Testing run_full_embedding_pipeline batch processing...")
    
    # Create more documents to test batching
    documents = [
        Document(id=f"doc{i}", content=f"Document {i} content about machine learning.", metadata={})
        for i in range(10)
    ]
    
    embedding_model, embeddings_array = run_full_embedding_pipeline(
        config=bert_config,
        documents=documents,
        logger=logger,
        batch_size=3  # Small batch size to test batching
    )
    
    assert embeddings_array.shape[0] == len(documents), "Should process all documents"
    assert embeddings_array.shape[1] == 768, "Embedding dimension should be consistent"
    
    logger.info(f"✓ Batch processing test passed: {len(documents)} documents processed")
    logger.info(f"  Final embeddings shape: {embeddings_array.shape}")


# ============================================================================
# RETRIEVAL PIPELINE STAGE TESTS
# ============================================================================

def test_run_retrieval_pipeline(logger, bert_config, bert_embedding_model):
    """Test run_retrieval_pipeline for retrieval engine initialization."""
    logger.info("Testing run_retrieval_pipeline...")
    
    retrieval_config = {
        'retrieval': {
            'method': 'faiss',
            'similarity_metric': 'cosine',
            'top_k': 5
        }
    }
    
    retrieval_engine = run_retrieval_pipeline(
        config=retrieval_config,
        embedding_model_or_fusion=bert_embedding_model,
        logger=logger
    )
    
    # Validate returned engine
    assert retrieval_engine is not None, "Retrieval engine should not be None"
    assert retrieval_engine.embedding_dim == 768, "Embedding dimension should match BERT (768)"
    assert retrieval_engine.index_type == "cosine", "Index type should match config"
    
    logger.info(f"✓ Retrieval pipeline initialized")
    logger.info(f"  Embedding dimension: {retrieval_engine.embedding_dim}")
    logger.info(f"  Index type: {retrieval_engine.index_type}")


def test_retrieval_engine_build_and_search(logger, bert_config, bert_embedding_model, sample_documents):
    """Test retrieval engine index building and search functionality."""
    logger.info("Testing retrieval engine build index and search...")
    
    # Generate embeddings first
    _, embeddings_array = run_full_embedding_pipeline(
        config=bert_config,
        documents=sample_documents,
        logger=logger
    )
    
    # Initialize retrieval engine
    retrieval_config = {
        'retrieval': {
            'method': 'faiss',
            'similarity_metric': 'cosine'
        }
    }
    retrieval_engine = run_retrieval_pipeline(
        config=retrieval_config,
        embedding_model_or_fusion=bert_embedding_model,
        logger=logger
    )
    
    # Build index
    doc_dicts = [doc.to_dict() for doc in sample_documents]
    retrieval_engine.build_index(embeddings_array, doc_dicts)
    
    logger.info("✓ Index built successfully")
    
    # Test search
    query_text = "machine learning"
    query_embedding = bert_embedding_model.get_embedding(query_text)
    
    results = retrieval_engine.search(query_embedding, top_k=2)
    
    # Validate search results
    assert len(results) == 2, "Should return top_k results"
    assert all('score' in result for result in results), "Results should have scores"
    assert all('id' in result for result in results), "Results should have document IDs"
    assert all(result['score'] >= 0 for result in results), "Scores should be non-negative"
    
    logger.info(f"✓ Search successful: retrieved {len(results)} documents")
    for i, result in enumerate(results, 1):
        logger.info(f"  [{i}] Doc ID: {result['id']}, Score: {result['score']:.4f}")




# ============================================================================
# GENERATION PIPELINE STAGE TESTS
# ============================================================================

def test_run_generation_pipeline(logger, full_rag_config, bert_embedding_model):
    """Test run_generation_pipeline for RAG module initialization."""
    logger.info("Testing run_generation_pipeline...")
    
    # Initialize retrieval engine (needed for pipeline, but not for RAGModule init)
    retrieval_config = {'retrieval': {'similarity_metric': 'cosine'}}
    retrieval_engine = run_retrieval_pipeline(
        config=retrieval_config,
        embedding_model_or_fusion=bert_embedding_model,
        logger=logger
    )
    
    rag_module = run_generation_pipeline(
        config=full_rag_config,
        retrieval_engine=retrieval_engine,
        logger=logger
    )
    
    # Validate returned module
    assert rag_module is not None, "RAG module should not be None"
    assert rag_module.model_name == "gpt2", "Model name should match config"
    assert rag_module.model is None, "Model should not be loaded yet (lazy loading)"
    
    logger.info(f"✓ Generation pipeline initialized")
    logger.info(f"  Model name: {rag_module.model_name}")
    logger.info(f"  Max context length: {rag_module.max_context_length}")


def test_rag_module_generation(logger, full_rag_config, bert_embedding_model, sample_documents):
    """Test RAG module text generation with retrieved context."""
    logger.info("Testing RAG module generation...")
    
    # Setup retrieval engine with documents
    _, embeddings_array = run_full_embedding_pipeline(
        config=full_rag_config,
        documents=sample_documents,
        logger=logger
    )
    
    retrieval_config = {'retrieval': {'similarity_metric': 'cosine'}}
    retrieval_engine = run_retrieval_pipeline(
        config=retrieval_config,
        embedding_model_or_fusion=bert_embedding_model,
        logger=logger
    )
    
    doc_dicts = [doc.to_dict() for doc in sample_documents]
    retrieval_engine.build_index(embeddings_array, doc_dicts)
    
    # Initialize RAG module
    rag_module = run_generation_pipeline(
        config=full_rag_config,
        retrieval_engine=retrieval_engine,
        logger=logger
    )
    
    # Retrieve documents for a query
    query = "What is retrieval augmented generation?"
    query_embedding = bert_embedding_model.get_embedding(query)
    retrieved_docs = retrieval_engine.search(query_embedding, top_k=2)
    
    # Generate response
    logger.info(f"  Generating response for query: '{query}'")
    logger.info(f"  Retrieved {len(retrieved_docs)} documents")
    
    response = rag_module.generate(
        query=query,
        retrieved_docs=retrieved_docs,
        max_length=50,  # Short for testing
        temperature=0.7
    )
    
    # Validate generated response
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    
    logger.info(f"✓ Generation successful")
    logger.info(f"  Query: '{query}'")
    logger.info(f"  Response length: {len(response)} characters")
    logger.info(f"  Response preview: '{response[:100]}...'")


# ============================================================================
# FULL PIPELINE INTEGRATION TESTS
# ============================================================================

@pytest.mark.smoke
def test_full_rag_pipeline(logger, full_rag_config):
    """Smoke test: Execute the full RAG pipeline on a small sample of documents."""
    logger.info("=" * 70)
    logger.info("Testing full RAG pipeline (smoke test)")
    logger.info("=" * 70)
    
    documents = [
        Document(id="1", content="What is retrieval augmented generation?", metadata={}),
        Document(id="2", content="RAG combines vector search with LLMs.", metadata={}),
        Document(id="3", content="BERT is used for text embeddings in RAG systems.", metadata={})
    ]
    
    # Execute the full pipeline
    emb, retr, rag, vecs = execute_full_rag_pipeline(
        config=full_rag_config,
        documents=documents,
        logger=logger
    )
    
    # Assertions to ensure pipeline executed correctly
    assert vecs.shape[0] == len(documents), "Number of embeddings should match number of documents"
    assert vecs.shape[1] > 0, "Embedding dimension should be greater than 0"
    assert vecs.shape[1] == 768, "BERT-base produces 768-dim embeddings"
    
    # Check types of returned components
    assert isinstance(emb, BERTEmbedding), "Embedding model should be BERTEmbedding"
    assert retr is not None, "Retrieval engine should not be None"
    assert rag is not None, "RAG module should not be None"
    
    # Verify retrieval engine has index built
    assert retr.index is not None, "Retrieval engine index should be built"
    
    logger.info("=" * 70)
    logger.info("✓ Full RAG pipeline test passed")
    logger.info(f"  Documents processed: {len(documents)}")
    logger.info(f"  Embeddings shape: {vecs.shape}")
    logger.info(f"  Embedding model: {type(emb).__name__}")
    logger.info(f"  Retrieval engine: Initialized with index")
    logger.info(f"  RAG module: {rag.model_name}")
    logger.info("=" * 70)


def test_full_rag_pipeline_with_retrieval_and_generation(logger, full_rag_config):
    """Test full RAG pipeline with actual retrieval and generation."""
    logger.info("=" * 70)
    logger.info("Testing full RAG pipeline with retrieval and generation")
    logger.info("=" * 70)
    
    documents = [
        Document(id="doc1", content="Machine learning is a subset of artificial intelligence.", metadata={}),
        Document(id="doc2", content="Deep learning uses neural networks with multiple layers.", metadata={}),
        Document(id="doc3", content="RAG combines retrieval systems with language models for better answers.", metadata={}),
        Document(id="doc4", content="BERT embeddings are used to find relevant documents.", metadata={})
    ]
    
    # Execute full pipeline
    embedding_model, retrieval_engine, rag_module, embeddings_array = execute_full_rag_pipeline(
        config=full_rag_config,
        documents=documents,
        logger=logger
    )
    
    # Test retrieval
    query = "What is machine learning?"
    logger.info(f"\nTesting retrieval with query: '{query}'")
    query_embedding = embedding_model.get_embedding(query)
    results = retrieval_engine.search(query_embedding, top_k=2)
    
    assert len(results) == 2, "Should retrieve top_k documents"
    logger.info(f"✓ Retrieved {len(results)} documents")
    for i, result in enumerate(results, 1):
        logger.info(f"  [{i}] Doc: {result['id']}, Score: {result.get('score', 'N/A'):.4f}")
    
    # Test generation
    logger.info(f"\nTesting generation with query: '{query}'")
    response = rag_module.generate(
        query=query,
        retrieved_docs=results,
        max_length=60,
        temperature=0.7
    )
    
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    
    logger.info(f"✓ Generated response: '{response[:150]}...'")
    logger.info("=" * 70)


def test_full_rag_pipeline_empty_documents(logger, full_rag_config):
    """Test full RAG pipeline with empty documents list (should raise error)."""
    logger.info("Testing full RAG pipeline with empty documents (error case)...")
    
    documents = []
    
    with pytest.raises(ValueError, match="Cannot execute RAG pipeline with empty documents"):
        execute_full_rag_pipeline(
            config=full_rag_config,
            documents=documents,
            logger=logger
        )
    
    logger.info("✓ Error handling test passed: empty documents correctly rejected")


def test_full_rag_pipeline_output_shapes(logger, full_rag_config):
    """Test that full pipeline returns correctly shaped outputs."""
    logger.info("Testing full RAG pipeline output shapes...")
    
    documents = [
        Document(id=f"doc{i}", content=f"Document {i} content.", metadata={})
        for i in range(5)
    ]
    
    embedding_model, retrieval_engine, rag_module, embeddings_array = execute_full_rag_pipeline(
        config=full_rag_config,
        documents=documents,
        logger=logger
    )
    
    # Validate all output shapes and types
    assert embeddings_array.shape == (5, 768), f"Expected (5, 768), got {embeddings_array.shape}"
    assert embeddings_array.dtype == np.float32, "Embeddings should be float32"
    
    assert embedding_model.embedding_dim == 768, "Embedding model dimension should match"
    assert retrieval_engine.embedding_dim == 768, "Retrieval engine dimension should match"
    assert retrieval_engine.index is not None, "Index should be built"
    
    logger.info("✓ All output shapes validated correctly")
    logger.info(f"  Embeddings: {embeddings_array.shape}")
    logger.info(f"  Embedding model dim: {embedding_model.embedding_dim}")
    logger.info(f"  Retrieval engine dim: {retrieval_engine.embedding_dim}")
