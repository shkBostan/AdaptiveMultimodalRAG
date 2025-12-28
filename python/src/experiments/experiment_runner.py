"""
Experiment runner for AdaptiveMultimodalRAG experiments.

This module handles experiment lifecycle: loading configuration, setting up
logging, executing the RAG pipeline, handling exceptions, and saving results.

Author: s Bostan
Created on: Nov, 2025
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..logging import setup_logging, get_logger
from ..retrieval import DocumentLoader
from ..pipeline.rag_pipeline import execute_rag_pipeline


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def _create_sample_documents(loader: DocumentLoader, dataset_path: str, logger: logging.Logger) -> list:
    """
    Create sample documents if none are found.
    
    Args:
        loader: DocumentLoader instance
        dataset_path: Path to dataset directory
        logger: Logger instance
        
    Returns:
        List of Document objects
    """
    sample_path = Path(dataset_path) / 'text'
    sample_path.mkdir(parents=True, exist_ok=True)
    
    sample_texts = [
        "Retrieval-Augmented Generation (RAG) enhances language models by incorporating external knowledge.",
        "Multimodal learning processes information from text, images, and audio simultaneously.",
        "Embedding models convert data into numerical vectors for similarity search.",
        "Evaluation metrics like Precision@K and Recall@K measure retrieval quality.",
        "Decision-centric systems require reliability and traceability for production use."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        sample_file = sample_path / f'sample{i}.txt'
        if not sample_file.exists():
            sample_file.write_text(text)
    
    logger.info(f"Created {len(sample_texts)} sample documents in {sample_path}")
    return loader.load_from_directory(str(sample_path))


def _load_documents(config: Dict[str, Any], logger: logging.Logger) -> list:
    """
    Load documents from dataset path specified in configuration.
    
    Args:
        config: Configuration dictionary with 'data' section
        logger: Logger instance
        
    Returns:
        List of Document objects
    """
    data_config = config.get('data', {})
    dataset_path = data_config.get('dataset_path', 'experiments/dataset1')
    
    logger.info(f"Loading documents from: {dataset_path}")
    loader = DocumentLoader()
    
    # Try to load from text directory
    text_path = Path(dataset_path) / 'text'
    if text_path.exists():
        documents = loader.load_from_directory(str(text_path))
    else:
        # Fallback to dataset_path directly
        documents = loader.load_from_directory(dataset_path)
    
    if not documents:
        logger.warning("No documents loaded. Creating sample documents...")
        # Create sample documents if none found
        documents = _create_sample_documents(loader, dataset_path, logger)
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def run_experiment(config_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a complete experiment: load config, execute pipeline, save results.
    
    This function manages the experiment lifecycle:
    1. Load configuration
    2. Setup logging
    3. Load documents
    4. Execute RAG pipeline
    5. Handle exceptions
    6. Save results to disk
    
    Args:
        config_path: Path to experiment configuration file
        output_dir: Optional output directory override (overrides config)
        
    Returns:
        Dictionary containing experiment results with status and metadata
    """
    # Load configuration
    config = load_config(config_path)
    exp_config = config.get('experiment', {})
    
    # Setup output directory
    output_dir = output_dir or config.get('output', {}).get('results_dir', 'experiments/results')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_config = config.get('logging', {})
    setup_logging(
        config_path=log_config.get('config_path'),
        env=log_config.get('env', 'dev')
    )
    
    # Get experiment logger
    logger = get_logger(exp_config.get('name', 'pipeline'))
    
    logger.info(f"Starting experiment: {exp_config.get('name', 'unknown')}")
    logger.info(f"Description: {exp_config.get('description', 'N/A')}")
    
    results = {
        'experiment': exp_config.get('name'),
        'status': 'running'
    }
    
    try:
        # Load documents
        documents = _load_documents(config, logger)
        results['documents'] = {'count': len(documents), 'status': 'loaded'}
        
        # Execute RAG pipeline
        embedding_model, retrieval_engine, rag_module, embeddings_array = execute_rag_pipeline(
            config=config,
            documents=documents,
            logger=logger
        )
        
        # Update results
        results['embeddings'] = {
            'status': 'completed',
            'documents_processed': len(documents),
            'embedding_shape': list(embeddings_array.shape)
        }
        results['retrieval'] = {
            'status': 'completed',
            'index_built': True,
            'index_size': len(documents)
        }
        results['generation'] = {'status': 'completed'}
        results['status'] = 'completed'
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        results['status'] = 'failed'
        results['error'] = str(e)
    
    # Save results
    results_file = output_path / 'pipeline_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    return results

