"""
Main pipeline runner for AdaptiveMultimodalRAG experiments.

Author: s Bostan
Created on: Nov, 2025
"""

import argparse
import sys
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings import BERTEmbedding, MultimodalFusion
from src.retrieval import RetrievalEngine, DocumentLoader
from src.generation import RAGModule
from src.logging import setup_logging, get_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_embedding_pipeline(config: Dict[str, Any], logger: logging.Logger) -> Any:
    """Run embedding generation pipeline."""
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
        # Note: batch_size and max_length are handled internally by BERTEmbedding
        return bert


def _create_sample_documents(loader: Any, dataset_path: str, logger: logging.Logger) -> list:
    """Create sample documents if none are found."""
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


def run_retrieval_pipeline(
    config: Dict[str, Any],
    embedding_model: Any,
    logger: logging.Logger
) -> RetrievalEngine:
    """Run retrieval pipeline."""
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
    """Run generation pipeline."""
    logger.info("Starting generation pipeline...")
    
    generation_config = config.get('generation', {})
    
    # RAGModule doesn't take retrieval_engine in __init__, it's used in generate() method
    rag = RAGModule(
        model_name=generation_config.get('model_type', 'gpt2'),
        max_context_length=generation_config.get('max_length', 512)
    )
    
    logger.info(f"Initialized RAG module with model: {generation_config.get('model_type', 'gpt2')}")
    return rag


def run_full_pipeline(config_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the full AdaptiveMultimodalRAG pipeline.
    
    Args:
        config_path: Path to experiment configuration file
        output_dir: Optional output directory override
        
    Returns:
        Dictionary containing pipeline results
    """
    # Load configuration
    config = load_config(config_path)
    exp_config = config.get('experiment', {})
    
    # Setup logging
    log_config = config.get('logging', {})
    output_dir = output_dir or config.get('output', {}).get('results_dir', 'experiments/results')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging system
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
        # Step 1: Embedding
        embedding_model = run_embedding_pipeline(config, logger)
        results['embeddings'] = {'status': 'completed'}
        
        # Step 2: Load documents
        from src.retrieval import DocumentLoader
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
        results['documents'] = {'count': len(documents), 'status': 'loaded'}
        
        # Step 3: Generate embeddings for documents
        logger.info("Generating document embeddings...")
        import numpy as np
        doc_embeddings = []
        for i, doc in enumerate(documents):
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(documents)} documents...")
            embedding = embedding_model.get_embedding(doc.content)
            doc_embeddings.append(embedding)
        
        embeddings_array = np.array(doc_embeddings)
        logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
        results['embeddings']['documents_processed'] = len(documents)
        
        # Step 4: Retrieval
        retrieval_engine = run_retrieval_pipeline(config, embedding_model, logger)
        
        # Build index
        logger.info("Building retrieval index...")
        doc_dicts = [doc.to_dict() for doc in documents]
        retrieval_engine.build_index(embeddings_array, doc_dicts)
        logger.info("Index built successfully")
        results['retrieval'] = {'status': 'completed', 'index_built': True}
        
        # Step 5: Generation
        rag_module = run_generation_pipeline(config, retrieval_engine, logger)
        results['generation'] = {'status': 'completed'}
        
        results['status'] = 'completed'
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        results['status'] = 'failed'
        results['error'] = str(e)
    
    # Save results
    results_file = output_path / 'pipeline_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    return results


def main():
    """Main entry point for pipeline script."""
    parser = argparse.ArgumentParser(
        description='Run AdaptiveMultimodalRAG pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    results = run_full_pipeline(args.config, args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("Pipeline Execution Summary")
    print("="*50)
    print(f"Experiment: {results.get('experiment', 'N/A')}")
    print(f"Status: {results.get('status', 'unknown')}")
    if results.get('status') == 'failed':
        print(f"Error: {results.get('error', 'Unknown error')}")
    print("="*50)


if __name__ == '__main__':
    main()

