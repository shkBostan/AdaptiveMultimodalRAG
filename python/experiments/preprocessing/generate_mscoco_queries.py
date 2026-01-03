"""
Generate retrieval queries from MS COCO captions for evaluation.

This script:
1. Loads generated documents from documents.jsonl
2. Generates queries from captions (direct, partial, or paraphrased)
3. Creates ground truth relevance pairs
4. Saves queries to JSONL format for evaluation

Author: s Bostan
Created on: Jan, 2026
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum
import random
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryStrategy(Enum):
    """Query generation strategies."""
    DIRECT = "direct"  # Use caption as-is
    PARTIAL = "partial"  # Use first part of caption
    RANDOM_WORDS = "random_words"  # Use random subset of words


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)


def load_documents(documents_file: Path) -> List[Dict[str, Any]]:
    """
    Load documents from JSONL file.
    
    Args:
        documents_file: Path to documents.jsonl file
        
    Returns:
        List of document dictionaries
        
    Raises:
        FileNotFoundError: If documents file doesn't exist
    """
    if not documents_file.exists():
        raise FileNotFoundError(f"Documents file not found: {documents_file}")
    
    logger.info(f"Loading documents from: {documents_file}")
    documents = []
    
    with open(documents_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                doc = json.loads(line.strip())
                documents.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def build_image_id_to_documents(documents: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Build index mapping image_id to all documents with that image.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Dictionary mapping image_id to list of documents
    """
    index = {}
    for doc in documents:
        image_id = doc.get('metadata', {}).get('image_id')
        if image_id is not None:
            if image_id not in index:
                index[image_id] = []
            index[image_id].append(doc)
    return index


def generate_query_direct(caption: str) -> str:
    """Generate query using direct caption strategy."""
    return caption.strip()


def generate_query_partial(caption: str, ratio: float = 0.5) -> str:
    """
    Generate query using partial caption strategy.
    
    Takes the first part of the caption based on ratio.
    
    Args:
        caption: Full caption text
        ratio: Ratio of caption to use (default: 0.5)
        
    Returns:
        Partial caption query
    """
    words = caption.strip().split()
    n_words = max(1, int(len(words) * ratio))
    return ' '.join(words[:n_words])


def generate_query_random_words(caption: str, ratio: float = 0.6) -> str:
    """
    Generate query using random words strategy.
    
    Selects random subset of words from caption.
    
    Args:
        caption: Full caption text
        ratio: Ratio of words to keep (default: 0.6)
        
    Returns:
        Query with random word subset
    """
    words = caption.strip().split()
    n_words = max(1, int(len(words) * ratio))
    selected_words = random.sample(words, min(n_words, len(words)))
    return ' '.join(selected_words)


def generate_query_text(
    caption: str,
    strategy: QueryStrategy,
    strategy_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate query text from caption using specified strategy.
    
    Args:
        caption: Original caption text
        strategy: Query generation strategy
        strategy_params: Optional parameters for strategy
        
    Returns:
        Generated query text
    """
    if strategy_params is None:
        strategy_params = {}
    
    if strategy == QueryStrategy.DIRECT:
        return generate_query_direct(caption)
    elif strategy == QueryStrategy.PARTIAL:
        ratio = strategy_params.get('ratio', 0.5)
        return generate_query_partial(caption, ratio)
    elif strategy == QueryStrategy.RANDOM_WORDS:
        ratio = strategy_params.get('ratio', 0.6)
        return generate_query_random_words(caption, ratio)
    else:
        raise ValueError(f"Unknown query strategy: {strategy}")


def create_relevance_pairs(
    query_doc: Dict[str, Any],
    all_documents: List[Dict[str, Any]],
    image_id_index: Dict[int, List[Dict[str, Any]]],
    split: str
) -> Set[str]:
    """
    Create ground truth relevance pairs for a query.
    
    A document is relevant if it has the same image_id as the query document.
    
    Args:
        query_doc: Query document
        all_documents: All available documents
        image_id_index: Index mapping image_id to documents
        split: Dataset split (for filtering)
        
    Returns:
        Set of relevant document IDs
    """
    query_image_id = query_doc.get('metadata', {}).get('image_id')
    if query_image_id is None:
        return set()
    
    # Find all documents with the same image_id (from different splits if needed)
    relevant_docs = image_id_index.get(query_image_id, [])
    
    # Filter by split if query is from val/test (only use train as relevant)
    if split in ['val', 'test']:
        relevant_docs = [
            doc for doc in relevant_docs
            if doc.get('metadata', {}).get('split') == 'train'
        ]
    else:
        # For train queries, use all documents with same image_id
        relevant_docs = [
            doc for doc in relevant_docs
            if doc.get('id') != query_doc.get('id')  # Exclude query document itself
        ]
    
    return {doc['id'] for doc in relevant_docs}


def create_query(
    query_id: str,
    query_text: str,
    relevant_doc_ids: Set[str],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a query dictionary.
    
    Args:
        query_id: Unique query identifier
        query_text: Query text
        relevant_doc_ids: Set of relevant document IDs
        metadata: Query metadata
        
    Returns:
        Query dictionary
    """
    return {
        "query_id": query_id,
        "query_text": query_text,
        "relevant_doc_ids": sorted(list(relevant_doc_ids)),  # Sorted for reproducibility
        "metadata": {
            **metadata,
            "num_relevant": len(relevant_doc_ids),
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
    }


def generate_queries(
    documents: List[Dict[str, Any]],
    image_id_index: Dict[int, List[Dict[str, Any]]],
    split: str,
    strategy: QueryStrategy = QueryStrategy.DIRECT,
    strategy_params: Optional[Dict[str, Any]] = None,
    max_queries: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate queries from documents.
    
    Args:
        documents: List of document dictionaries to generate queries from
        image_id_index: Index mapping image_id to documents
        split: Dataset split (train/val/test)
        strategy: Query generation strategy
        strategy_params: Optional parameters for strategy
        max_queries: Optional maximum number of queries to generate
        
    Returns:
        List of query dictionaries
    """
    queries = []
    query_counter = 0
    
    logger.info(f"Generating queries for split '{split}' using strategy '{strategy.value}'")
    
    # Filter documents by split
    split_documents = [doc for doc in documents if doc.get('metadata', {}).get('split') == split]
    
    if max_queries:
        split_documents = split_documents[:max_queries]
    
    for doc in split_documents:
        query_counter += 1
        doc_id = doc['id']
        caption = doc.get('content', '')
        
        if not caption:
            logger.warning(f"Skipping document with empty content: {doc_id}")
            continue
        
        # Generate query text
        query_text = generate_query_text(caption, strategy, strategy_params)
        
        if not query_text:
            logger.warning(f"Generated empty query for document: {doc_id}")
            continue
        
        # Create relevance pairs
        relevant_doc_ids = create_relevance_pairs(
            query_doc=doc,
            all_documents=documents,
            image_id_index=image_id_index,
            split=split
        )
        
        # Create query ID
        query_id = f"q_{doc_id}"
        
        # Create query
        query = create_query(
            query_id=query_id,
            query_text=query_text,
            relevant_doc_ids=relevant_doc_ids,
            metadata={
                "split": split,
                "source": "caption",
                "generation_strategy": strategy.value,
                "source_doc_id": doc_id,
                "image_id": doc.get('metadata', {}).get('image_id')
            }
        )
        
        queries.append(query)
        
        if query_counter % 1000 == 0:
            logger.info(f"Generated {query_counter} queries...")
    
    logger.info(f"Generated {len(queries)} queries for split '{split}'")
    return queries


def save_queries_jsonl(queries: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Save queries to JSONL file.
    
    Args:
        queries: List of query dictionaries
        output_file: Path to output JSONL file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(queries)} queries to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    logger.info(f"Queries saved successfully to: {output_file}")


def generate_statistics(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics for queries.
    
    Args:
        queries: List of query dictionaries
        
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_queries": len(queries),
        "total_relevant_docs": sum(len(q['relevant_doc_ids']) for q in queries),
        "avg_relevant_per_query": 0,
        "queries_with_relevant": 0,
        "queries_without_relevant": 0,
        "by_split": {}
    }
    
    if queries:
        stats["avg_relevant_per_query"] = stats["total_relevant_docs"] / len(queries)
        stats["queries_with_relevant"] = sum(
            1 for q in queries if len(q['relevant_doc_ids']) > 0
        )
        stats["queries_without_relevant"] = len(queries) - stats["queries_with_relevant"]
        
        # Statistics by split
        for query in queries:
            split = query.get('metadata', {}).get('split', 'unknown')
            if split not in stats["by_split"]:
                stats["by_split"][split] = {
                    "count": 0,
                    "total_relevant": 0
                }
            stats["by_split"][split]["count"] += 1
            stats["by_split"][split]["total_relevant"] += len(query['relevant_doc_ids'])
    
    return stats


def main(
    documents_file: Path,
    output_file: Optional[Path] = None,
    splits: List[str] = None,
    strategy: str = "direct",
    strategy_params: Optional[Dict[str, Any]] = None,
    max_queries: Optional[int] = None,
    seed: int = 42
) -> None:
    """
    Main function to generate queries from documents.
    
    Args:
        documents_file: Path to documents.jsonl file
        output_file: Path to output queries JSONL file
        splits: List of splits to generate queries for (default: ['val'])
        strategy: Query generation strategy (direct/partial/random_words)
        strategy_params: Optional parameters for strategy
        max_queries: Optional maximum queries per split
        seed: Random seed for reproducibility
    """
    if splits is None:
        splits = ['val']
    
    set_seed(seed)
    
    # Parse strategy
    try:
        query_strategy = QueryStrategy(strategy.lower())
    except ValueError:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {[s.value for s in QueryStrategy]}"
        )
    
    logger.info("=" * 70)
    logger.info("MS COCO Query Generator")
    logger.info("=" * 70)
    logger.info(f"Documents file: {documents_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Splits: {splits}")
    logger.info(f"Strategy: {query_strategy.value}")
    logger.info(f"Random seed: {seed}")
    
    # Load documents
    documents = load_documents(documents_file)
    
    if not documents:
        logger.error("No documents loaded. Cannot generate queries.")
        sys.exit(1)
    
    # Build image_id index
    image_id_index = build_image_id_to_documents(documents)
    logger.info(f"Indexed {len(image_id_index)} unique images")
    
    # Generate queries for each split
    all_queries = []
    for split in splits:
        queries = generate_queries(
            documents=documents,
            image_id_index=image_id_index,
            split=split,
            strategy=query_strategy,
            strategy_params=strategy_params,
            max_queries=max_queries
        )
        all_queries.extend(queries)
    
    # Save queries
    if all_queries:
        if output_file is None:
            output_file = documents_file.parent / "queries.jsonl"
        
        save_queries_jsonl(all_queries, output_file)
        
        # Generate and log statistics
        stats = generate_statistics(all_queries)
        logger.info("=" * 70)
        logger.info("Query Generation Complete")
        logger.info("=" * 70)
        logger.info(f"Total queries: {stats['total_queries']}")
        logger.info(f"Average relevant docs per query: {stats['avg_relevant_per_query']:.2f}")
        logger.info(f"Queries with relevant docs: {stats['queries_with_relevant']}")
        logger.info(f"Queries without relevant docs: {stats['queries_without_relevant']}")
    else:
        logger.error("No queries were generated.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate retrieval queries from MS COCO captions"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="experiments/datasets/mscoco/processed/pipeline/documents.jsonl",
        help="Path to documents.jsonl file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output queries.jsonl file (default: same dir as documents)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val"],
        help="Dataset splits to generate queries for (default: val)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="direct",
        choices=["direct", "partial", "random_words"],
        help="Query generation strategy (default: direct)"
    )
    parser.add_argument(
        "--strategy-ratio",
        type=float,
        default=None,
        help="Ratio for partial/random_words strategy (default: 0.5/0.6)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum queries per split (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Prepare strategy params
    strategy_params = None
    if args.strategy in ["partial", "random_words"] and args.strategy_ratio:
        strategy_params = {"ratio": args.strategy_ratio}
    elif args.strategy == "partial":
        strategy_params = {"ratio": 0.5}
    elif args.strategy == "random_words":
        strategy_params = {"ratio": 0.6}
    
    main(
        documents_file=Path(args.documents),
        output_file=Path(args.output) if args.output else None,
        splits=args.splits,
        strategy=args.strategy,
        strategy_params=strategy_params,
        max_queries=args.max_queries,
        seed=args.seed
    )

