"""
Main evaluation runner for AdaptiveMultimodalRAG.

Author: s Bostan
Created on: Nov, 2025
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    normalized_dcg
)
from evaluation.metrics.generation_metrics import (
    bleu_score,
    rouge_score,
    bert_score
)


def run_evaluation(
    config_path: str,
    results_path: str = None,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Run evaluation pipeline.
    
    Args:
        config_path: Path to evaluation configuration file
        results_path: Path to results directory
        output_path: Path to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Default paths
    if results_path is None:
        results_path = config.get('results_path', 'experiments/results')
    if output_path is None:
        output_path = config.get('output_path', 'evaluation_results.json')
    
    results_path = Path(results_path)
    output_path = Path(output_path)
    
    # Load results - look for pipeline_results.json files
    results = []
    if results_path.exists():
        # Look in subdirectories for pipeline_results.json
        for result_file in results_path.rglob('pipeline_results.json'):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    results.append(result_data)
            except Exception as e:
                print(f"Warning: Could not load {result_file}: {e}")
    
    # Calculate metrics
    metrics = {
        'retrieval': {},
        'generation': {},
        'summary': {}
    }
    
    if results:
        # Extract pipeline statistics
        total_experiments = len(results)
        completed = sum(1 for r in results if r.get('status') == 'completed')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        
        metrics['summary'] = {
            'total_experiments': total_experiments,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / total_experiments if total_experiments > 0 else 0
        }
        
        # Aggregate retrieval metrics from pipeline results
        total_documents = 0
        total_indexes_built = 0
        
        for result in results:
            if 'retrieval' in result:
                ret = result['retrieval']
                if 'num_documents' in ret:
                    total_documents += ret['num_documents']
                if ret.get('index_built'):
                    total_indexes_built += 1
            
            if 'documents' in result:
                doc_info = result['documents']
                if 'count' in doc_info:
                    total_documents += doc_info['count']
        
        if total_experiments > 0:
            metrics['retrieval'] = {
                'total_documents_processed': total_documents,
                'indexes_built': total_indexes_built,
                'avg_documents_per_experiment': total_documents / total_experiments if total_experiments > 0 else 0
            }
        
        # Aggregate generation metrics
        generation_completed = sum(1 for r in results if 'generation' in r and r['generation'].get('status') == 'completed')
        
        if total_experiments > 0:
            metrics['generation'] = {
                'generation_modules_initialized': generation_completed,
                'success_rate': generation_completed / total_experiments if total_experiments > 0 else 0
            }
        
        # Extract embedding statistics
        total_embeddings = 0
        for result in results:
            if 'embeddings' in result:
                emb = result['embeddings']
                if 'documents_processed' in emb:
                    total_embeddings += emb['documents_processed']
        
        if total_embeddings > 0:
            metrics['embeddings'] = {
                'total_embeddings_generated': total_embeddings,
                'avg_embeddings_per_experiment': total_embeddings / total_experiments if total_experiments > 0 else 0
            }
    else:
        metrics['summary'] = {
            'message': 'No results found',
            'results_path': str(results_path)
        }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {output_path}")
    return metrics


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description='Run evaluation for AdaptiveMultimodalRAG')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to evaluation configuration file')
    parser.add_argument('--results', type=str, default=None,
                       help='Path to results directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    metrics = run_evaluation(
        config_path=args.config,
        results_path=args.results,
        output_path=args.output
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

