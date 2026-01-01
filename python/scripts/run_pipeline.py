"""
Main entry point script for AdaptiveMultimodalRAG experiments.

This is a thin CLI layer that validates arguments and delegates to the
experiment runner. It contains NO business logic, pipeline logic, or
research logic.

Author: s Bostan
Created on: Nov, 2025
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.experiment_core.experiment_runner import run_experiment


def main():
    """
    Main entry point for pipeline script.
    
    Flow:
    1. Parse command-line arguments
    2. Validate configuration file exists
    3. Call experiment_runner.run_experiment(...)
    4. Print execution summary
    5. Exit with appropriate exit code
    """
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
    
    # Validate configuration file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Execute experiment
    results = run_experiment(args.config, args.output)
    
    # Print execution summary
    print("\n" + "="*50)
    print("Pipeline Execution Summary")
    print("="*50)
    print(f"Experiment: {results.get('experiment', 'N/A')}")
    print(f"Status: {results.get('status', 'unknown')}")
    if results.get('status') == 'failed':
        print(f"Error: {results.get('error', 'Unknown error')}")
    print("="*50)
    
    # Exit with appropriate code
    exit_code = 0 if results.get('status') == 'completed' else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

