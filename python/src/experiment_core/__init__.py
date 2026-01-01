"""
Experiments module for AdaptiveMultimodalRAG.

Contains experiment execution logic and lifecycle management.
"""

from .experiment_runner import (
    load_config,
    run_experiment
)

__all__ = [
    'load_config',
    'run_experiment'
]

