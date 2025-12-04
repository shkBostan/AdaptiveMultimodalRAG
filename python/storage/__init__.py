"""
Storage module for AdaptiveMultimodalRAG.

Provides persistence capabilities for documents, embeddings, and indices.

Author: s Bostan
Created on: Nov, 2025
"""

from .persistence_manager import PersistenceManager, ensure_dir, validate_file_path

__all__ = ['PersistenceManager', 'ensure_dir', 'validate_file_path']

