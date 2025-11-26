"""
Tests for generation modules.

Author: s Bostan
Created on: Nov, 2025
"""

import pytest
from generation.rag_module import RAGModule
from generation.generator import Generator


class TestRAGModule:
    """Test cases for RAG module."""
    
    def test_rag_initialization(self):
        """Test RAG module initialization."""
        rag = RAGModule(model_name="gpt2")
        assert rag.model_name == "gpt2"
    
    def test_context_construction(self):
        """Test context construction from retrieved documents."""
        rag = RAGModule()
        retrieved_docs = [
            {'content': 'Document 1'},
            {'content': 'Document 2'}
        ]
        context = rag._construct_context(retrieved_docs)
        assert '[1] Document 1' in context
        assert '[2] Document 2' in context


class TestGenerator:
    """Test cases for text generator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        gen = Generator(model_name="gpt2")
        assert gen.model_name == "gpt2"

