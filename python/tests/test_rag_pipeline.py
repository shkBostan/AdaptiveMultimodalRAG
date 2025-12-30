# tests/test_rag_pipeline.py

import logging
import pytest
from src.pipeline.rag_pipeline import execute_full_rag_pipeline
from src.retrieval import Document

# Optional: configure logging for test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke-test")

@pytest.mark.smoke
def test_full_rag_pipeline():
    """Smoke test: Execute the full RAG pipeline on a small sample of documents."""
    
    config = {
        "embeddings": {
            "model_type": "bert",
            "model_name": "bert-base-uncased",
            "batch_size": 2
        },
        "retrieval": {
            "similarity_metric": "cosine"
        },
        "generation": {
            "model_type": "gpt2",
            "max_length": 64
        }
    }

    documents = [
        Document(id="1", content="What is retrieval augmented generation?", metadata={}),
        Document(id="2", content="RAG combines vector search with LLMs.", metadata={})
    ]

    # Execute the full pipeline
    emb, retr, rag, vecs = execute_full_rag_pipeline(
        config=config,
        documents=documents,
        logger=logger
    )

    # Assertions to ensure pipeline executed correctly
    assert vecs.shape[0] == len(documents), "Number of embeddings should match number of documents"
    assert vecs.shape[1] > 0, "Embedding dimension should be greater than 0"

    # Optional: check types of returned components
    from src.embeddings.bert_embedding import BERTEmbedding
    assert isinstance(emb, BERTEmbedding), "Embedding model should be BERTEmbedding"
    assert retr is not None, "Retrieval engine should not be None"
    assert rag is not None, "RAG module should not be None"
