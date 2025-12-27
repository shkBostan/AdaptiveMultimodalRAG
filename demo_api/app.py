"""
FastAPI application for AdaptiveMultimodalRAG demo.

Author: s Bostan
Created on: Nov, 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import sys
import os
import time

# Add python directory to path
python_dir = os.path.join(os.path.dirname(__file__), '..', 'python')
sys.path.insert(0, python_dir)

# Initialize logging
from src.logging import setup_logging, get_logger, get_correlation_id

env = os.getenv("ENV", "dev")
setup_logging(env=env)
logger = get_logger(__name__)

from src.embeddings import BERTEmbedding
from src.retrieval import RetrievalEngine
from src.generation import RAGModule
from src.logging import CorrelationIDMiddleware

app = FastAPI(title="AdaptiveMultimodalRAG API", version="1.0.0")

# Add correlation ID middleware (should be first)
app.add_middleware(CorrelationIDMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models (lazy loading)
embedding_model = None
retrieval_engine = None
rag_module = None


def get_embedding_model():
    """Get or initialize embedding model."""
    global embedding_model
    if embedding_model is None:
        embedding_model = BERTEmbedding()
        embedding_model.load_model()
    return embedding_model


def get_retrieval_engine():
    """Get or initialize retrieval engine."""
    global retrieval_engine
    if retrieval_engine is None:
        retrieval_engine = RetrievalEngine(embedding_dim=768)
    return retrieval_engine


def get_rag_module():
    """Get or initialize RAG module."""
    global rag_module
    if rag_module is None:
        rag_module = RAGModule()
        rag_module.load_model()
    return rag_module


@app.get("/")
async def root(request: Request):
    """Root endpoint."""
    correlation_id = get_correlation_id()
    logger.info(
        "Root endpoint accessed",
        extra={"correlation_id": correlation_id}
    )
    return {"message": "AdaptiveMultimodalRAG API", "version": "1.0.0"}


@app.post("/api/query")
async def query(query: str, top_k: int = 5, request: Request = None):  # type: ignore
    """
    Process a text query and return RAG response.
    
    Args:
        query: Text query
        top_k: Number of top results to retrieve
        request: FastAPI request object
        
    Returns:
        JSON response with retrieved documents and generated answer
    """
    correlation_id = get_correlation_id()
    start_time = time.time()
    
    logger.info(
        "Query request received",
        extra={
            "correlation_id": correlation_id,
            "query_length": len(query),
            "top_k": top_k
        }
    )
    
    try:
        # Get embedding for query
        logger.debug(
            "Getting embedding model",
            extra={"correlation_id": correlation_id}
        )
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.get_embedding(query)
        
        logger.debug(
            "Query embedding generated",
            extra={
                "correlation_id": correlation_id,
                "embedding_dim": len(query_embedding)
            }
        )
        
        # Retrieve documents
        retrieval_engine = get_retrieval_engine()
        # TODO: Build index with documents first
        # results = retrieval_engine.search(query_embedding, top_k=top_k)
        
        # Generate response
        rag_module = get_rag_module()
        # response = rag_module.generate(query, results)
        
        duration = time.time() - start_time
        
        logger.info(
            "Query request completed",
            extra={
                "correlation_id": correlation_id,
                "duration_ms": duration * 1000,
                "top_k": top_k
            }
        )
        
        return JSONResponse({
            "query": query,
            "results": [],  # Placeholder
            "response": "Response generation not yet implemented"
        })
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error(
            "Query request failed",
            extra={
                "correlation_id": correlation_id,
                "duration_ms": duration * 1000,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...), request: Request = None):
    """
    Upload and process an image.
    
    Args:
        file: Image file
        request: FastAPI request object
        
    Returns:
        JSON response with image processing results
    """
    correlation_id = get_correlation_id()
    
    logger.info(
        "Image upload request received",
        extra={
            "correlation_id": correlation_id,
            "filename": file.filename,
            "content_type": file.content_type
        }
    )
    
    # TODO: Implement image processing
    logger.warning(
        "Image upload not yet implemented",
        extra={"correlation_id": correlation_id}
    )
    
    return JSONResponse({"message": "Image upload not yet implemented"})


@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...), request: Request = None):
    """
    Upload and process an audio file.
    
    Args:
        file: Audio file
        request: FastAPI request object
        
    Returns:
        JSON response with audio processing results
    """
    correlation_id = get_correlation_id()
    
    logger.info(
        "Audio upload request received",
        extra={
            "correlation_id": correlation_id,
            "filename": file.filename,
            "content_type": file.content_type
        }
    )
    
    # TODO: Implement audio processing
    logger.warning(
        "Audio upload not yet implemented",
        extra={"correlation_id": correlation_id}
    )
    
    return JSONResponse({"message": "Audio upload not yet implemented"})


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application", extra={"host": "0.0.0.0", "port": 8000})
    uvicorn.run(app, host="0.0.0.0", port=8000)

