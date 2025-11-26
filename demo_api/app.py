"""
FastAPI application for AdaptiveMultimodalRAG demo.

Author: s Bostan
Created on: Nov, 2025
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import sys
import os

# Add python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

from embeddings.bert_embedding import BERTEmbedding
from retrieval.retrieval_engine import RetrievalEngine
from generation.rag_module import RAGModule

app = FastAPI(title="AdaptiveMultimodalRAG API", version="1.0.0")

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
async def root():
    """Root endpoint."""
    return {"message": "AdaptiveMultimodalRAG API", "version": "1.0.0"}


@app.post("/api/query")
async def query(query: str, top_k: int = 5):
    """
    Process a text query and return RAG response.
    
    Args:
        query: Text query
        top_k: Number of top results to retrieve
        
    Returns:
        JSON response with retrieved documents and generated answer
    """
    try:
        # Get embedding for query
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.get_embedding(query)
        
        # Retrieve documents
        retrieval_engine = get_retrieval_engine()
        # TODO: Build index with documents first
        # results = retrieval_engine.search(query_embedding, top_k=top_k)
        
        # Generate response
        rag_module = get_rag_module()
        # response = rag_module.generate(query, results)
        
        return JSONResponse({
            "query": query,
            "results": [],  # Placeholder
            "response": "Response generation not yet implemented"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and process an image.
    
    Args:
        file: Image file
        
    Returns:
        JSON response with image processing results
    """
    # TODO: Implement image processing
    return JSONResponse({"message": "Image upload not yet implemented"})


@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload and process an audio file.
    
    Args:
        file: Audio file
        
    Returns:
        JSON response with audio processing results
    """
    # TODO: Implement audio processing
    return JSONResponse({"message": "Audio upload not yet implemented"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

