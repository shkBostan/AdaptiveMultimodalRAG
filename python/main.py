"""
Main entry point for AdaptiveMultimodalRAG system.

Author: s Bostan
Created on: Nov, 2025
"""

import argparse
from embeddings.bert_embedding import BERTEmbedding
from retrieval.retrieval_engine import RetrievalEngine
from generation.rag_module import RAGModule


def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(description="AdaptiveMultimodalRAG System")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results")
    parser.add_argument("--model", type=str, default="bert-base-uncased", 
                       help="Embedding model name")
    
    args = parser.parse_args()
    
    # Initialize components
    print("Loading embedding model...")
    embedding_model = BERTEmbedding(model_name=args.model)
    embedding_model.load_model()
    
    print("Initializing retrieval engine...")
    retrieval_engine = RetrievalEngine(embedding_dim=768)
    
    print("Loading RAG module...")
    rag_module = RAGModule()
    rag_module.load_model()
    
    # Generate query embedding
    print(f"Processing query: {args.query}")
    query_embedding = embedding_model.get_embedding(args.query)
    
    # TODO: Load documents and build index
    # For now, this is a placeholder
    
    # Retrieve documents
    # results = retrieval_engine.search(query_embedding, top_k=args.top_k)
    
    # Generate response
    # response = rag_module.generate(args.query, results)
    # print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()

