"""
Main entry point for AdaptiveMultimodalRAG system.

Author: s Bostan
Created on: Nov, 2025
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to path for data imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from embeddings.bert_embedding import BERTEmbedding
from retrieval.retrieval_engine import RetrievalEngine
from generation.rag_module import RAGModule
from data.document_loader import DocumentLoader


def main():
    """Main function to run the RAG system."""
    parser = argparse.ArgumentParser(description="AdaptiveMultimodalRAG System")
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results")
    parser.add_argument("--model", type=str, default="bert-base-uncased", 
                       help="Embedding model name")
    parser.add_argument("--documents", type=str, default="data/text",
                       help="Path to documents directory or file (default: data/text)")
    parser.add_argument("--doc_type", type=str, choices=["txt", "json", "csv", "mixed"],
                       default="txt", help="Document type (default: txt)")
    
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
    
    # Load documents and build index
    print(f"\nLoading documents from: {args.documents}")
    try:
        loader = DocumentLoader()
        
        # Load documents based on type
        if args.doc_type == "txt":
            documents = loader.load_from_directory(args.documents)
        elif args.doc_type == "json":
            documents = loader.load_from_json(args.documents)
        elif args.doc_type == "csv":
            documents = loader.load_from_csv(args.documents)
        elif args.doc_type == "mixed":
            documents = loader.load_from_mixed(args.documents, recursive=True)
        else:
            raise ValueError(f"Unknown document type: {args.doc_type}")
        
        if not documents:
            print("Warning: No documents loaded. Please check the document path.")
            return
        
        print(f"Loaded {len(documents)} documents")
        
        # Generate embeddings for all documents
        print("Generating document embeddings...")
        embeddings_list = []
        for i, doc in enumerate(documents):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(documents)} documents...")
            embedding = embedding_model.get_embedding(doc.content)
            embeddings_list.append(embedding)
        
        embeddings = np.array(embeddings_list)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Convert documents to dictionary format for RetrievalEngine
        doc_dicts = [doc.to_dict() for doc in documents]
        
        # Build index
        print("Building retrieval index...")
        retrieval_engine.build_index(embeddings, doc_dicts)
        print("Index built successfully!")
        
    except Exception as e:
        print(f"Error loading documents: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate query embedding
    print(f"\nProcessing query: {args.query}")
    query_embedding = embedding_model.get_embedding(args.query)
    
    # Retrieve documents
    print(f"Searching for top {args.top_k} results...")
    results = retrieval_engine.search(query_embedding, top_k=args.top_k)
    print(f"Found {len(results)} results")
    
    # Display retrieved documents
    print("\nRetrieved Documents:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result.get('score', 'N/A'):.4f}")
        print(f"    ID: {result.get('id', 'N/A')}")
        content_preview = result.get('content', '')[:150]
        print(f"    Content: {content_preview}...")
    
    # Generate response
    print("\n" + "=" * 70)
    print("Generating response...")
    response = rag_module.generate(args.query, results)
    print("\n" + "=" * 70)
    print("RESPONSE:")
    print("=" * 70)
    print(response)
    print("=" * 70)


if __name__ == "__main__":
    main()

