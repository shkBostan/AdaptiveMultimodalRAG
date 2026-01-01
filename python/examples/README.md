# RAG Pipeline Examples

This directory contains example scripts demonstrating how to use the RAG pipeline.

## Quick Start

### Example 1: Run RAG Pipeline and See Results

The main example script shows how to:
- Load documents
- Run the complete RAG pipeline
- Inspect returned results
- Perform retrieval
- Generate responses
- Analyze embeddings

**Run the example:**
```bash
cd python
python examples/run_rag_pipeline_example.py
```

## What the Pipeline Returns

When you run `execute_full_rag_pipeline()`, it returns a tuple with 4 components:

```python
embedding_model, retrieval_engine, rag_module, embeddings_array = execute_full_rag_pipeline(...)
```

### 1. `embedding_model`
- Type: `BERTEmbedding`, `Word2VecModel`, `CLIPImageEmbedding`, or `MultimodalFusion`
- Purpose: Used to generate embeddings for queries and new documents
- Methods:
  - `get_embedding(text)` - Generate embedding for single text
  - `get_embeddings_batch(texts)` - Generate embeddings for batch
  - `embedding_dim` - Get embedding dimension

### 2. `retrieval_engine`
- Type: `RetrievalEngine`
- Purpose: Search for similar documents using vector similarity
- Methods:
  - `search(query_embedding, top_k=10)` - Search for similar documents
  - `build_index(embeddings, documents)` - Build search index (already called)

### 3. `rag_module`
- Type: `RAGModule`
- Purpose: Generate responses using retrieved context
- Methods:
  - `generate(query, retrieved_docs, max_length=512)` - Generate response

### 4. `embeddings_array`
- Type: `numpy.ndarray`
- Shape: `(n_documents, embedding_dim)`
- Purpose: Pre-computed embeddings for all documents
- Use: Can be saved, analyzed, or used for batch operations

## Usage Patterns

### Pattern 1: Simple Query-Response

```python
from src.pipeline.rag_pipeline import execute_full_rag_pipeline
from src.retrieval import DocumentLoader
import logging

# Setup
logger = logging.getLogger(__name__)
config = {
    'embeddings': {'model_type': 'bert', 'model_name': 'bert-base-uncased'},
    'retrieval': {'similarity_metric': 'cosine'},
    'generation': {'model_type': 'gpt2'}
}

# Load documents and run pipeline
loader = DocumentLoader()
documents = loader.load_from_directory('experiments/datasets/text')
embedding_model, retrieval_engine, rag_module, embeddings = execute_full_rag_pipeline(
    config, documents, logger
)

# Query and get response
query = "What is machine learning?"
query_embedding = embedding_model.get_embedding(query)
results = retrieval_engine.search(query_embedding, top_k=5)
response = rag_module.generate(query, results)
print(response)
```

### Pattern 2: Batch Retrieval

```python
# Process multiple queries
queries = ["machine learning", "deep learning", "neural networks"]
for query in queries:
    query_embedding = embedding_model.get_embedding(query)
    results = retrieval_engine.search(query_embedding, top_k=3)
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents")
```

### Pattern 3: Save and Reuse Embeddings

```python
import numpy as np

# Save embeddings
np.save('embeddings.npy', embeddings_array)

# Later, load embeddings
embeddings_array = np.load('embeddings.npy')
print(f"Loaded embeddings: {embeddings_array.shape}")
```

## Expected Output

When you run the example script, you'll see:

1. **Pipeline Execution Logs**: Progress through embedding, retrieval, and generation stages
2. **Results Summary**: Overview of all returned components
3. **Retrieval Demo**: Shows top similar documents for a query
4. **Generation Demo**: Shows generated response using retrieved context
5. **Embedding Analysis**: Similarity statistics between documents

## Troubleshooting

- **Import Errors**: Ensure you're in the `python/` directory or have added it to `PYTHONPATH`
- **Model Downloads**: Models download on first use; ensure internet connection
- **Memory Issues**: Reduce `batch_size` in config if you run out of memory
- **Missing Documents**: The script creates sample documents if the dataset path doesn't exist

