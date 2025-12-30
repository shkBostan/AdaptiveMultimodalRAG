# Test Suite for AdaptiveMultimodalRAG

This directory contains comprehensive pytest test suites for the AdaptiveMultimodalRAG project.

## Overview

The test suite is organized into modular test files, each testing specific components:

- **`test_embeddings.py`**: Tests for embedding models (BERT, CLIP, MultimodalFusion)
- **`test_rag_pipeline.py`**: Tests for RAG pipeline stages and full pipeline execution
- **`conftest.py`**: Shared pytest fixtures for common test data and configurations

## Structure

```
tests/
├── conftest.py              # Shared fixtures (configs, sample data, models)
├── test_embeddings.py       # Embedding component tests
├── test_rag_pipeline.py     # Pipeline stage and integration tests
└── README.md                # This file
```

## Running Tests

### Run All Tests

```bash
cd python
pytest tests/
```

### Run Specific Test File

```bash
# Test embeddings only
pytest tests/test_embeddings.py

# Test RAG pipeline only
pytest tests/test_rag_pipeline.py
```

### Run Specific Test Function

```bash
# Test BERT embedding initialization
pytest tests/test_embeddings.py::test_bert_embedding_initialization

# Test full RAG pipeline
pytest tests/test_rag_pipeline.py::test_full_rag_pipeline
```

### Run with Verbose Output

```bash
pytest tests/ -v
```

### Run with Logging Output

```bash
pytest tests/ -v -s
```

### Run Only Smoke Tests

```bash
pytest tests/ -m smoke
```

## Test Coverage

### Embedding Tests (`test_embeddings.py`)

#### BERT Embedding Tests
- ✅ Initialization with default and custom parameters
- ✅ Single text embedding generation
- ✅ Batch text embedding generation
- ✅ Different pooling strategies (CLS, MEAN, MAX)
- ✅ L2 normalization option

#### CLIP Image Embedding Tests
- ✅ Initialization
- ✅ Embedding from image file path
- ✅ Embedding from PIL Image object
- ✅ Batch image embedding generation
- ✅ L2 normalization option

#### Multimodal Fusion Tests
- ✅ Initialization with different strategies (concatenation, weighted_sum, attention)
- ✅ Concatenation fusion
- ✅ Weighted sum fusion
- ✅ Attention fusion
- ✅ Batch fusion
- ✅ Error handling (mismatched dimensions, empty inputs)

### RAG Pipeline Tests (`test_rag_pipeline.py`)

#### Embedding Pipeline Stage Tests
- ✅ `run_embedding_pipeline` for BERT model initialization
- ✅ `run_embedding_pipeline` for multimodal fusion initialization
- ✅ `run_full_embedding_pipeline` for BERT embeddings generation
- ✅ Batch processing with `run_full_embedding_pipeline`

#### Retrieval Pipeline Stage Tests
- ✅ `run_retrieval_pipeline` for retrieval engine initialization
- ✅ Index building and search functionality

#### Generation Pipeline Stage Tests
- ✅ `run_generation_pipeline` for RAG module initialization
- ✅ RAG module text generation with retrieved context

#### Full Pipeline Integration Tests
- ✅ Full RAG pipeline execution (smoke test)
- ✅ Full pipeline with retrieval and generation
- ✅ Error handling for empty documents
- ✅ Output shape validation

## Test Fixtures (`conftest.py`)

The `conftest.py` file provides reusable fixtures for all tests:

### Data Fixtures
- `sample_texts`: Sample text strings for embedding tests
- `sample_documents`: Sample Document objects for pipeline tests
- `sample_documents_with_images`: Documents with image paths
- `dummy_image_path`: Dummy image file path
- `dummy_pil_image`: PIL Image object

### Configuration Fixtures
- `bert_config`: BERT embedding configuration
- `clip_config`: CLIP embedding configuration
- `multimodal_config`: Multimodal fusion configuration
- `full_rag_config`: Full RAG pipeline configuration

### Model Fixtures
- `bert_embedding_model`: Initialized BERT embedding model with loaded weights

### Embedding Fixtures
- `sample_embeddings_text`: Sample text embeddings (768-dim)
- `sample_embeddings_image`: Sample image embeddings (512-dim)
- `sample_embeddings_dict`: Dictionary of embeddings for fusion tests
- `sample_embeddings_dict_batch`: Dictionary of batch embeddings

### Utility Fixtures
- `logger`: Configured logger for tests

## Writing New Tests

### Adding a New Test Function

1. Import necessary modules:
```python
import pytest
from src.embeddings import BERTEmbedding
```

2. Use fixtures from `conftest.py`:
```python
def test_my_feature(logger, sample_documents):
    # Your test code
    pass
```

3. Add assertions and logging:
```python
def test_my_feature(logger, sample_documents):
    logger.info("Testing my feature...")
    
    # Test code
    result = some_function()
    
    # Assertions
    assert result is not None
    assert result.shape == expected_shape
    
    logger.info("✓ Test passed")
```

### Best Practices

1. **Use fixtures**: Reuse fixtures from `conftest.py` rather than creating test data inline
2. **Add logging**: Include informative log messages to trace test execution
3. **Test independently**: Each test should be able to run independently
4. **Validate outputs**: Check shapes, types, and values of outputs
5. **Test edge cases**: Include error handling and edge case tests
6. **Keep tests fast**: Use minimal data and lightweight models

## Requirements

Tests require the following dependencies (should be in `requirements.txt`):

- `pytest`: Test framework
- `pytest-mock`: Mocking support (optional)
- `numpy`: Numerical operations
- `Pillow`: Image processing (for CLIP tests)
- `torch`: PyTorch (for model loading)
- `transformers`: HuggingFace transformers (for models)
- `faiss-cpu` or `faiss-gpu`: Vector search (for retrieval tests)

## Notes

- Tests use lightweight models (bert-base-uncased, clip-vit-base-patch32, gpt2) to keep execution fast
- Models are downloaded on first use (requires internet connection)
- Tests create temporary files and clean them up automatically
- The smoke test (`@pytest.mark.smoke`) can be run quickly to verify basic functionality

## Troubleshooting

### Import Errors
- Ensure you're in the `python/` directory when running tests
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Model Download Issues
- Ensure internet connection for first-time model downloads
- Models are cached after first download

### Memory Issues
- Reduce `batch_size` in test configurations if running out of memory
- Use smaller models or fewer test documents

### Test Failures
- Run tests with `-v -s` flags to see detailed output
- Check that all required dependencies are installed
- Verify that test data files exist (if any)

## Author

**Author**: s Bostan  
**Created on**: Dec, 2025

