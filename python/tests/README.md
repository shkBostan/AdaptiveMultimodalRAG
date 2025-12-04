# AdaptiveMultimodalRAG Test Suite

Comprehensive pytest test suite for the AdaptiveMultimodalRAG system.

## Test Structure

```
tests/
├── fixtures.py              # Shared fixtures and utilities
├── conftest.py              # Pytest configuration
├── test_document_loader.py  # DocumentLoader tests
├── test_index_builder.py    # IndexBuilder tests
├── test_retrieval_engine.py # RetrievalEngine tests
├── test_rag_module.py       # RAGModule tests
├── test_integration.py      # Full pipeline integration tests
└── data/                    # Test data directory (README only)
```

## Running Tests

### Run all tests
```bash
cd python
pytest
```

### Run specific test file
```bash
pytest tests/test_document_loader.py
```

### Run specific test class
```bash
pytest tests/test_document_loader.py::TestDocumentLoader
```

### Run specific test method
```bash
pytest tests/test_document_loader.py::TestDocumentLoader::test_load_from_txt_file
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage (if pytest-cov installed)
```bash
pytest --cov=python --cov-report=html
```

## Test Categories

### 1. DocumentLoader Tests (`test_document_loader.py`)
- TXT file loading
- JSON file loading (list and dict formats)
- CSV file loading
- Mixed file type loading
- Text normalization
- Character and token-based chunking
- Document conversion and metadata handling

### 2. IndexBuilder Tests (`test_index_builder.py`)
- Index initialization (flat_l2, ivf_flat)
- Embedding shape validation
- Index building with different types
- Search functionality
- Save/load operations
- Incremental updates (add_embeddings)
- Utility function validation

### 3. RetrievalEngine Tests (`test_retrieval_engine.py`)
- Legacy mode initialization
- IndexBuilder mode initialization
- Document loading
- Index building (legacy and new modes)
- Search with different return formats
- Incremental document addition
- Statistics retrieval

### 4. RAGModule Tests (`test_rag_module.py`)
- Module initialization
- Context preparation with different formats
- Metadata marker formatting
- Length control and truncation
- Generation with mocked models
- Empty context fallback
- Answer extraction

### 5. Integration Tests (`test_integration.py`)
- Full pipeline: load → embed → index → retrieve → generate
- Pipeline with mocked components
- Pipeline with synthetic data
- Persistence integration
- Error handling
- Consistency checks

## Test Features

- **No External Dependencies**: All tests use mocks to avoid downloading models
- **Fast Execution**: All tests run in under 2 seconds
- **Comprehensive Coverage**: Tests cover all major components and edge cases
- **Synthetic Data**: Test data is generated on-the-fly using fixtures
- **Isolated Tests**: Each test is independent and can run in any order

## Fixtures

Key fixtures available in `fixtures.py`:

- `temp_dir`: Temporary directory for test files
- `sample_text_files`: Sample TXT files
- `sample_json_file_list`: JSON file with list format
- `sample_json_file_dict`: JSON file with dict format
- `sample_csv_file`: Sample CSV file
- `sample_embeddings`: Synthetic embedding array
- `sample_query_embedding`: Synthetic query embedding
- `sample_documents`: Sample Document objects
- `mock_embedding_model`: Mock embedding model
- `mock_faiss_index`: Mock FAISS index
- `mock_rag_model`: Mock RAG generation model

## Requirements

- pytest
- numpy
- unittest.mock (standard library)

## Notes

- Tests use `unittest.mock` to avoid external API calls
- Synthetic embeddings use fixed random seed for reproducibility
- All file I/O uses temporary directories that are cleaned up automatically
- Tests are designed to run quickly without network access

## Troubleshooting

### Import Errors
If you see import errors, ensure you're running tests from the `python/` directory:
```bash
cd python
pytest
```

### Missing Dependencies
Install required packages:
```bash
pip install pytest numpy
```

### Test Failures
Run with verbose output to see detailed error messages:
```bash
pytest -v -s
```

