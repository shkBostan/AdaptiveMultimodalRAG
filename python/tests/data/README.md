# Test Data Directory

This directory contains synthetic test data samples for AdaptiveMultimodalRAG tests.

## Files

- `sample_text.txt` - Sample text file for DocumentLoader tests
- `sample_documents.json` - Sample JSON file with document list
- `sample_documents_dict.json` - Sample JSON file with document dictionary
- `sample_documents.csv` - Sample CSV file with documents

## Usage

These files are created dynamically by pytest fixtures in `fixtures.py`.
No manual data files are required - all test data is generated on-the-fly.

