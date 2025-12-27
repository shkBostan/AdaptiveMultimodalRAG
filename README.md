# AdaptiveMultimodalRAG

**Adaptive Multimodal Retrieval-Augmented Generation for Reliable Decision-Centric Systems**

A multimodal Retrieval-Augmented Generation (RAG) system that adaptively handles text, images, and audio inputs to provide intelligent, context-aware responses for decision support systems. The system combines retrieval-based search with generative AI capabilities to deliver reliable, evidence-based decision-making support.

> **Note:** This is a prototype project and is not fully implemented. Many components are in development stages with placeholder implementations and TODOs throughout the codebase.

## Overview

AdaptiveMultimodalRAG is designed to enhance Large Language Models (LLMs) by incorporating real-time retrieval of relevant information from multimodal sources. The system is specifically architected for **decision-centric applications** where reliability, accuracy, and traceability are critical. It processes and understands text, images, and audio inputs, generating embeddings that can be fused together to create comprehensive multimodal representations for improved retrieval and generation tasks.

### Key Focus: Decision-Centric Systems

This system emphasizes:
- **Reliability**: Comprehensive evaluation metrics and error handling for production decision support
- **Traceability**: Full audit trails with correlation IDs and structured logging
- **Evidence-Based Decisions**: Retrieval of relevant context before generation ensures decisions are grounded in available information
- **Multimodal Understanding**: Support for text, image, and audio inputs enables richer decision contexts
- **Adaptive Retrieval**: Configurable similarity search and fusion strategies adapt to different decision scenarios

## Quick Start

### Prerequisites

- Python 3.10+ (Python 3.13 recommended)
- pip package manager
- (Optional) Java 11+ for Java components

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AdaptiveMultimodalRAG
   ```

2. **Set up Python environment:**
   
   **Windows (PowerShell):**
   ```powershell
   .\scripts\setup_python.ps1
   ```
   
   **Windows (Command Prompt):**
   ```cmd
   scripts\setup_python.bat
   ```
   
   **Linux/Mac:**
   ```bash
   bash scripts/setup_python.sh
   ```

3. **Activate virtual environment:**
   
   **Windows:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

4. **Verify installation:**
   ```bash
   cd python
   python -c "from src.embeddings import BERTEmbedding; print('Installation successful!')"
   ```

For detailed environment setup instructions, see [Environment Setup Guide](docs/setup.md).

## Key Components

### 1. **Embedding Generation**
- **Text Embeddings**: BERT-based and Word2Vec models for semantic text representation
- **Image Embeddings**: Support for image processing and embedding generation (structure in place)
- **Audio Embeddings**: Audio processing pipeline for speech/audio content (structure in place)
- **Multimodal Fusion**: Strategies for combining embeddings across different modalities (concatenation, weighted sum, attention-based)

### 2. **Retrieval Engine**
- Vector-based similarity search using FAISS
- Configurable indexing strategies (L2 distance, cosine similarity)
- Support for dynamic document indexing and updates
- Top-k retrieval with relevance scoring

### 3. **RAG Generation Module**
- Integration with transformer-based language models (GPT-2 as base)
- Context construction from retrieved documents
- Query-aware response generation
- Support for multiple generation strategies

### 4. **Production Infrastructure**
- **Comprehensive Logging System**: Structured logging with correlation IDs across Java, Python, and frontend
- **Environment Configuration**: Separate configurations for development, production, and testing
- **API Layer**: FastAPI-based RESTful API for system integration
- **Modern Web UI**: React/TypeScript frontend with Tailwind CSS for interactive exploration

### 5. **Multi-Language Support**
- **Python**: Primary implementation with full feature set
- **Java**: Complementary implementation for enterprise integration
- Interoperability between language implementations

## Technology Stack

### Backend
- **Python**: PyTorch, Transformers, FAISS, FastAPI, Uvicorn
- **Java**: SLF4J + Log4j2 for logging, Maven for build management
- **Vector Database**: FAISS for efficient similarity search
- **ML Frameworks**: Hugging Face Transformers, Sentence Transformers

### Frontend
- **React**: Modern UI framework with TypeScript
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework

### Infrastructure
- **Logging**: Production-grade structured logging (JSON in production)
- **Configuration**: YAML-based configuration management
- **Containerization**: Docker support (Dockerfile included)

## Project Structure

```
AdaptiveMultimodalRAG/
├── python/                      # Python implementation
│   ├── src/                    # Core source code modules
│   │   ├── embeddings/         # Embedding models (BERT, Word2Vec, multimodal fusion)
│   │   ├── generation/         # RAG generation module
│   │   ├── retrieval/          # Retrieval engine and search utilities
│   │   └── utils/              # Utilities (preprocessing, evaluation, logging)
│   ├── evaluation/             # Evaluation framework
│   │   ├── metrics/            # Retrieval and generation metrics
│   │   └── run_eval.py         # Evaluation runner
│   ├── experiments/            # Experiment data and results
│   │   ├── dataset1/           # Dataset 1 experiments
│   │   ├── dataset2/           # Dataset 2 experiments
│   │   └── results/            # Experiment results
│   ├── configs/                # Experiment configurations (YAML)
│   ├── scripts/                # Utility scripts
│   │   └── run_pipeline.py     # Main pipeline runner
│   ├── notebooks/              # Jupyter notebooks for experimentation
│   ├── tests/                  # Test suite
│   ├── middleware/             # FastAPI middleware (correlation IDs)
│   └── storage/                # Storage and persistence
├── java/                       # Java implementation
│   └── src/main/java/com/adaptiveRAG/
│       ├── embeddings/         # Java embedding classes
│       ├── retrieval/          # Retrieval engine
│       ├── generation/         # RAG module
│       └── utils/               # Utilities and logging
├── demo_api/                   # FastAPI demo application
├── ui/web/                     # React frontend application
├── config/                     # Environment configurations
├── scripts/                    # Setup and utility scripts
├── docs/                       # Documentation (including logging system)
├── requirements.txt            # Python dependencies
├── ENVIRONMENT_SETUP.md        # Environment setup guide
└── logs/                       # Log files directory
```

For detailed Python package structure, see [Python Package Structure](docs/python-structure.md).

## Features

### Core Capabilities
- ✅ **Multimodal Embedding Generation**: Text, image, and audio embedding support for comprehensive context understanding
- ✅ **Adaptive Retrieval Engine**: Configurable similarity search with FAISS and ChromaDB for reliable information retrieval
- ✅ **RAG-based Generation**: Context-aware response generation with transformer models, ensuring evidence-based outputs
- ✅ **Multimodal Fusion**: Multiple strategies for combining embeddings (weighted average, concatenation, attention-based) to support diverse decision scenarios
- ✅ **Comprehensive Evaluation**: Retrieval metrics (Precision@K, Recall@K, MRR, NDCG) and generation metrics (BLEU, ROUGE, BERTScore) for reliability assessment
- ✅ **Decision Support Features**: Structured logging, correlation tracking, and evaluation frameworks for decision-centric applications

### Infrastructure
- ✅ **RESTful API**: FastAPI-based API for easy integration
- ✅ **Modern Web UI**: Interactive React-based interface with TypeScript
- ✅ **Production Logging**: Comprehensive logging with correlation IDs across Java, Python, and frontend
- ✅ **Experiment Management**: YAML-based configuration system for reproducible experiments
- ✅ **Pipeline Scripts**: Automated pipeline runner for end-to-end experiments

### Development Status
- ✅ **Python Implementation**: Fully structured with modular design
- ✅ **Java Implementation**: Complementary implementation for enterprise integration
- ✅ **Test Suite**: Comprehensive test coverage for core components
- ⚠️ **Work in Progress**: Image and audio processing pipelines are partially implemented
- ⚠️ **Work in Progress**: Some fusion strategies are placeholders
- ⚠️ **Work in Progress**: Document indexing and storage backend needs completion

## Usage Examples

### Running an Experiment

Run a complete pipeline experiment using a configuration file:

```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate      # Linux/Mac

# Run experiment (from project root)
cd python
python scripts/run_pipeline.py --config configs/exp1.yaml
```

### Running Evaluation

Evaluate experiment results:

**Linux/Mac:**
```bash
cd python
python evaluation/run_eval.py \
    --config configs/exp1.yaml \
    --results experiments/results \
    --output experiments/results/evaluation_results.json
```

**Windows (PowerShell):**
```powershell
cd python
python evaluation/run_eval.py `
    --config configs/exp1.yaml `
    --results experiments/results `
    --output experiments/results/evaluation_results.json
```

**Windows (Command Prompt):**
```cmd
cd python
python evaluation/run_eval.py --config configs/exp1.yaml --results experiments/results --output experiments/results/evaluation_results.json
```

### Using in Python Code

```python
# Make sure you're in the python/ directory or have it in your path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

from src.embeddings import BERTEmbedding, MultimodalFusion
from src.retrieval import RetrievalEngine
from src.generation import RAGModule

# Initialize embedding model
embedding = BERTEmbedding(model_name='bert-base-uncased', batch_size=32)

# Initialize retrieval engine
retrieval = RetrievalEngine(method='faiss', similarity_metric='cosine', top_k=10)

# Initialize RAG module
rag = RAGModule(retrieval_engine=retrieval, model_type='gpt2')

# Use the system
query = "What is machine learning?"
results = rag.generate(query)
```

### Jupyter Notebooks

Interactive experimentation is available through Jupyter notebooks:

```bash
cd python/notebooks
jupyter notebook demo_experiments.ipynb
```

## Configuration

Experiments are configured using YAML files in `python/configs/`:

- **exp1.yaml**: Baseline retrieval experiment with BERT embeddings
- **exp2.yaml**: Multimodal fusion experiment with text, image, and audio

See the configuration files for detailed parameter descriptions.

## Testing

Run the test suite:

```bash
# From project root
cd python
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Decision-Centric Applications

This system is designed for applications where reliable, evidence-based decision-making is critical:

- **Enterprise Decision Support**: Business intelligence and strategic planning with multimodal context
- **Healthcare Decision Systems**: Medical information retrieval and evidence-based recommendations
- **Financial Analysis**: Multi-source financial data analysis and risk assessment
- **Legal Research**: Case law and document retrieval for legal decision support
- **Technical Documentation**: Engineering and technical decision support with multimodal inputs

## Current Status

This prototype demonstrates the architecture and core concepts of a multimodal RAG system for decision-centric applications. While the foundational components are in place, several features require further development:

- ✅ **Completed**: Core structure, embedding models, retrieval engine, evaluation framework
- ✅ **Completed**: Experiment configuration system, pipeline scripts, notebooks
- ✅ **Completed**: Structured logging and correlation tracking for decision traceability
- ⚠️ **In Progress**: Complete image and audio embedding implementations
- ⚠️ **In Progress**: Full multimodal fusion strategies
- ⚠️ **In Progress**: Document indexing and storage backend
- ⚠️ **Planned**: Production deployment configurations for decision systems
- ⚠️ **Planned**: Performance optimization and benchmarking for real-time decision support

## Documentation

### Getting Started
- [Environment Setup Guide](docs/setup.md) - Detailed Python environment setup instructions
- [Quick Start (Persian)](docs/fa/QUICK_START.md) - راهنمای سریع شروع (فارسی)

### Architecture & Structure
- [Project Architecture](docs/architecture.md) - Complete architecture and refactoring history
- [Python Package Structure](docs/python-structure.md) - Detailed Python package documentation
- [Decision-Centric Design](docs/decision-centric.md) - Design principles for decision support systems

### System Components
- [Logging System](docs/logging/README.md) - Comprehensive logging system documentation
- [Logging Architecture](docs/logging/ARCHITECTURE.md) - Detailed logging architecture

### Persian Documentation (فارسی)
- [راهنمای کامل پروژه](docs/fa/README.md) - Complete project guide in Persian
- [راهنمای سریع شروع](docs/fa/QUICK_START.md) - Quick start guide in Persian

## Contributing

This is a research/prototype project. Contributions and feedback are welcome!

## License

See LICENSE file for details.

## Author

**s Bostan**  
Created on: Nov, 2025
