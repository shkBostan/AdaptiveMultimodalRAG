"""
Persistence subsystem for AdaptiveMultimodalRAG.

Provides versioned storage for documents, embeddings, and FAISS indices,
enabling fast repeated runs without rebuilding embeddings each time.

Author: s Bostan
Created on: Nov, 2025
"""

import sys
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import numpy as np

# Add python directory to path
python_dir = Path(__file__).parent.parent
sys.path.insert(0, str(python_dir))

from src.retrieval.index_builder import IndexBuilder
from src.retrieval.document_loader import Document
from src.logging import get_logger


# Version constants
STORAGE_VERSION = 1
DOCUMENTS_VERSION = 1
EMBEDDINGS_VERSION = 1


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
        
    Returns:
        Path object to the directory
        
    Raises:
        OSError: If directory cannot be created
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def validate_file_path(filepath: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate and normalize file path.
    
    Args:
        filepath: Path to file
        must_exist: Whether file must exist
        
    Returns:
        Normalized Path object
        
    Raises:
        ValueError: If path is invalid
        FileNotFoundError: If must_exist=True and file doesn't exist
    """
    path = Path(filepath)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.exists() and path.is_dir():
        raise ValueError(f"Path is a directory, not a file: {path}")
    
    return path


class PersistenceManager:
    """
    Manages persistence of documents, embeddings, and indices.
    
    Provides versioned storage with automatic directory creation and validation.
    Fully compatible with DocumentLoader and IndexBuilder.
    
    Attributes:
        storage_version: Version of storage format
        default_storage_dir: Default directory for storage operations
    """
    
    def __init__(self, default_storage_dir: Union[str, Path] = "storage"):
        """
        Initialize persistence manager.
        
        Args:
            default_storage_dir: Default directory for storage operations
        """
        self.storage_version = STORAGE_VERSION
        self.default_storage_dir = Path(default_storage_dir)
        self.logger = get_logger(__name__)
        
        # Ensure default directory exists
        ensure_dir(self.default_storage_dir)
    
    def save_documents(
        self,
        documents: List[Document],
        filepath: Union[str, Path],
        include_embeddings: bool = False
    ) -> None:
        """
        Save documents to JSON file with versioning.
        
        Args:
            documents: List of Document objects to save
            filepath: Path to save JSON file
            include_embeddings: Whether to include embeddings in saved documents
                (embeddings can be large, set to False to save space)
        
        Raises:
            ValueError: If documents list is empty or invalid
            OSError: If file cannot be written
        """
        if not documents:
            raise ValueError("Cannot save empty documents list")
        
        filepath = Path(filepath)
        ensure_dir(filepath.parent)
        
        self.logger.info(
            f"Saving {len(documents)} documents to {filepath}",
            extra={
                "num_documents": len(documents),
                "include_embeddings": include_embeddings
            }
        )
        
        # Convert documents to serializable format
        documents_data = []
        for doc in documents:
            if hasattr(doc, 'to_dict'):
                doc_dict = doc.to_dict()
            elif isinstance(doc, dict):
                doc_dict = doc.copy()
            else:
                self.logger.warning(f"Skipping invalid document type: {type(doc)}")
                continue
            
            # Optionally remove embeddings to save space
            if not include_embeddings and 'embedding' in doc_dict:
                doc_dict.pop('embedding', None)
            
            documents_data.append(doc_dict)
        
        # Create versioned storage format
        storage_data = {
            "version": DOCUMENTS_VERSION,
            "storage_version": self.storage_version,
            "timestamp": datetime.utcnow().isoformat(),
            "num_documents": len(documents_data),
            "include_embeddings": include_embeddings,
            "data": documents_data
        }
        
        # Save to JSON
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(
                f"Documents saved successfully to {filepath}",
                extra={
                    "file_size_bytes": filepath.stat().st_size,
                    "num_documents": len(documents_data)
                }
            )
        except Exception as e:
            self.logger.error(
                f"Failed to save documents to {filepath}",
                extra={"error": str(e), "filepath": str(filepath)},
                exc_info=True
            )
            raise
    
    def load_documents(self, filepath: Union[str, Path]) -> List[Document]:
        """
        Load documents from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of Document objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid or version incompatible
            json.JSONDecodeError: If JSON is malformed
        """
        filepath = validate_file_path(filepath, must_exist=True)
        
        self.logger.info(f"Loading documents from {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                storage_data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Invalid JSON in {filepath}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise
        
        # Validate version
        if not isinstance(storage_data, dict) or 'version' not in storage_data:
            raise ValueError(f"Invalid storage format in {filepath}: missing version")
        
        file_version = storage_data.get('version', 0)
        if file_version > DOCUMENTS_VERSION:
            raise ValueError(
                f"File version {file_version} is newer than supported version {DOCUMENTS_VERSION}"
            )
        
        # Extract documents
        if 'data' not in storage_data:
            raise ValueError(f"Invalid storage format in {filepath}: missing data field")
        
        documents_data = storage_data['data']
        
        if not isinstance(documents_data, list):
            raise ValueError(f"Invalid storage format in {filepath}: data is not a list")
        
        self.logger.debug(
            f"Loaded {len(documents_data)} documents from version {file_version} file"
        )
        
        # Convert to Document objects
        documents = []
        from src.retrieval.document_loader import Document as DocClass
            
            for doc_dict in documents_data:
                try:
                    # Extract fields
                    doc_id = doc_dict.get('id', '')
                    content = doc_dict.get('content', '')
                    metadata = {k: v for k, v in doc_dict.items() 
                              if k not in ['id', 'content', 'embedding', 'image_path', 'audio_path']}
                    embedding = doc_dict.get('embedding')
                    image_path = doc_dict.get('image_path')
                    audio_path = doc_dict.get('audio_path')
                    
                    doc = DocClass(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                        embedding=embedding,
                        image_path=image_path,
                        audio_path=audio_path
                    )
                    documents.append(doc)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to reconstruct document: {e}",
                        extra={"doc_dict_keys": list(doc_dict.keys())}
                    )
                    continue
        else:
            # Fallback: return as dictionaries
            documents = documents_data
        
        self.logger.info(
            f"Successfully loaded {len(documents)} documents",
            extra={"filepath": str(filepath), "num_documents": len(documents)}
        )
        
        return documents
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save embeddings to pickle file with versioning.
        
        Args:
            embeddings: Array of embeddings, shape [N, D]
            filepath: Path to save pickle file
            metadata: Optional metadata to store with embeddings
                (e.g., model_name, embedding_dim, timestamp)
        
        Raises:
            ValueError: If embeddings are invalid
            OSError: If file cannot be written
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError(f"Embeddings must be numpy array, got {type(embeddings)}")
        
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array [N, D], got shape {embeddings.shape}")
        
        filepath = Path(filepath)
        ensure_dir(filepath.parent)
        
        num_vectors, embedding_dim = embeddings.shape
        
        self.logger.info(
            f"Saving embeddings to {filepath}",
            extra={
                "num_vectors": num_vectors,
                "embedding_dim": embedding_dim,
                "shape": embeddings.shape
            }
        )
        
        # Prepare storage data
        storage_data = {
            "version": EMBEDDINGS_VERSION,
            "storage_version": self.storage_version,
            "timestamp": datetime.utcnow().isoformat(),
            "num_vectors": num_vectors,
            "embedding_dim": embedding_dim,
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "metadata": metadata or {}
        }
        
        # Save with pickle (embeddings) and JSON (metadata)
        try:
            # Save embeddings as pickle
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata as JSON (same name with .meta.json extension)
            meta_filepath = filepath.with_suffix('.meta.json')
            with open(meta_filepath, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2)
            
            self.logger.info(
                f"Embeddings saved successfully",
                extra={
                    "filepath": str(filepath),
                    "file_size_bytes": filepath.stat().st_size,
                    "metadata_file": str(meta_filepath)
                }
            )
        except Exception as e:
            self.logger.error(
                f"Failed to save embeddings to {filepath}",
                extra={"error": str(e), "filepath": str(filepath)},
                exc_info=True
            )
            raise
    
    def load_embeddings(
        self,
        filepath: Union[str, Path],
        validate_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Load embeddings from pickle file.
        
        Args:
            filepath: Path to pickle file
            validate_shape: Optional tuple (num_vectors, embedding_dim) to validate shape
            
        Returns:
            Array of embeddings, shape [N, D]
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If embeddings shape doesn't match validation
            pickle.UnpicklingError: If pickle file is corrupted
        """
        filepath = validate_file_path(filepath, must_exist=True)
        
        self.logger.info(f"Loading embeddings from {filepath}")
        
        # Load metadata if available
        meta_filepath = filepath.with_suffix('.meta.json')
        metadata = {}
        if meta_filepath.exists():
            try:
                with open(meta_filepath, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Validate version
                file_version = metadata.get('version', 0)
                if file_version > EMBEDDINGS_VERSION:
                    raise ValueError(
                        f"File version {file_version} is newer than supported version {EMBEDDINGS_VERSION}"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load metadata: {e}")
        
        # Load embeddings
        try:
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
        except Exception as e:
            self.logger.error(
                f"Failed to load embeddings from {filepath}",
                extra={"error": str(e)},
                exc_info=True
            )
            raise
        
        # Validate
        if not isinstance(embeddings, np.ndarray):
            raise ValueError(f"Loaded data is not numpy array, got {type(embeddings)}")
        
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
        
        # Validate shape if provided
        if validate_shape is not None:
            expected_num, expected_dim = validate_shape
            num_vectors, embedding_dim = embeddings.shape
            if num_vectors != expected_num or embedding_dim != expected_dim:
                raise ValueError(
                    f"Shape mismatch: expected {validate_shape}, got {embeddings.shape}"
                )
        
        self.logger.info(
            f"Successfully loaded embeddings",
            extra={
                "shape": embeddings.shape,
                "dtype": str(embeddings.dtype),
                "filepath": str(filepath)
            }
        )
        
        return embeddings
    
    def save_index(
        self,
        index_builder: IndexBuilder,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save FAISS index using IndexBuilder.
        
        Args:
            index_builder: IndexBuilder instance with built index
            filepath: Path to save index file
            metadata: Optional metadata to store (saved as JSON)
        
        Raises:
            ValueError: If index not built
            OSError: If file cannot be written
        """
        if index_builder.index is None:
            raise ValueError("Index not built. Cannot save empty index.")
        
        filepath = Path(filepath)
        ensure_dir(filepath.parent)
        
        self.logger.info(f"Saving index to {filepath}")
        
        # Save index using IndexBuilder
        try:
            index_builder.save(filepath)
        except Exception as e:
            self.logger.error(
                f"Failed to save index: {e}",
                extra={"filepath": str(filepath)},
                exc_info=True
            )
            raise
        
        # Save metadata as JSON (same name with .meta.json extension)
        if metadata is not None:
            meta_filepath = filepath.with_suffix('.meta.json')
            storage_data = {
                "version": self.storage_version,
                "timestamp": datetime.utcnow().isoformat(),
                "index_type": index_builder.index_type,
                "embedding_dim": index_builder.embedding_dim,
                "distance_metric": index_builder.distance_metric,
                "num_vectors": index_builder.num_vectors,
                "is_trained": index_builder.is_trained,
                "nlist": getattr(index_builder, 'nlist', None),
                "metadata": metadata
            }
            
            try:
                with open(meta_filepath, 'w', encoding='utf-8') as f:
                    json.dump(storage_data, f, indent=2)
                
                self.logger.debug(f"Index metadata saved to {meta_filepath}")
            except Exception as e:
                self.logger.warning(f"Failed to save index metadata: {e}")
        
        self.logger.info(
            f"Index saved successfully",
            extra={
                "filepath": str(filepath),
                "file_size_bytes": filepath.stat().st_size,
                "num_vectors": index_builder.num_vectors
            }
        )
    
    def load_index(
        self,
        index_builder: IndexBuilder,
        filepath: Union[str, Path],
        load_metadata: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load FAISS index using IndexBuilder.
        
        Args:
            index_builder: IndexBuilder instance to load index into
            filepath: Path to index file
            load_metadata: Whether to load and return metadata
            
        Returns:
            Dictionary with metadata if load_metadata=True, None otherwise
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If index dimension doesn't match
        """
        filepath = validate_file_path(filepath, must_exist=True)
        
        self.logger.info(f"Loading index from {filepath}")
        
        # Load index using IndexBuilder
        try:
            index_builder.load(filepath)
        except Exception as e:
            self.logger.error(
                f"Failed to load index: {e}",
                extra={"filepath": str(filepath)},
                exc_info=True
            )
            raise
        
        # Load metadata if requested
        metadata = None
        if load_metadata:
            meta_filepath = filepath.with_suffix('.meta.json')
            if meta_filepath.exists():
                try:
                    with open(meta_filepath, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    self.logger.debug(f"Loaded index metadata from {meta_filepath}")
                except Exception as e:
                    self.logger.warning(f"Failed to load index metadata: {e}")
        
        self.logger.info(
            f"Index loaded successfully",
            extra={
                "filepath": str(filepath),
                "num_vectors": index_builder.num_vectors,
                "index_type": index_builder.index_type
            }
        )
        
        return metadata
    
    def save_complete_state(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        index_builder: IndexBuilder,
        base_path: Union[str, Path],
        include_document_embeddings: bool = False
    ) -> Dict[str, Path]:
        """
        Save complete RAG state (documents, embeddings, index) to a directory.
        
        Args:
            documents: List of Document objects
            embeddings: Array of embeddings
            index_builder: IndexBuilder with built index
            base_path: Base directory path for storage
            include_document_embeddings: Whether to include embeddings in documents JSON
        
        Returns:
            Dictionary mapping component names to saved file paths
        
        Raises:
            ValueError: If any component is invalid
        """
        base_path = Path(base_path)
        ensure_dir(base_path)
        
        self.logger.info(f"Saving complete RAG state to {base_path}")
        
        # Define file paths
        documents_path = base_path / "documents.json"
        embeddings_path = base_path / "embeddings.pkl"
        index_path = base_path / "index.faiss"
        
        saved_paths = {}
        
        # Save documents
        self.save_documents(documents, documents_path, include_embeddings=include_document_embeddings)
        saved_paths['documents'] = documents_path
        
        # Save embeddings
        embeddings_metadata = {
            "num_documents": len(documents),
            "embedding_dim": embeddings.shape[1],
            "model_info": "Saved with complete state"
        }
        self.save_embeddings(embeddings, embeddings_path, metadata=embeddings_metadata)
        saved_paths['embeddings'] = embeddings_path
        
        # Save index
        index_metadata = {
            "num_documents": len(documents),
            "embedding_dim": embeddings.shape[1],
            "saved_with_state": True
        }
        self.save_index(index_builder, index_path, metadata=index_metadata)
        saved_paths['index'] = index_path
        
        self.logger.info(
            "Complete state saved successfully",
            extra={"base_path": str(base_path), "saved_files": list(saved_paths.keys())}
        )
        
        return saved_paths
    
    def load_complete_state(
        self,
        base_path: Union[str, Path],
        index_builder: Optional[IndexBuilder] = None,
        embedding_dim: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load complete RAG state (documents, embeddings, index) from a directory.
        
        Args:
            base_path: Base directory path containing saved state
            index_builder: Optional IndexBuilder instance (will create one if not provided)
            embedding_dim: Embedding dimension (required if index_builder not provided)
        
        Returns:
            Dictionary with keys: 'documents', 'embeddings', 'index_builder', 'metadata'
        
        Raises:
            FileNotFoundError: If state files don't exist
            ValueError: If parameters are invalid
        """
        base_path = Path(base_path)
        
        if not base_path.exists():
            raise FileNotFoundError(f"State directory not found: {base_path}")
        
        self.logger.info(f"Loading complete RAG state from {base_path}")
        
        # Define file paths
        documents_path = base_path / "documents.json"
        embeddings_path = base_path / "embeddings.pkl"
        index_path = base_path / "index.faiss"
        
        # Validate files exist
        for path in [documents_path, embeddings_path, index_path]:
            if not path.exists():
                raise FileNotFoundError(f"State file not found: {path}")
        
        # Load documents
        documents = self.load_documents(documents_path)
        
        # Load embeddings
        embeddings = self.load_embeddings(embeddings_path)
        
        # Validate consistency
        if len(documents) != embeddings.shape[0]:
            raise ValueError(
                f"Document count ({len(documents)}) doesn't match embedding count ({embeddings.shape[0]})"
            )
        
        # Create or use provided IndexBuilder
        if index_builder is None:
            if embedding_dim is None:
                embedding_dim = embeddings.shape[1]
            
            from retrieval.index_builder import IndexBuilder
            index_builder = IndexBuilder(
                embedding_dim=embedding_dim,
                index_type="flat_l2",
                distance_metric="L2"
            )
        
        # Load index
        metadata = self.load_index(index_builder, index_path, load_metadata=True)
        
        self.logger.info(
            "Complete state loaded successfully",
            extra={
                "num_documents": len(documents),
                "embedding_shape": embeddings.shape,
                "index_type": index_builder.index_type
            }
        )
        
        return {
            'documents': documents,
            'embeddings': embeddings,
            'index_builder': index_builder,
            'metadata': metadata
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Usage examples for PersistenceManager.
    """
    import numpy as np
    
    # Example 1: Save and load documents
    print("Example 1: Document persistence")
    try:
        from src.retrieval.document_loader import DocumentLoader, Document
        
        manager = PersistenceManager(default_storage_dir="storage")
        loader = DocumentLoader()
        docs = loader.load_from_directory("data/text")
        
        # Save documents
        manager.save_documents(docs, "storage/documents.json", include_embeddings=False)
        print(f"✓ Saved {len(docs)} documents")
        
        # Load documents
        loaded_docs = manager.load_documents("storage/documents.json")
        print(f"✓ Loaded {len(loaded_docs)} documents")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Save and load embeddings
    print("Example 2: Embedding persistence")
    try:
        manager = PersistenceManager()
        
        # Create sample embeddings
        embeddings = np.random.rand(10, 768).astype('float32')
        
        # Save with metadata
        metadata = {
            "model_name": "bert-base-uncased",
            "num_documents": 10,
            "description": "Sample embeddings"
        }
        manager.save_embeddings(embeddings, "storage/embeddings.pkl", metadata=metadata)
        print(f"✓ Saved embeddings shape: {embeddings.shape}")
        
        # Load embeddings
        loaded_embeddings = manager.load_embeddings("storage/embeddings.pkl")
        print(f"✓ Loaded embeddings shape: {loaded_embeddings.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Save and load index
    print("Example 3: Index persistence")
    try:
        from retrieval.index_builder import IndexBuilder
        
        manager = PersistenceManager()
        
        # Create and build index
        index_builder = IndexBuilder(embedding_dim=768, index_type="flat_l2")
        sample_embeddings = np.random.rand(10, 768).astype('float32')
        index_builder.build(sample_embeddings)
        
        # Save index
        manager.save_index(index_builder, "storage/index.faiss", metadata={"test": True})
        print(f"✓ Saved index with {index_builder.num_vectors} vectors")
        
        # Load index
        new_index_builder = IndexBuilder(embedding_dim=768)
        metadata = manager.load_index(new_index_builder, "storage/index.faiss")
        print(f"✓ Loaded index with {new_index_builder.num_vectors} vectors")
        if metadata:
            print(f"  Metadata: {metadata.get('index_type')}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Complete state save/load
    print("Example 4: Complete state persistence")
    try:
        from src.retrieval import DocumentLoader
        from src.retrieval.index_builder import IndexBuilder
        
        manager = PersistenceManager()
        
        # Load documents
        loader = DocumentLoader()
        docs = loader.load_from_directory("data/text")
        
        # Create sample embeddings
        embeddings = np.random.rand(len(docs), 768).astype('float32')
        
        # Build index
        index_builder = IndexBuilder(embedding_dim=768)
        index_builder.build(embeddings)
        
        # Save complete state
        saved_paths = manager.save_complete_state(
            docs, embeddings, index_builder, "storage/complete_state"
        )
        print(f"✓ Saved complete state to:")
        for component, path in saved_paths.items():
            print(f"  {component}: {path}")
        
        # Load complete state
        state = manager.load_complete_state("storage/complete_state", embedding_dim=768)
        print(f"✓ Loaded complete state:")
        print(f"  Documents: {len(state['documents'])}")
        print(f"  Embeddings: {state['embeddings'].shape}")
        print(f"  Index vectors: {state['index_builder'].num_vectors}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

