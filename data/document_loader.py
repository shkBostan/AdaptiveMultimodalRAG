"""
Document loading subsystem for AdaptiveMultimodalRAG.

This module provides a scalable and extensible document loading system that supports
multiple file formats, text normalization, chunking, and is designed to be future-proof
for multimodal inputs (text, image, audio).

Author: s Bostan
Created on: Nov, 2025
"""

import os
import json
import csv
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Union, Callable, Any, Iterator
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import re


@dataclass
class Document:
    """
    Represents a document in the RAG system.
    
    Attributes:
        id: Unique identifier for the document
        content: Main text content of the document
        metadata: Additional metadata (source, author, timestamp, etc.)
        embedding: Optional embedding vector (set after embedding generation)
        image_path: Optional path to associated image (for multimodal support)
        audio_path: Optional path to associated audio (for multimodal support)
    """
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary format compatible with RetrievalEngine.
        
        Returns:
            Dictionary with 'id', 'content', and metadata fields
        """
        result = {
            'id': self.id,
            'content': self.content,
            **self.metadata
        }
        if self.embedding is not None:
            result['embedding'] = self.embedding
        if self.image_path is not None:
            result['image_path'] = self.image_path
        if self.audio_path is not None:
            result['audio_path'] = self.audio_path
        return result
    
    def __str__(self) -> str:
        """String representation of the document."""
        return f"Document(id={self.id}, content_length={len(self.content)}, metadata_keys={list(self.metadata.keys())})"


class TextNormalizer(ABC):
    """
    Abstract base class for text normalization strategies.
    Follows Strategy pattern for extensibility.
    """
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize text according to the strategy.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        pass


class BasicTextNormalizer(TextNormalizer):
    """
    Basic text normalization: lowercase, remove extra whitespace, optional punctuation removal.
    """
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = False):
        """
        Initialize basic text normalizer.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation marks
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def normalize(self, text: str) -> str:
        """
        Normalize text using basic transformations.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            # Remove punctuation but keep alphanumeric and whitespace
            text = re.sub(r'[^\w\s]', '', text)
        
        # Normalize whitespace: replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text


class Chunker:
    """
    Handles text chunking with configurable size and overlap.
    Uses sliding window approach for overlap.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, chunk_by: str = "characters"):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Size of each chunk (in characters or tokens)
            overlap: Number of characters/tokens to overlap between chunks
            chunk_by: Method to chunk by - "characters" or "tokens"
            
        Raises:
            ValueError: If chunk_size <= 0 or overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        if chunk_by not in ["characters", "tokens"]:
            raise ValueError("chunk_by must be 'characters' or 'tokens'")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_by = chunk_by
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        if self.chunk_by == "tokens":
            return self._chunk_by_tokens(text)
        else:
            return self._chunk_by_characters(text)
    
    def _chunk_by_characters(self, text: str) -> List[str]:
        """Chunk text by characters."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk = text[start:end]
            
            # Try to break at word boundary if not at end
            if end < text_length and chunk:
                # Find last space in chunk
                last_space = chunk.rfind(' ')
                if last_space > self.chunk_size * 0.5:  # Only break if reasonable
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - self.overlap if end < text_length else end
        
        return [chunk for chunk in chunks if chunk]  # Remove empty chunks
    
    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Chunk text by tokens (words)."""
        tokens = text.split()
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk = ' '.join(chunk_tokens)
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap if end < len(tokens) else end
        
        return chunks


class DocumentLoaderStrategy(ABC):
    """
    Abstract base class for document loading strategies.
    Follows Strategy pattern to support different file formats.
    """
    
    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load documents from a source.
        
        Args:
            source: Path to the source (file or directory)
            
        Returns:
            List of Document objects
        """
        pass


class TextFileLoader(DocumentLoaderStrategy):
    """
    Loads documents from a directory of .txt files.
    Each .txt file becomes one document.
    """
    
    def __init__(self, normalizer: Optional[TextNormalizer] = None):
        """
        Initialize text file loader.
        
        Args:
            normalizer: Optional text normalizer to apply
        """
        self.normalizer = normalizer or BasicTextNormalizer()
    
    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load documents from directory of .txt files.
        
        Args:
            source: Path to directory containing .txt files
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If source is not a directory or doesn't exist
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source}")
        if not source_path.is_dir():
            raise ValueError(f"Source must be a directory: {source}")
        
        documents = []
        
        # Find all .txt files recursively
        txt_files = list(source_path.rglob("*.txt"))
        
        if not txt_files:
            raise ValueError(f"No .txt files found in directory: {source}")
        
        for txt_file in txt_files:
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Normalize text
                normalized_content = self.normalizer.normalize(content)
                
                if normalized_content:  # Only create document if content exists
                    doc = Document(
                        id=str(uuid.uuid4()),
                        content=normalized_content,
                        metadata={
                            'source': str(txt_file),
                            'filename': txt_file.name,
                            'file_path': str(txt_file.relative_to(source_path))
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                # Log error but continue with other files
                print(f"Warning: Failed to load {txt_file}: {e}")
                continue
        
        return documents


class JSONLoader(DocumentLoaderStrategy):
    """
    Loads documents from JSON files.
    Supports both list format and dict format.
    """
    
    def __init__(
        self,
        content_key: str = "content",
        id_key: str = "id",
        metadata_keys: Optional[List[str]] = None,
        normalizer: Optional[TextNormalizer] = None
    ):
        """
        Initialize JSON loader.
        
        Args:
            content_key: Key in JSON object that contains the text content
            id_key: Key in JSON object that contains the document ID (optional)
            metadata_keys: List of keys to include in metadata
            normalizer: Optional text normalizer to apply
        """
        self.content_key = content_key
        self.id_key = id_key
        self.metadata_keys = metadata_keys or []
        self.normalizer = normalizer or BasicTextNormalizer()
    
    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load documents from JSON file(s).
        
        Args:
            source: Path to JSON file or directory containing JSON files
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If source doesn't exist or is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source}")
        
        documents = []
        
        # Handle single file or directory
        if source_path.is_file():
            json_files = [source_path]
        elif source_path.is_dir():
            json_files = list(source_path.rglob("*.json"))
        else:
            raise ValueError(f"Source must be a file or directory: {source}")
        
        if not json_files:
            raise ValueError(f"No JSON files found: {source}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle list of documents
                if isinstance(data, list):
                    for item in data:
                        doc = self._create_document_from_dict(item, json_file)
                        if doc:
                            documents.append(doc)
                
                # Handle single document dict
                elif isinstance(data, dict):
                    doc = self._create_document_from_dict(data, json_file)
                    if doc:
                        documents.append(doc)
                
                else:
                    raise ValueError(f"JSON must be a list or dict, got {type(data)}")
            
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in {json_file}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        return documents
    
    def _create_document_from_dict(self, data: Dict[str, Any], source_file: Path) -> Optional[Document]:
        """
        Create a Document from a dictionary.
        
        Args:
            data: Dictionary containing document data
            source_file: Source file path for metadata
            
        Returns:
            Document object or None if content is missing
        """
        if self.content_key not in data:
            return None
        
        content = str(data[self.content_key])
        normalized_content = self.normalizer.normalize(content)
        
        if not normalized_content:
            return None
        
        # Get ID from data or generate one
        doc_id = data.get(self.id_key, str(uuid.uuid4()))
        
        # Build metadata
        metadata = {
            'source': str(source_file),
            'filename': source_file.name
        }
        
        # Add specified metadata keys
        for key in self.metadata_keys:
            if key in data and key != self.content_key and key != self.id_key:
                metadata[key] = data[key]
        
        # Add any remaining keys as metadata (except content and id)
        for key, value in data.items():
            if key not in [self.content_key, self.id_key] and key not in metadata:
                metadata[key] = value
        
        return Document(
            id=str(doc_id),
            content=normalized_content,
            metadata=metadata
        )


class CSVLoader(DocumentLoaderStrategy):
    """
    Loads documents from CSV files.
    Configurable column name for content.
    """
    
    def __init__(
        self,
        content_column: str = "content",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        normalizer: Optional[TextNormalizer] = None,
        delimiter: str = ","
    ):
        """
        Initialize CSV loader.
        
        Args:
            content_column: Name of column containing text content
            id_column: Optional name of column containing document ID
            metadata_columns: List of column names to include in metadata
            normalizer: Optional text normalizer to apply
            delimiter: CSV delimiter character
        """
        self.content_column = content_column
        self.id_column = id_column
        self.metadata_columns = metadata_columns or []
        self.normalizer = normalizer or BasicTextNormalizer()
        self.delimiter = delimiter
    
    def load(self, source: Union[str, Path]) -> List[Document]:
        """
        Load documents from CSV file(s).
        
        Args:
            source: Path to CSV file or directory containing CSV files
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If source doesn't exist or content_column not found
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source}")
        
        documents = []
        
        # Handle single file or directory
        if source_path.is_file():
            csv_files = [source_path]
        elif source_path.is_dir():
            csv_files = list(source_path.rglob("*.csv"))
        else:
            raise ValueError(f"Source must be a file or directory: {source}")
        
        if not csv_files:
            raise ValueError(f"No CSV files found: {source}")
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=self.delimiter)
                    
                    # Check if content column exists
                    if self.content_column not in reader.fieldnames:
                        raise ValueError(
                            f"Content column '{self.content_column}' not found in {csv_file}. "
                            f"Available columns: {reader.fieldnames}"
                        )
                    
                    for row in reader:
                        content = str(row[self.content_column])
                        normalized_content = self.normalizer.normalize(content)
                        
                        if not normalized_content:
                            continue
                        
                        # Get ID from column or generate one
                        doc_id = row.get(self.id_column, str(uuid.uuid4())) if self.id_column else str(uuid.uuid4())
                        
                        # Build metadata
                        metadata = {
                            'source': str(csv_file),
                            'filename': csv_file.name
                        }
                        
                        # Add specified metadata columns
                        for col in self.metadata_columns:
                            if col in row and col != self.content_column and col != self.id_column:
                                metadata[col] = row[col]
                        
                        # Add all other columns as metadata
                        for key, value in row.items():
                            if key not in [self.content_column, self.id_column] and key not in metadata:
                                metadata[key] = value
                        
                        doc = Document(
                            id=str(doc_id),
                            content=normalized_content,
                            metadata=metadata
                        )
                        documents.append(doc)
            
            except Exception as e:
                print(f"Warning: Failed to load {csv_file}: {e}")
                continue
        
        return documents


class DocumentLoader:
    """
    Main document loader class that orchestrates loading from various sources.
    Follows Facade pattern to provide a simple interface.
    """
    
    def __init__(
        self,
        normalizer: Optional[TextNormalizer] = None,
        chunker: Optional[Chunker] = None,
        auto_chunk: bool = False
    ):
        """
        Initialize document loader.
        
        Args:
            normalizer: Text normalizer to apply (default: BasicTextNormalizer)
            chunker: Chunker for splitting long documents (optional)
            auto_chunk: Whether to automatically chunk documents
        """
        self.normalizer = normalizer or BasicTextNormalizer()
        self.chunker = chunker
        self.auto_chunk = auto_chunk
    
    def load_from_directory(
        self,
        directory: Union[str, Path],
        file_pattern: str = "*.txt"
    ) -> List[Document]:
        """
        Load documents from a directory of text files.
        
        Args:
            directory: Path to directory containing text files
            file_pattern: Glob pattern for files to load (default: "*.txt")
            
        Returns:
            List of Document objects
        """
        loader = TextFileLoader(normalizer=self.normalizer)
        return self._process_documents(loader.load(directory))
    
    def load_from_json(
        self,
        source: Union[str, Path],
        content_key: str = "content",
        id_key: str = "id",
        metadata_keys: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load documents from JSON file(s).
        
        Args:
            source: Path to JSON file or directory
            content_key: Key containing text content
            id_key: Key containing document ID
            metadata_keys: Keys to include in metadata
            
        Returns:
            List of Document objects
        """
        loader = JSONLoader(
            content_key=content_key,
            id_key=id_key,
            metadata_keys=metadata_keys,
            normalizer=self.normalizer
        )
        return self._process_documents(loader.load(source))
    
    def load_from_csv(
        self,
        source: Union[str, Path],
        content_column: str = "content",
        id_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ","
    ) -> List[Document]:
        """
        Load documents from CSV file(s).
        
        Args:
            source: Path to CSV file or directory
            content_column: Column name containing text content
            id_column: Optional column name containing document ID
            metadata_columns: Column names to include in metadata
            delimiter: CSV delimiter
            
        Returns:
            List of Document objects
        """
        loader = CSVLoader(
            content_column=content_column,
            id_column=id_column,
            metadata_columns=metadata_columns,
            normalizer=self.normalizer,
            delimiter=delimiter
        )
        return self._process_documents(loader.load(source))
    
    def load_from_mixed(
        self,
        directory: Union[str, Path],
        recursive: bool = True
    ) -> List[Document]:
        """
        Load documents from a directory containing mixed file types.
        Automatically detects and loads .txt, .json, and .csv files.
        
        Args:
            directory: Path to directory
            recursive: Whether to search recursively
            
        Returns:
            List of Document objects
        """
        directory_path = Path(directory)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")
        
        all_documents = []
        
        # Load text files
        try:
            txt_loader = TextFileLoader(normalizer=self.normalizer)
            txt_files = list(directory_path.rglob("*.txt")) if recursive else list(directory_path.glob("*.txt"))
            for txt_file in txt_files:
                try:
                    docs = txt_loader.load(txt_file.parent)
                    all_documents.extend([d for d in docs if d.metadata.get('filename') == txt_file.name])
                except:
                    pass
        except Exception as e:
            print(f"Warning: Failed to load text files: {e}")
        
        # Load JSON files
        try:
            json_loader = JSONLoader(normalizer=self.normalizer)
            json_files = list(directory_path.rglob("*.json")) if recursive else list(directory_path.glob("*.json"))
            for json_file in json_files:
                try:
                    docs = json_loader.load(json_file)
                    all_documents.extend(docs)
                except:
                    pass
        except Exception as e:
            print(f"Warning: Failed to load JSON files: {e}")
        
        # Load CSV files
        try:
            csv_loader = CSVLoader(normalizer=self.normalizer)
            csv_files = list(directory_path.rglob("*.csv")) if recursive else list(directory_path.glob("*.csv"))
            for csv_file in csv_files:
                try:
                    docs = csv_loader.load(csv_file)
                    all_documents.extend(docs)
                except:
                    pass
        except Exception as e:
            print(f"Warning: Failed to load CSV files: {e}")
        
        return self._process_documents(all_documents)
    
    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents (apply chunking if enabled).
        
        Args:
            documents: List of documents to process
            
        Returns:
            Processed list of documents (may be longer if chunking enabled)
        """
        if not self.auto_chunk or self.chunker is None:
            return documents
        
        chunked_documents = []
        for doc in documents:
            chunks = self.chunker.chunk(doc.content)
            
            if not chunks:
                # If chunking produces no chunks, keep original
                chunked_documents.append(doc)
            elif len(chunks) == 1:
                # Single chunk, update content
                doc.content = chunks[0]
                chunked_documents.append(doc)
            else:
                # Multiple chunks, create new documents
                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        id=f"{doc.id}_chunk_{i}",
                        content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'original_id': doc.id
                        },
                        image_path=doc.image_path,
                        audio_path=doc.audio_path
                    )
                    chunked_documents.append(chunk_doc)
        
        return chunked_documents


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Usage examples for the document loader.
    """
    
    # Example 1: Load from directory of text files
    print("Example 1: Loading from text files directory")
    loader = DocumentLoader()
    try:
        docs = loader.load_from_directory("data/text")
        print(f"Loaded {len(docs)} documents from text files")
        if docs:
            print(f"First document: {docs[0]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Load from JSON file with custom keys
    print("Example 2: Loading from JSON file")
    loader = DocumentLoader()
    try:
        docs = loader.load_from_json(
            "data/metadata/sample.json",
            content_key="text",
            id_key="doc_id",
            metadata_keys=["author", "date"]
        )
        print(f"Loaded {len(docs)} documents from JSON")
        if docs:
            print(f"First document metadata: {docs[0].metadata}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Load from CSV with chunking
    print("Example 3: Loading from CSV with automatic chunking")
    chunker = Chunker(chunk_size=500, overlap=50)
    loader = DocumentLoader(chunker=chunker, auto_chunk=True)
    try:
        docs = loader.load_from_csv(
            "data/metadata/sample.csv",
            content_column="description",
            id_column="id",
            metadata_columns=["category", "tags"]
        )
        print(f"Loaded {len(docs)} documents (may be chunked) from CSV")
        if docs:
            print(f"First document: {docs[0]}")
            print(f"Content length: {len(docs[0].content)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Load from mixed directory
    print("Example 4: Loading from mixed directory (txt, json, csv)")
    loader = DocumentLoader()
    try:
        docs = loader.load_from_mixed("data", recursive=True)
        print(f"Loaded {len(docs)} documents from mixed sources")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 5: Custom normalizer
    print("Example 5: Using custom text normalizer")
    custom_normalizer = BasicTextNormalizer(lowercase=True, remove_punctuation=True)
    loader = DocumentLoader(normalizer=custom_normalizer)
    try:
        docs = loader.load_from_directory("data/text")
        if docs:
            print(f"Sample normalized content: {docs[0].content[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 6: Convert to dictionary format for RetrievalEngine
    print("Example 6: Converting documents to dictionary format")
    loader = DocumentLoader()
    try:
        docs = loader.load_from_directory("data/text")
        if docs:
            doc_dict = docs[0].to_dict()
            print(f"Document as dict keys: {list(doc_dict.keys())}")
            print(f"Has 'id': {'id' in doc_dict}")
            print(f"Has 'content': {'content' in doc_dict}")
    except Exception as e:
        print(f"Error: {e}")

