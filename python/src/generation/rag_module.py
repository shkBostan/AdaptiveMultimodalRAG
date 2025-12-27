"""
RAG (Retrieval-Augmented Generation) module.

This module provides context-aware generation with structured merging of retrieved
documents, supporting both Document objects and legacy dictionary format.

Author: s Bostan
Created on: Nov, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..retrieval.document_loader import Document


class RAGModule:
    """
    RAG module for generating responses using retrieved context.
    
    Provides context-aware generation with structured document merging, metadata
    integration, and intelligent length control. Supports both Document objects
    and legacy dictionary format for backward compatibility.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_context_length: int = 2000,
        include_metadata: bool = True,
        metadata_marker_format: str = "brackets"
    ):
        """
        Initialize RAG module.
        
        Args:
            model_name: Name of the language model to use
            max_context_length: Maximum characters for context (before tokenization)
            include_metadata: Whether to include metadata markers in context
            metadata_marker_format: Format for metadata markers
                - "brackets": [Document 1: source=file.txt]
                - "parentheses": (Document 1: source=file.txt)
                - "none": No metadata markers
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        # Context preparation settings
        self.max_context_length = max_context_length
        self.include_metadata = include_metadata
        self.metadata_marker_format = metadata_marker_format
        
    def load_model(self):
        """Load language model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_context(
        self,
        retrieved_docs: Union[List[Document], List[Dict]],
        max_length: Optional[int] = None
    ) -> str:
        """
        Prepare context string from retrieved documents with structured merging.
        
        Merges document content with metadata markers, applies intelligent trimming,
        and returns a clean, structured context string ready for generation.
        
        Args:
            retrieved_docs: List of Document objects or document dictionaries
            max_length: Maximum context length in characters (overrides instance default)
            
        Returns:
            Clean, structured context string with metadata markers
            
        Examples:
            >>> docs = [Document(id="1", content="Text...", metadata={"source": "file.txt"})]
            >>> context = rag_module.prepare_context(docs)
            >>> # Returns: "[Document 1: source=file.txt]\nText..."
        """
        if not retrieved_docs:
            return ""
        
        max_len = max_length if max_length is not None else self.max_context_length
        
        # Convert to unified format
        doc_list = self._normalize_documents(retrieved_docs)
        
        if not doc_list:
            return ""
        
        # Build context parts with metadata markers
        context_parts = []
        current_length = 0
        
        for i, doc_data in enumerate(doc_list, 1):
            content = doc_data.get('content', doc_data.get('text', ''))
            if not content:
                continue
            
            # Build metadata marker
            metadata_str = self._build_metadata_marker(doc_data, i)
            
            # Format document entry
            if metadata_str:
                doc_entry = f"{metadata_str}\n{content}"
            else:
                doc_entry = content
            
            # Check if adding this document would exceed length limit
            doc_length = len(doc_entry)
            if current_length + doc_length > max_len and context_parts:
                # Truncate current document if needed
                remaining = max_len - current_length - len(metadata_str) - 10  # Buffer for formatting
                if remaining > 100:  # Only add if meaningful content remains
                    truncated_content = self._truncate_text(content, remaining)
                    if metadata_str:
                        doc_entry = f"{metadata_str}\n{truncated_content}"
                    else:
                        doc_entry = truncated_content
                    context_parts.append(doc_entry)
                break
            
            context_parts.append(doc_entry)
            current_length += doc_length + 2  # +2 for newline separator
        
        # Join with clear separators
        context = "\n\n".join(context_parts)
        
        # Final length check and trimming if needed
        if len(context) > max_len:
            context = context[:max_len].rsplit('\n', 1)[0]  # Trim at last newline
        
        return context.strip()
    
    def _normalize_documents(
        self,
        retrieved_docs: Union[List[Document], List[Dict]]
    ) -> List[Dict]:
        """
        Normalize documents to unified dictionary format.
        
        Args:
            retrieved_docs: List of Document objects or dictionaries
            
        Returns:
            List of normalized dictionaries
        """
        normalized = []
        
        for doc in retrieved_docs:
            if hasattr(doc, 'to_dict'):
                # Document object
                normalized.append(doc.to_dict())
            elif isinstance(doc, dict):
                # Already a dictionary
                normalized.append(doc)
            else:
                # Unknown type, skip with warning
                continue
        
        return normalized
    
    def _build_metadata_marker(self, doc_data: Dict, index: int) -> str:
        """
        Build metadata marker string for a document.
        
        Args:
            doc_data: Document dictionary
            index: Document index (1-based)
            
        Returns:
            Metadata marker string or empty string
        """
        if not self.include_metadata:
            return ""
        
        # Extract relevant metadata
        metadata_items = []
        
        # Priority metadata fields
        priority_fields = ['source', 'filename', 'author', 'date', 'category']
        for field in priority_fields:
            if field in doc_data:
                value = str(doc_data[field])
                if value and value not in ['None', '']:
                    metadata_items.append(f"{field}={value}")
        
        # Add other metadata (excluding content, id, score, rank)
        excluded = {'content', 'text', 'id', 'score', 'rank', 'embedding', 
                   'image_path', 'audio_path', *priority_fields}
        for key, value in doc_data.items():
            if key not in excluded and value not in [None, '']:
                metadata_items.append(f"{key}={value}")
        
        if not metadata_items:
            return ""
        
        metadata_str = ", ".join(metadata_items[:3])  # Limit to 3 items
        
        # Format based on marker format
        if self.metadata_marker_format == "brackets":
            return f"[Document {index}: {metadata_str}]"
        elif self.metadata_marker_format == "parentheses":
            return f"(Document {index}: {metadata_str})"
        else:
            return f"Document {index}: {metadata_str}"
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Intelligently truncate text while preserving word boundaries.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            
        Returns:
            Truncated text with ellipsis if needed
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        # Use the later of period or newline
        cut_point = max(last_period, last_newline)
        
        if cut_point > max_length * 0.7:  # Only use if reasonable
            truncated = text[:cut_point + 1]
        else:
            # Truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.7:
                truncated = text[:last_space]
        
        return truncated.strip() + "..."
    
    def generate(
        self,
        query: str,
        retrieved_docs: Union[List[Document], List[Dict]],
        max_length: int = 512,
        temperature: float = 0.7,
        context_max_length: Optional[int] = None
    ) -> str:
        """
        Generate response using query and retrieved documents.
        
        Automatically prepares context from retrieved documents, handles empty
        context gracefully, and ensures deterministic prompt structure.
        
        Args:
            query: User query string
            retrieved_docs: List of Document objects or document dictionaries
            max_length: Maximum length of generated response (in tokens)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            context_max_length: Maximum context length in characters (overrides instance default)
            
        Returns:
            Generated response text
            
        Raises:
            ValueError: If query is empty
            
        Examples:
            >>> # With Document objects
            >>> docs = [Document(id="1", content="RAG is...", metadata={"source": "doc.txt"})]
            >>> response = rag_module.generate("What is RAG?", docs)
            
            >>> # With dictionaries (backward compatible)
            >>> docs = [{"id": "1", "content": "RAG is...", "source": "doc.txt"}]
            >>> response = rag_module.generate("What is RAG?", docs)
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if self.model is None:
            self.load_model()
        
        # Prepare context using new method
        context = self.prepare_context(retrieved_docs, max_length=context_max_length)
        
        # Safety fallback for empty context
        if not context:
            # Fallback prompt when no context available
            prompt = self._build_fallback_prompt(query)
        else:
            # Standard prompt with context
            prompt = self._build_prompt(query, context)
        
        # Ensure deterministic structure: always tokenize with same parameters
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        # Generate response with consistent parameters
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2  # Prevent repetition
        )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part (remove prompt)
        response = self._extract_answer(response, prompt)
        
        return response.strip()
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build structured prompt from query and context.
        
        Ensures deterministic, consistent prompt structure.
        
        Args:
            query: User query
            context: Prepared context string
            
        Returns:
            Structured prompt string
        """
        # Deterministic prompt structure
        prompt = (
            "Context:\n"
            f"{context}\n\n"
            "Query:\n"
            f"{query}\n\n"
            "Answer:"
        )
        return prompt
    
    def _build_fallback_prompt(self, query: str) -> str:
        """
        Build fallback prompt when no context is available.
        
        Args:
            query: User query
            
        Returns:
            Fallback prompt string
        """
        return (
            "Query:\n"
            f"{query}\n\n"
            "Answer based on your knowledge:"
        )
    
    def _extract_answer(self, response: str, prompt: str) -> str:
        """
        Extract answer from model response, removing prompt.
        
        Args:
            response: Full model response
            prompt: Original prompt
            
        Returns:
            Extracted answer text
        """
        # Try multiple extraction strategies
        if "Answer:" in response:
            # Extract after "Answer:" marker
            parts = response.split("Answer:", 1)
            if len(parts) > 1:
                return parts[-1].strip()
        
        # Remove prompt if it appears in response
        if prompt in response:
            answer = response.replace(prompt, "").strip()
            if answer:
                return answer
        
        # If prompt starts response, remove it
        if response.startswith(prompt):
            answer = response[len(prompt):].strip()
            if answer:
                return answer
        
        # Return as-is if no extraction needed
        return response
    
    def _construct_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Construct context string from retrieved documents (legacy method).
        
        This method is maintained for backward compatibility.
        New code should use prepare_context() instead.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Context string
        """
        # Delegate to prepare_context for consistency
        return self.prepare_context(retrieved_docs)

