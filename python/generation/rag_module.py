"""
RAG (Retrieval-Augmented Generation) module.

Author: s Bostan
Created on: Nov, 2025
"""

from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


class RAGModule:
    """RAG module for generating responses using retrieved context."""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize RAG module.
        
        Args:
            model_name: Name of the language model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load language model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, query: str, retrieved_docs: List[Dict], 
                max_length: int = 512) -> str:
        """
        Generate response using query and retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document dictionaries
            max_length: Maximum length of generated response
            
        Returns:
            Generated response text
        """
        if self.model is None:
            self.load_model()
        
        # Construct context from retrieved documents
        context = self._construct_context(retrieved_docs)
        
        # Combine query and context
        prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                               max_length=max_length)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        
        return response
    
    def _construct_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Construct context string from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved document dictionaries
            
        Returns:
            Context string
        """
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get('content', doc.get('text', ''))
            if content:
                context_parts.append(f"[{i}] {content}")
        
        return "\n".join(context_parts)

