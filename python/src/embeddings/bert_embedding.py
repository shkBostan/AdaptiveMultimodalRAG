"""
BERT embedding model implementation.

Author: s Bostan
Created on: Nov, 2025
"""

import torch
from transformers import BertTokenizer, BertModel
from typing import List, Optional
import numpy as np


class BERTEmbedding:
    """BERT-based embedding model for text processing."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize BERT model.
        
        Args:
            model_name: Name of the BERT model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load BERT tokenizer and model."""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for input text.
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector
        """
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            Array of embedding vectors
        """
        if self.model is None:
            self.load_model()
        
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True,
                               max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings

