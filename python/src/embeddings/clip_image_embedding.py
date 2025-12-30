"""
CLIP image embedding model implementation.

This module provides CLIP-based image embeddings with support for various CLIP model
variants and flexible configuration options. Designed for research-grade applications
with comprehensive error handling and production-ready features.

Author: s Bostan
Created on: Nov, 2025
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import logging

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore

logger = logging.getLogger(__name__)


class CLIPImageEmbedding:
    """
    CLIP-based image embedding model for image processing.
    
    Uses CLIP (Contrastive Language-Image Pre-training) vision encoder to convert
    images into fixed-size vector representations suitable for multimodal retrieval
    and similarity search.
    
    Features:
    - Lazy loading (models loaded on first use)
    - Batch processing support
    - GPU/CPU automatic device selection
    - Support for file paths and PIL.Image objects
    - Optional L2 normalization for cosine similarity
    - Comprehensive error handling
    
    Example:
        >>> # Basic usage with default settings
        >>> model = CLIPImageEmbedding(model_name="openai/clip-vit-base-patch32")
        >>> embedding = model.get_embedding("path/to/image.jpg")
        
        >>> # Using PIL Image object
        >>> from PIL import Image
        >>> img = Image.open("image.jpg")
        >>> embedding = model.get_embedding(img)
        
        >>> # Batch processing
        >>> embeddings = model.get_embeddings_batch(["img1.jpg", "img2.jpg"])
        
        >>> # With normalization for cosine similarity
        >>> model = CLIPImageEmbedding(normalize_embeddings=True)
        >>> embedding = model.get_embedding("image.jpg")
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[Union[torch.device, str]] = None,
        normalize_embeddings: bool = False,
        trust_remote_code: bool = False
    ):
        """
        Initialize CLIP image embedding model.
        
        Args:
            model_name: Name or path of the CLIP model to use.
                       Supports any HuggingFace CLIP model:
                       - "openai/clip-vit-base-patch32"
                       - "openai/clip-vit-base-patch16"
                       - "openai/clip-vit-large-patch14"
                       - Custom model paths
            device: PyTorch device ('cuda', 'cpu', or torch.device).
                   If None, automatically selects CUDA if available
            normalize_embeddings: Whether to L2-normalize embeddings.
                                 Useful for cosine similarity calculations
            trust_remote_code: Whether to trust remote code when loading models from HuggingFace
        
        Raises:
            ImportError: If transformers, torch, or PIL is not installed
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL (Pillow) is required for CLIPImageEmbedding. "
                "Install it with: pip install Pillow"
            )
        
        self.model_name = model_name
        self.processor: Optional[CLIPProcessor] = None
        self.model: Optional[CLIPModel] = None
        
        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Processing parameters
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        
        # Lazy loading tracking
        self._model_loaded = False
        self._embedding_dim: Optional[int] = None
        
        logger.info(
            f"Initialized CLIPImageEmbedding with model={model_name}, "
            f"device={self.device}, normalize={normalize_embeddings}"
        )
    
    def load_model(self) -> None:
        """
        Load CLIP processor and model.
        
        Raises:
            OSError: If model cannot be loaded from HuggingFace Hub
            ValueError: If model configuration is invalid
        """
        if self._model_loaded and self.processor is not None and self.model is not None:
            return
        
        try:
            logger.info(f"Loading CLIP model: {self.model_name}")
            
            # Load processor
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            # Load model
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            # Determine embedding dimension from model config
            config = self.model.config
            if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'hidden_size'):
                self._embedding_dim = config.vision_config.hidden_size
            elif hasattr(config, 'projection_dim'):
                # Some CLIP models use projection_dim
                self._embedding_dim = config.projection_dim
            else:
                # Fallback: infer from a dummy forward pass
                with torch.no_grad():
                    dummy_image = Image.new('RGB', (224, 224), color='black')
                    dummy_inputs = self.processor(images=[dummy_image], return_tensors="pt")
                    dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}
                    dummy_outputs = self.model.get_image_features(**dummy_inputs)
                    self._embedding_dim = dummy_outputs.shape[-1]
            
            self._model_loaded = True
            logger.info(
                f"Model loaded successfully. Embedding dimension: {self._embedding_dim}, "
                f"Device: {self.device}"
            )
            
        except Exception as e:
            raise OSError(
                f"Failed to load CLIP model '{self.model_name}'. "
                f"Error: {str(e)}. "
                f"Ensure the model name is correct and you have internet access "
                f"to download from HuggingFace Hub."
            ) from e
    
    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Load image from various input types.
        
        Args:
            image: Image input (file path as str/Path, or PIL.Image.Image)
        
        Returns:
            PIL.Image.Image object
        
        Raises:
            FileNotFoundError: If image file path doesn't exist
            ValueError: If image cannot be loaded
        """
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            try:
                return Image.open(image_path).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to load image from {image_path}: {str(e)}") from e
        elif isinstance(image, Image.Image):
            # Ensure RGB format
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                f"Expected str, Path, or PIL.Image.Image"
            )
    
    def get_embedding(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image: Image input. Can be:
                  - File path (str or Path)
                  - PIL.Image.Image object
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        
        Raises:
            ValueError: If model is not loaded
            RuntimeError: If embedding generation fails
        """
        if not self._model_loaded:
            self.load_model()
        
        if self.processor is None or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Load and preprocess image
            pil_image = self._load_image(image)
            
            # Process image (resize, normalize, etc.)
            inputs = self.processor(images=[pil_image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                # Use vision encoder to get image features
                embedding = self.model.get_image_features(**inputs)
                
                # Normalize if requested
                if self.normalize_embeddings:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                # Convert to numpy (single sample, so squeeze batch dimension)
                embedding = embedding.cpu().numpy()[0]
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate embedding for image. Error: {str(e)}"
            ) from e
    
    def get_embeddings_batch(self, images: List[Union[str, Path, Image.Image]]) -> np.ndarray:
        """
        Generate embeddings for a batch of images.
        
        Args:
            images: List of image inputs. Each can be:
                   - File path (str or Path)
                   - PIL.Image.Image object
        
        Returns:
            Array of embedding vectors of shape (n_images, embedding_dim)
        
        Raises:
            ValueError: If model is not loaded or images list is empty
            RuntimeError: If embedding generation fails
        """
        if not images:
            return np.array([])
        
        if not self._model_loaded:
            self.load_model()
        
        if self.processor is None or self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Load and preprocess all images
            pil_images = [self._load_image(img) for img in images]
            
            # Process images batch
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                # Use vision encoder to get image features
                embeddings = self.model.get_image_features(**inputs)
                
                # Normalize if requested
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy
                embeddings = embeddings.cpu().numpy()
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate batch embeddings. Error: {str(e)}"
            ) from e
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        # If not loaded yet, provide estimate based on model name
        if "large" in self.model_name.lower():
            return 768
        elif "base" in self.model_name.lower():
            return 512
        else:
            return 512  # Default for CLIP base models
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"CLIPImageEmbedding(model_name='{self.model_name}', "
            f"device={self.device}, normalize={self.normalize_embeddings}, "
            f"loaded={self._model_loaded})"
        )
