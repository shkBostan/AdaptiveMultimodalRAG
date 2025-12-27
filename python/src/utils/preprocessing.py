"""
Preprocessing utilities for multimodal data.

Author: s Bostan
Created on: Nov, 2025
"""

import re
from typing import List, Optional
from PIL import Image
import numpy as np


def preprocess_text(text: str, lower: bool = True, 
                   remove_punctuation: bool = False) -> str:
    """
    Preprocess text data.
    
    Args:
        text: Input text
        lower: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Preprocessed text
    """
    if lower:
        text = text.lower()
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_image(image_path: str, size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image data.
    
    Args:
        image_path: Path to image file
        size: Target size (width, height)
        
    Returns:
        Preprocessed image array
    """
    image = Image.open(image_path)
    image = image.resize(size)
    image_array = np.array(image)
    
    # Normalize to [0, 1]
    if image_array.max() > 1:
        image_array = image_array / 255.0
    
    return image_array


def preprocess_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Preprocess audio data.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Preprocessed audio array
    """
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        return audio
    except ImportError:
        raise ImportError("librosa is required for audio preprocessing")


def tokenize_text(text: str, max_length: Optional[int] = None) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        max_length: Maximum number of tokens
        
    Returns:
        List of tokens
    """
    tokens = text.split()
    if max_length:
        tokens = tokens[:max_length]
    return tokens

