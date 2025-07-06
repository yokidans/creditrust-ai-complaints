# src/models/memory_efficient_embedder.py
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

class MemoryEfficientEmbedder:
    """
    A wrapper for SentenceTransformer that provides memory-efficient embedding.
    Implements the minimum interface needed for VectorStoreManager.
    """
    def __init__(self, model_name: str = 'paraphrase-MiniLM-L3-v2'):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of strings to embed
            kwargs: Additional arguments to pass to the underlying model
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        return self.model.encode(texts, **kwargs)
    
    def get_sentence_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embeddings.
        """
        return self.model.get_sentence_embedding_dimension()