# src/models/embedding_service.py
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import os

# Configure environment for optimal CPU performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

class EmbeddingService:  # Note: Corrected class name (kept as EmbeddingService)
    """Optimized embedding service for CPU-only systems"""
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        # Configure torch for CPU optimization
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = False
        
        # Explicitly set device to CPU
        self.device = "cpu"
        logger.info(f"Initializing CPU-only embedding service with model: {model_name}")
        
        # Initialize model with retry and memory safety
        self.model = self._initialize_model(model_name)
        self.dimension = 384  # Hardcoded for MiniLM-L3-v2
        logger.info(f"Service initialized. Embedding dimension: {self.dimension}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _initialize_model(self, model_name: str) -> SentenceTransformer:
        """Safe model initialization with memory limits"""
        try:
            # Test memory allocation first
            test_tensor = torch.randn(384, 384, device="cpu")
            del test_tensor
            
            model = SentenceTransformer(model_name, device="cpu")
            return model
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Memory-optimized embedding generation for CPU"""
        if not texts:
            return np.array([])
            
        try:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device="cpu"
            )
        except Exception as e:
            logger.error(f"Encoding failed: {str(e)}")
            raise

    async def embed(self, text: str) -> List[float]:
        """Single text embedding with memory safety"""
        return self.encode([text], batch_size=1)[0].tolist()
    
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding with automatic memory management"""
        results = []
        for i in range(0, len(texts), 4):  # Small batch size
            batch = texts[i:i+4]
            embeddings = self.encode(batch)
            results.extend(emb.tolist() for emb in embeddings)
        return results