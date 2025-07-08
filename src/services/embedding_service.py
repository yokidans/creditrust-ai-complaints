# src/services/embedding_service.py
from typing import List, Dict, Any  # Ensure all typing imports
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import os

# Configure environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2"):
        torch.set_num_threads(1)
        self.model = self._initialize_model(model_name)
        self.dimension = 384
        logger.info(f"Initialized embedding service with dimension {self.dimension}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _initialize_model(self, model_name: str) -> SentenceTransformer:
        try:
            return SentenceTransformer(model_name, device="cpu")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[List[float]]:
        texts = [doc.get('text', '') for doc in documents]
        return self.model.encode(texts, convert_to_numpy=True).tolist()

def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()