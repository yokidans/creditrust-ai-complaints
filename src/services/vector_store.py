from typing import List, Dict, Any, Optional, Union
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

class VectorStoreManager:
    """Enhanced vector store with hybrid search capabilities"""
    
    def __init__(self, embedder: Any, dimension: Optional[int] = None):
        """
        Args:
            embedder: Embedding model with encode() method
            dimension: Optional pre-specified embedding dimension
        """
        self.embedder = self._validate_embedder(embedder)
        self.index = None
        self.metadata_store = []
        self.dimension = dimension or self._infer_dimension()
        self._init_storage()

    def _init_storage(self):
        """Initialize storage paths from config"""
        config = load_config('model_config.yaml')
        self.vector_store_path = Path(config['vector_store']['path'])
        os.makedirs(self.vector_store_path.parent, exist_ok=True)

    def _validate_embedder(self, embedder) -> Any:
        """Ensure embedder has required interface"""
        if not (hasattr(embedder, 'encode') and callable(embedder.encode)):
            raise ValueError("Embedder must implement encode() method")
        return embedder

    def _infer_dimension(self) -> int:
        """Safely determine embedding dimension"""
        test_embedding = self.embedder.encode(["test"])
        return len(test_embedding[0])

    def create_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Create FAISS index from text chunks"""
        if not chunks:
            raise ValueError("Empty chunks list provided")
            
        texts = [chunk['text'] for chunk in chunks]
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        embeddings = np.array(self.embedder.encode(texts)).astype('float32')
        self._validate_embeddings(embeddings, len(chunks))
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        self.metadata_store = chunks
        logger.info(f"Created index with {len(chunks)} vectors")

    def _validate_embeddings(self, embeddings: np.ndarray, expected_count: int):
        """Validate embedding dimensions"""
        if embeddings.shape != (expected_count, self.dimension):
            raise ValueError(f"Embedding shape mismatch: {embeddings.shape}")

    def save_index(self) -> None:
        """Persist index to disk"""
        if not self.index:
            raise ValueError("No index to save")
            
        faiss.write_index(self.index, str(self.vector_store_path))
        metadata_path = self.vector_store_path.with_suffix('.parquet')
        pd.DataFrame(self.metadata_store).to_parquet(metadata_path)
        logger.info(f"Saved index to {self.vector_store_path}")

    def load_index(self) -> bool:
        """Load index from disk"""
        if not self.vector_store_path.exists():
            logger.warning("Index file not found")
            return False
            
        self.index = faiss.read_index(str(self.vector_store_path))
        metadata_path = self.vector_store_path.with_suffix('.metadata.parquet')
        self.metadata_store = pd.read_parquet(metadata_path).to_dict('records')
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        return True

    def similarity_search(self, query: str, k: int = 5, **filters) -> List[Dict[str, Any]]:
        """Search with metadata filtering"""
        if not self.index:
            raise ValueError("Index not initialized")
            
        query_embedding = np.array(self.embedder.encode([query])).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        return [
            {**self.metadata_store[idx], 'score': float(score)}
            for idx, score in zip(indices[0], distances[0])
            if idx >= 0 and self._matches_filters(self.metadata_store[idx], filters)
        ]

    def _matches_filters(self, item: Dict, filters: Dict) -> bool:
        """Check if item matches all filters"""
        return all(item.get(k) == v for k, v in filters.items())

def get_vector_store(embedder=None, force_rebuild=False) -> VectorStoreManager:
    """Factory function with smart initialization"""
    from src.models.embedding_service import EmbeddingService
    
    manager = VectorStoreManager(embedder or EmbeddingService())
    
    if not force_rebuild and manager.load_index():
        return manager
        
    from src.core.data_pipeline import load_processed_data
    complaints = load_processed_data()
    chunks = [{
        'text': row['clean_text'],
        'product': row['product'],
        'date': row['date_received'].isoformat(),
        'complaint_id': row['complaint_id']
    } for _, row in complaints.iterrows()]
    
    manager.create_index(chunks)
    manager.save_index()
    return manager