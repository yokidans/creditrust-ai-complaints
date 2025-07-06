# src/services/vector_store.py
# src/services/vector_store.py
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
    """Robust vector store with improved embedder handling"""
    
    def __init__(self, embedder: Any, dimension: Optional[int] = None):
        """
        Args:
            embedder: Must implement encode() or be callable
            dimension: Optional pre-specified embedding dimension
        """
        self.embedder = self._validate_embedder(embedder)
        self.index = None
        self.metadata_store = []
        self.dimension = dimension or self._infer_dimension()
        
        # Load config
        config = load_config('model_config.yaml')
        self.vector_store_path = Path(config['vector_store']['path'])
        os.makedirs(self.vector_store_path.parent, exist_ok=True)
    
    def _validate_embedder(self, embedder) -> Any:
        """Ensure embedder has required interface"""
        if hasattr(embedder, 'encode') and callable(embedder.encode):
            return embedder
        if callable(embedder):
            return embedder
        raise ValueError(
            "Embedder must implement encode() method or be callable. "
            f"Got: {type(embedder)}"
        )
    
    def _infer_dimension(self) -> int:
        """Safely determine embedding dimension"""
        try:
            test_embedding = (
                self.embedder.encode(["test"]) 
                if hasattr(self.embedder, 'encode')
                else self.embedder(["test"])
            )
            return len(test_embedding[0])
        except Exception as e:
            raise ValueError(
                f"Could not determine embedding dimension: {str(e)}"
            ) from e

    def create_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Create FAISS index from text chunks with metadata"""
        if not chunks:
            raise ValueError("Empty chunks list provided")
            
        try:
            texts = [chunk['text'] for chunk in chunks]
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            embeddings = self._embedding_fn(texts)
            embeddings = np.array(embeddings).astype('float32')
            
            # Validate embeddings
            if embeddings.shape != (len(chunks), self.dimension):
                raise ValueError(f"Embedding shape mismatch: {embeddings.shape}")
            
            # Create index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            self.metadata_store = chunks
            
            logger.info(f"Created index with {len(chunks)} vectors (dim={self.dimension})")
            
        except Exception as e:
            logger.error(f"Index creation failed: {str(e)}")
            raise

    def save_index(self) -> None:
        """Save index and metadata to disk"""
        if not self.index:
            raise ValueError("No index to save")
            
        try:
            faiss.write_index(self.index, str(self.vector_store_path))
            
            metadata_path = self.vector_store_path.with_suffix('.metadata.pkl')
            pd.DataFrame(self.metadata_store).to_pickle(metadata_path)
            
            logger.info(f"Saved index to {self.vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise

    def load_index(self) -> bool:
        """Load index from disk"""
        try:
            if not self.vector_store_path.exists():
                logger.warning("Index file not found")
                return False
                
            self.index = faiss.read_index(str(self.vector_store_path))
            
            metadata_path = self.vector_store_path.with_suffix('.metadata.pkl')
            self.metadata_store = pd.read_pickle(metadata_path).to_dict('records')
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False

    def similarity_search(self, query: str, k: int = 5, **filters) -> List[Dict[str, Any]]:
        """
        Search for similar complaints
        
        Args:
            query: Search text
            k: Number of results
            filters: Metadata filters (e.g., product='credit_card')
            
        Returns:
            List of matching complaints with scores
        """
        if not self.index:
            raise ValueError("Index not initialized")
            
        try:
            query_embedding = self._embedding_fn([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx >= 0:
                    result = self.metadata_store[idx].copy()
                    result['score'] = float(score)
                    
                    # Apply filters
                    if filters and not all(
                        result.get(k) == v for k, v in filters.items()
                    ):
                        continue
                        
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            'vector_count': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'metadata_fields': list(self.metadata_store[0].keys()) if self.metadata_store else []
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Return health status of the vector store"""
        return {
            'index_initialized': self.index is not None,
            'vector_count': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'metadata_loaded': len(self.metadata_store) > 0,
            'embedder_ready': hasattr(self.embedder, 'encode') or callable(self.embedder)
        }