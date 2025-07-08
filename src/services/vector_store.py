# src/services/vector_store.py
from typing import List, Dict, Tuple, Any  # Add all required typing imports
from langchain.schema import Document
import numpy as np
import faiss
import logging
import os
from pathlib import Path

# Import embedding service (use either relative or absolute)
try:
    from .embedding_service import get_embedding_service
except ImportError:
    from src.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, embedding_service, index_path: str):
        self.embedding_service = embedding_service
        self.index_path = Path(index_path)
        self.metadata_store = []
        self.index = self._load_index()
        self._validate_components()
    
    def _validate_components(self):
        """Validate required components are initialized"""
        if not hasattr(self.embedding_service, 'embed_query'):
            raise AttributeError("Embedding service must implement embed_query()")
        if not hasattr(self.index, 'search'):
            raise AttributeError("Vector index must implement search()")

    def _load_index(self):
        """Load or create FAISS index"""
        if self.index_path.exists():
            logger.info(f"Loading index from {self.index_path}")
            index = faiss.read_index(str(self.index_path))
            metadata_path = self.index_path.with_suffix('.metadata.parquet')
            if metadata_path.exists():
                self.metadata_store = pd.read_parquet(metadata_path).to_dict('records')
            return index
        logger.info("Creating new empty index")
        return faiss.IndexFlatL2(self.embedding_service.dimension)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector store"""
        if not documents:
            return
            
        texts = [doc.get('text', '') for doc in documents]
        embeddings = np.array(
            [self.embedding_service.embed_query(text) for text in texts],
            dtype='float32'
        )
        self.index.add(embeddings)
        self.metadata_store.extend(documents)
        self._save_index()

    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if self.index.ntotal == 0:
            logger.warning("Attempted search on empty index")
            return []
        
        try:
            query_embedding = self.embedding_service.embed_query(query)
            query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
            
            # Increase search scope if initial results are poor
            search_k = min(k * 3, self.index.ntotal)
            distances, indices = self.index.search(query_vector, search_k)
            
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx >= 0:
                    doc = self._reconstruct_document(idx)
                    if doc:  # Add quality checks here
                        results.append((doc, 1/(1+score)))
            
            return sorted(results, key=lambda x: x[1], reverse=True)[:k]
        
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def _save_index(self):
        """Save index to disk"""
        faiss.write_index(self.index, str(self.index_path))
        metadata_path = self.index_path.with_suffix('.metadata.parquet')
        pd.DataFrame(self.metadata_store).to_parquet(metadata_path)

def get_vector_store() -> VectorStoreManager:
    """Factory function to get vector store instance"""
    return VectorStoreManager(
        embedding_service=get_embedding_service(),
        index_path=os.getenv('VECTOR_INDEX_PATH', 'data/vector_store/faiss.index')
    )