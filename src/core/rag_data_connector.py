import json
from pathlib import Path
from typing import List, Dict
import logging
from src.services.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class RAGDataConnector:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def load_vector_ready_data(self, data_dir: str = "data/vector_ready") -> List[Dict]:
        """Load all prepared vector documents"""
        docs = []
        for path in Path(data_dir).glob("*.json"):
            with open(path, 'r') as f:
                docs.extend(json.load(f))
        return docs

    def populate_vector_store(self, batch_size: int = 1000):
        """Load and index documents in batches"""
        documents = self.load_vector_ready_data()
        logger.info(f"Found {len(documents)} vector-ready documents")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(batch)
            logger.info(f"Indexed batch {i//batch_size} ({len(batch)} docs)")
        
        logger.info(f"Total documents in vector store: {self.vector_store.index.ntotal}")