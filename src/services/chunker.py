# creditrust-ai-complaints/src/services/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import logging
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

class TextChunker:
    """
    Handles the splitting of complaint narratives into chunks suitable for embedding.
    Uses recursive character splitting to preserve semantic meaning where possible.
    """
    
    def __init__(self):
        config = load_config('model_config.yaml')
        self.chunk_size = config['chunking']['chunk_size']
        self.chunk_overlap = config['chunking']['chunk_overlap']
        self.separators = config['chunking']['separators']
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        logger.info(f"Initialized TextChunker with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")

    def chunk_complaint(self, complaint: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a single complaint narrative into chunks while preserving metadata.
        
        Args:
            complaint: Dictionary containing 'narrative' (text) and metadata fields
            
        Returns:
            List of dictionaries, each containing a chunk of text with attached metadata
        """
        try:
            # Extract the narrative text
            text = complaint['narrative']
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid narrative text for complaint ID: {complaint.get('id', 'unknown')}")
                return []
            
            # Split the text
            chunks = self.splitter.split_text(text)
            
            # Prepare chunk documents with metadata
            chunk_docs = []
            for chunk in chunks:
                chunk_doc = {
                    'text': chunk,
                    'source_id': complaint.get('id'),
                    'product': complaint.get('product'),
                    'issue': complaint.get('issue'),
                    'sub_product': complaint.get('sub_product'),
                    'date_received': complaint.get('date_received')
                }
                chunk_docs.append(chunk_doc)
            
            logger.debug(f"Split complaint ID {complaint.get('id')} into {len(chunk_docs)} chunks")
            return chunk_docs
            
        except Exception as e:
            logger.error(f"Error chunking complaint ID {complaint.get('id', 'unknown')}: {str(e)}")
            return []

    def chunk_batch(self, complaints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of complaints into chunks.
        
        Args:
            complaints: List of complaint dictionaries
            
        Returns:
            Flattened list of all chunks from all complaints
        """
        all_chunks = []
        for complaint in complaints:
            chunks = self.chunk_complaint(complaint)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(complaints)} complaints into {len(all_chunks)} total chunks")
        return all_chunks