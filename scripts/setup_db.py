#!/usr/bin/env python3
# creditrust-ai-complaints/scripts/setup_db.py
"""
Initializes the vector database by:
1. Loading raw complaint data
2. Preprocessing and filtering
3. Chunking narratives
4. Generating embeddings
5. Creating and persisting FAISS index
"""

import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.services.preprocessor import DataPreprocessor
from src.services.chunker import TextChunker
from src.services.vector_store import VectorStoreManager
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
import time

logger = get_logger("setup_db")

def load_complaints_data(filepath: str) -> pd.DataFrame:
    """Load and validate the raw complaints CSV file."""
    try:
        logger.info(f"Loading complaints data from {filepath}")
        df = pd.read_csv(filepath, parse_dates=['Date received', 'Date sent to company'])
        
        # Validate required columns
        required_cols = {
            'Date received', 'Product', 'Sub-product', 'Issue', 
            'Consumer complaint narrative', 'Complaint ID'
        }
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Loaded {len(df)} raw complaints")
        return df
        
    except Exception as e:
        logger.error(f"Error loading complaints data: {str(e)}")
        raise

def prepare_complaints(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dictionaries with standardized field names."""
    product_mapping = {
        'Credit card': 'Credit card',
        'Credit Card': 'Credit card',
        'Payday loan': 'Personal loan',
        'Personal loan': 'Personal loan',
        'Bank account or service': 'Savings account',
        'Savings account': 'Savings account',
        'Money transfer': 'Money transfers',
        'Money transfers': 'Money transfers',
        'Consumer Loan': 'Personal loan',
        'Payday Loan': 'Personal loan'
    }
    
    complaints = []
    for _, row in df.iterrows():
        # Standardize product names and filter to only include our target products
        product = product_mapping.get(row['Product'], None)
        if product not in product_mapping.values():
            continue
            
        complaint = {
            'id': str(row['Complaint ID']),
            'date_received': row['Date received'].isoformat() if pd.notna(row['Date received']) else None,
            'product': product,
            'sub_product': row['Sub-product'] if pd.notna(row['Sub-product']) else None,
            'issue': row['Issue'] if pd.notna(row['Issue']) else None,
            'sub_issue': row['Sub-issue'] if 'Sub-issue' in row and pd.notna(row['Sub-issue']) else None,
            'narrative': row['Consumer complaint narrative'] if pd.notna(row['Consumer complaint narrative']) else None,
            'company': row['Company'] if 'Company' in row and pd.notna(row['Company']) else None,
            'state': row['State'] if 'State' in row and pd.notna(row['State']) else None,
            'consumer_disputed': row['Consumer disputed?'] if 'Consumer disputed?' in row and pd.notna(row['Consumer disputed?']) else None,
            'company_response': row['Company response to consumer'] if 'Company response to consumer' in row and pd.notna(row['Company response to consumer']) else None
        }
        
        # Only include complaints with narratives
        if complaint['narrative']:
            complaints.append(complaint)
    
    logger.info(f"Prepared {len(complaints)} complaints with narratives")
    return complaints

def main():
    try:
        # Load configuration
        config = load_config('model_config.yaml')
        raw_data_path = Path("C:/Users/tefer/creditrust-ai-complaints/data/raw/complaints.csv")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        start_time = time.time()
        df = load_complaints_data(raw_data_path)
        
        # Step 2: Prepare and filter complaints
        complaints = prepare_complaints(df)
        
        # Save filtered data
        filtered_path = processed_dir / "filtered_complaints.csv"
        pd.DataFrame(complaints).to_csv(filtered_path, index=False)
        logger.info(f"Saved filtered complaints to {filtered_path}")
        
        # Step 3: Initialize services
        preprocessor = DataPreprocessor()
        embedder = SentenceTransformer(config['embedding']['model_name'])
        chunker = TextChunker()
        vector_store = VectorStoreManager(embedder)
        
        # Step 4: Preprocess narratives
        logger.info("Preprocessing narratives...")
        for complaint in complaints:
            complaint['narrative'] = preprocessor.clean_text(complaint['narrative'])
        
        # Step 5: Chunk narratives
        logger.info("Chunking narratives...")
        chunks = chunker.chunk_batch(complaints)
        
        # Step 6: Create and save vector store
        logger.info("Creating vector store...")
        vector_store.create_index(chunks)
        vector_store.save_index()
        
        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Database setup completed in {elapsed:.2f} seconds")
        logger.info(f"Final index contains {vector_store.get_stats()['vector_count']} vectors")
        
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()