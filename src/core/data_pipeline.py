import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Optional, Generator, Union, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialTextPreprocessor:
    def __init__(self):
        self.stop_phrases = [
            "i am writing to complain about",
            "this is a complaint regarding",
            "i would like to file a complaint"
        ]
        self.product_mapping = {
            "credit card": "credit_card",
            "bnpl": "bnpl",
            "buy now pay later": "bnpl",
            "mortgage": "mortgage"
        }
        self.regulatory_keywords = {
            'violation': 'Potential Compliance Violation',
            'unauthorized': 'Unauthorized Activity',
            'overcharge': 'Pricing Issue'
        }

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning for RAG system"""
        if not isinstance(text, str) or not text.strip():
            return ""
            
        text = text.lower().strip()
        
        # Remove boilerplate
        for phrase in self.stop_phrases:
            text = re.sub(re.escape(phrase), "", text)
            
        # Remove special chars but keep financial terms
        text = re.sub(r"[^a-zA-Z0-9\s.,;?!$%-]", "", text)
        
        # Normalize monetary terms
        text = re.sub(r"\$\s*(\d+)", r"$\1", text)
        
        return text.strip()

    def extract_monetary_terms(self, text: str) -> List[float]:
        """Extract monetary values for risk assessment"""
        amounts = []
        for match in re.finditer(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text):
            try:
                amount = float(match.group(1).replace(",", ""))
                amounts.append(amount)
            except:
                continue
        return amounts

    def identify_regulatory_flags(self, text: str) -> List[str]:
        """Detect regulatory issues in text"""
        flags = []
        for term, flag in self.regulatory_keywords.items():
            if term in text.lower():
                flags.append(flag)
        return flags

class ComplaintDataPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.preprocessor = FinancialTextPreprocessor()
        self.embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
        
        # Create required directories
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "vector_ready").mkdir(exist_ok=True)

    def load_processed_data(self, file_path: Union[str, Path] = None) -> pd.DataFrame:
        """Load data with enhanced validation for RAG system"""
        default_path = self.data_dir / "processed/processed_complaints.parquet"
        path = Path(file_path) if file_path else default_path
        
        df = pd.read_parquet(path)
        
        # Validate required RAG fields
        required = ['product', 'clean_text', 'complaint_id', 'date_received']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing RAG required columns: {missing}")
            
        return df

    def prepare_for_vector_store(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert processed data to vector store format"""
        vector_docs = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing vector docs"):
            try:
                text = row['clean_text']
                if not text or len(text) < 50:  # Minimum length for meaningful embeddings
                    continue
                    
                monetary_terms = self.preprocessor.extract_monetary_terms(text)
                regulatory_flags = self.preprocessor.identify_regulatory_flags(text)
                
                vector_docs.append({
                    'text': text,
                    'product': row['product'],
                    'date': row['date_received'].strftime('%Y-%m-%d'),
                    'complaint_id': row['complaint_id'],
                    'monetary_terms': monetary_terms,
                    'regulatory_flags': regulatory_flags,
                    'embedding': self.embedder.encode(text, convert_to_numpy=True).tolist()
                })
            except Exception as e:
                logger.error(f"Error processing row {row['complaint_id']}: {str(e)}")
                continue
                
        return vector_docs

    def save_vector_ready_data(self, documents: List[Dict[str, Any]], batch_size: int = 1000):
        """Save in chunks for memory efficiency"""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            output_path = self.data_dir / f"vector_ready/batch_{i//batch_size}.json"
            with open(output_path, 'w') as f:
                json.dump(batch, f)
            logger.info(f"Saved vector batch {i//batch_size} with {len(batch)} docs")

    def run_rag_preparation(self, max_docs: int = 50000):
        """Complete pipeline from processed data to RAG-ready format"""
        try:
            df = self.load_processed_data()
            if len(df) > max_docs:
                df = df.sample(max_docs, random_state=42)
                
            vector_docs = self.prepare_for_vector_store(df)
            self.save_vector_ready_data(vector_docs)
            
            logger.info(f"Successfully prepared {len(vector_docs)} documents for RAG system")
            return vector_docs
        except Exception as e:
            logger.error(f"RAG preparation failed: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = ComplaintDataPipeline()
    pipeline.run_rag_preparation(max_docs=10000)