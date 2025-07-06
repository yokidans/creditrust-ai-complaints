import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Optional, Generator
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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
            "i would like to file a complaint",
            "please be advised that",
            "i am dissatisfied with"
        ]
        self.product_mapping = {
            "credit card": "credit_card",
            "credit cards": "credit_card",
            "personal loan": "personal_loan",
            "personal loans": "personal_loan",
            "bnpl": "bnpl",
            "buy now pay later": "bnpl",
            "savings account": "savings",
            "savings accounts": "savings",
            "money transfer": "money_transfer",
            "money transfers": "money_transfer",
            "mortgage": "mortgage",
            "debt collection": "debt_collection"
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize complaint text"""
        try:
            if not isinstance(text, str) or not text.strip():
                return ""
                
            text = text.lower().strip()
            
            # Remove stop phrases
            for phrase in self.stop_phrases:
                text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
                
            # Remove special characters but keep basic punctuation
            text = re.sub(r"[^a-zA-Z0-9\s.,;?!-]", "", text)
            
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return ""

    def map_product(self, product: str) -> Optional[str]:
        """Map raw product names to standardized categories"""
        try:
            if not isinstance(product, str):
                return None
                
            product = product.lower().strip()
            for k, v in self.product_mapping.items():
                if k in product:
                    return v
            return None
        except Exception as e:
            logger.error(f"Error mapping product: {str(e)}")
            return None


class ComplaintDataPipeline:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.preprocessor = FinancialTextPreprocessor()
        
        # Create directories if they don't exist
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    def _chunked_loader(self, file_path: str, chunksize: int = 10000) -> Generator[pd.DataFrame, None, None]:
        """Load CSV in chunks to conserve memory"""
        column_mapping = {
            "Date received": "date_received",
            "Product": "product",
            "Sub-product": "sub_product",
            "Issue": "issue",
            "Sub-issue": "sub_issue",
            "Consumer complaint narrative": "consumer_complaint_narrative",
            "Company public response": "company_public_response",
            "Company": "company",
            "Complaint ID": "complaint_id"
        }
        
        for chunk in pd.read_csv(
            file_path,
            parse_dates=["Date received"],
            dtype={
                "Complaint ID": str,
                "Consumer complaint narrative": str,
                "Product": str,
                "Sub-product": str,
                "Issue": str,
                "Sub-issue": str,
                "Company public response": str,
                "Company": str
            },
            chunksize=chunksize,
            low_memory=False,
            na_values=["", "NA", "N/A", "NaN", "null"]
        ):
            chunk = chunk.rename(columns=column_mapping)
            # Convert all string columns to str type explicitly
            for col in column_mapping.values():
                if col in chunk.columns and pd.api.types.is_string_dtype(chunk[col]):
                    chunk[col] = chunk[col].astype(str).fillna("")
            yield chunk

    def load_raw_data(self, file_path: str, max_rows: int = 100000) -> pd.DataFrame:
        """Load and validate raw complaint data in chunks with row limit"""
        try:
            chunks = []
            required_cols = [
                "date_received", "product", "sub_product", "issue",
                "sub_issue", "consumer_complaint_narrative",
                "company_public_response", "company", "complaint_id"
            ]
            
            total_loaded = 0
            for chunk in self._chunked_loader(file_path):
                # Validate columns in each chunk
                if not all(col in chunk.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in chunk.columns]
                    raise ValueError(f"Missing required columns: {missing}")
                
                # Check if we need the full chunk or just part of it
                remaining_rows = max_rows - total_loaded
                if remaining_rows <= 0:
                    break
                    
                if len(chunk) > remaining_rows:
                    chunk = chunk.iloc[:remaining_rows]
                
                chunks.append(chunk)
                total_loaded += len(chunk)
                
                if total_loaded >= max_rows:
                    break
            
            if not chunks:
                raise ValueError("No data was loaded - check your input file")
            
            return pd.concat(chunks, ignore_index=True)
        
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise

    def _safe_text_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle text processing with proper type checking"""
        try:
            df = df.copy()
            
            # Ensure we're working with strings and handle nulls
            df["consumer_complaint_narrative"] = (
                df["consumer_complaint_narrative"]
                .astype(str)
                .replace(["nan", "None", "null", ""], "")
                .str.strip()
            )
            
            # Filter empty narratives
            df = df[df["consumer_complaint_narrative"] != ""]
            
            if len(df) == 0:
                return df
            
            # Clean text in parallel
            with ThreadPoolExecutor() as executor:
                cleaned_texts = list(tqdm(
                    executor.map(self.preprocessor.clean_text, df["consumer_complaint_narrative"]),
                    total=len(df),
                    desc="Cleaning texts"
                ))
            
            df["clean_text"] = cleaned_texts
            
            # Filter out empty or very short texts
            df = df[
                df["clean_text"].apply(
                    lambda x: isinstance(x, str) and len(x) > 20
                )
            ]
            
            return df
        except Exception as e:
            logger.error(f"Error in text processing: {str(e)}")
            # Return the original DataFrame if processing fails
            return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform complaint data in memory-efficient way"""
        try:
            # Process in batches if DataFrame is large
            if len(df) > 50000:
                return self._batch_preprocess(df)
            
            # Ensure product column is string type
            df["product"] = df["product"].astype(str).fillna("")
            
            # Map products
            df["product"] = df["product"].apply(self.preprocessor.map_product)
            df = df[df["product"].notna()]
            
            if len(df) == 0:
                return df
            
            # Process text
            df = self._safe_text_processing(df)
            
            if len(df) == 0:
                return df
            
            # Add metadata
            df["year_month"] = df["date_received"].dt.to_period("M")
            df["word_count"] = df["clean_text"].str.split().str.len()
            df["complaint_id"] = df["complaint_id"].astype(str).str.strip()
            
            return df
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            # Return the original DataFrame if processing fails
            return df

    def _batch_preprocess(self, df: pd.DataFrame, batch_size: int = 10000) -> pd.DataFrame:
        """Process large DataFrames in batches"""
        processed_chunks = []
        total_batches = (len(df) // batch_size) + 1
        
        for i in tqdm(range(total_batches), desc="Processing batches"):
            batch = df.iloc[i*batch_size : (i+1)*batch_size].copy()
            try:
                processed_chunk = self.preprocess_data(batch)
                if not processed_chunk.empty:
                    processed_chunks.append(processed_chunk)
            except Exception as e:
                logger.error(f"Error processing batch {i}: {str(e)}")
                logger.debug(f"Problematic batch data:\n{batch.head()}")
                continue
        
        if processed_chunks:
            return pd.concat(processed_chunks, ignore_index=True)
        return pd.DataFrame()

    def run_pipeline(self, input_file: str, output_base_name: str = "processed_complaints", max_rows: int = 100000):
        """Execute the complete data processing pipeline"""
        raw_path = self.data_dir / "raw" / input_file
        processed_dir = self.data_dir / "processed"
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Input file not found: {raw_path}")
        
        logger.info(f"Starting data pipeline (max_rows={max_rows})")
        try:
            # Load data in chunks with row limit
            df = self.load_raw_data(raw_path, max_rows=max_rows)
            logger.info(f"Loaded {len(df)} raw complaints")
            
            # Process data (in batches if needed)
            processed_df = self.preprocess_data(df)
            logger.info(f"Processed {len(processed_df)} complaints after cleaning")
            
            if processed_df.empty:
                logger.warning("No complaints were processed - check your input data")
                return processed_df
            
            # Generate output file paths
            parquet_path = processed_dir / f"{output_base_name}.parquet"
            csv_path = processed_dir / f"{output_base_name}.csv"
            
            # Save in parquet format
            processed_df.to_parquet(parquet_path)
            
            # Save in CSV format (in chunks if large)
            if len(processed_df) > 50000:
                processed_df.to_csv(csv_path, index=False, chunksize=10000)
            else:
                processed_df.to_csv(csv_path, index=False)
            
            logger.info(f"Saved processed data to:\n- {parquet_path}\n- {csv_path}")
            return processed_df
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        pipeline = ComplaintDataPipeline()
        result = pipeline.run_pipeline(
            input_file="complaints.csv",
            output_base_name="processed_complaints_100k",
            max_rows=100000
        )
        print(f"\nPipeline completed successfully. Processed {len(result)} complaints.")
        print(f"Output files created in 'data/processed' directory:")
        print("- processed_complaints_100k.parquet")
        print("- processed_complaints_100k.csv")
    except Exception as e:
        logger.exception("Pipeline execution failed")
        print("\nPipeline failed - check logs for details")