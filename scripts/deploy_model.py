#!/usr/bin/env python3
# creditrust-ai-complaints/scripts/deploy_model.py
"""
Deploys the RAG model as a FastAPI service with:
1. Pre-loaded vector store
2. Embedding model
3. LLM for generation
4. API endpoints for querying
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from src.services.vector_store import VectorStoreManager
from src.core.rag_engine import RAGEngine
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from typing import Optional
import os
import time

logger = get_logger("deploy_model")

app = FastAPI(
    title="CrediTrust AI Complaints Analyzer",
    description="RAG-powered API for analyzing financial service complaints",
    version="1.0.0"
)

# Global variables for initialized components
vector_store = None
rag_engine = None

class ModelLoader:
    """Handles loading of ML models and vector store."""
    
    @staticmethod
    def load_models():
        """Loads all required models and services."""
        try:
            start_time = time.time()
            config = load_config('model_config.yaml')
            
            # Load embedding model
            logger.info(f"Loading embedding model: {config['embedding']['model_name']}")
            embedder = SentenceTransformer(config['embedding']['model_name'])
            
            # Load vector store
            global vector_store
            vector_store = VectorStoreManager(embedder)
            if not vector_store.load_index():
                raise RuntimeError("Failed to load vector store index")
            
            # Initialize RAG engine
            global rag_engine
            rag_engine = RAGEngine(
                vector_store=vector_store,
                llm_config=config['llm']
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Models loaded in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

@app.on_event("startup")
async def startup_event():
    """Initialize models when the application starts."""
    ModelLoader.load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_ready": vector_store is not None,
        "rag_engine_ready": rag_engine is not None
    }

@app.get("/api/search")
async def search_complaints(
    query: str,
    product: Optional[str] = None,
    limit: int = 5
):
    """
    Search complaints using semantic search.
    
    Args:
        query: Natural language search query
        product: Optional product filter (e.g., "Credit card")
        limit: Maximum number of results to return
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not loaded")
            
        filter_by = {'product': product} if product else None
        results = vector_store.similarity_search(query, k=limit, filter_by=filter_by)
        
        # Format results for API response
        formatted_results = []
        for res in results:
            formatted = {
                'text': res['text'],
                'product': res['product'],
                'issue': res['issue'],
                'date_received': res['date_received'],
                'similarity_score': res['score'],
                'complaint_id': res['source_id']
            }
            formatted_results.append(formatted)
        
        return {
            "query": query,
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ask")
async def ask_question(
    question: str,
    product: Optional[str] = None,
    analyze_sentiment: bool = False
):
    """
    Get an LLM-generated answer based on relevant complaints.
    
    Args:
        question: Natural language question
        product: Optional product filter
        analyze_sentiment: Whether to include sentiment analysis
    """
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not loaded")
            
        response = rag_engine.generate_response(
            question=question,
            product_filter=product,
            analyze_sentiment=analyze_sentiment
        )
        
        return {
            "question": question,
            "answer": response['answer'],
            "source_complaints": response['sources'],
            "sentiment_analysis": response.get('sentiment')
        }
        
    except Exception as e:
        logger.error(f"Question answering error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def run_server():
    """Run the FastAPI server with uvicorn."""
    config = load_config('model_config.yaml')
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        workers=int(os.getenv("WORKERS", 1))
    
if __name__ == "__main__":
    run_server()