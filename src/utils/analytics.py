import logging
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import json

logger = logging.getLogger(__name__)

class AnalyticsTracker:
    def __init__(self):
        self.queries = []
        self.responses = []
        self.sources = []
        
    def log_retrieval(self, query: str, documents: List[Dict[str, Any]]):
        """Track retrieval performance"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_sources": len(documents),
            "avg_confidence": sum(doc.get('score', 0) for doc in documents) / len(documents) if documents else 0,
            "top_source": documents[0]['metadata']['product'] if documents else None
        }
        self.sources.append(entry)
        
    def log_generation(self, query: str, response: str):
        """Track generation metrics"""
        self.responses.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response_length": len(response),
            "contains_citation": "[" in response
        })
    
    def get_retrieval_metrics(self) -> pd.DataFrame:
        """Generate retrieval analytics"""
        return pd.DataFrame(self.sources)
    
    def get_generation_metrics(self) -> pd.DataFrame:
        """Generate response analytics"""
        return pd.DataFrame(self.responses)
    
    def generate_dashboard(self):
        """Create interactive analytics dashboard"""
        retrieval_df = self.get_retrieval_metrics()
        generation_df = self.get_generation_metrics()
        
        if not retrieval_df.empty:
            fig1 = px.line(
                retrieval_df,
                x="timestamp",
                y="avg_confidence",
                title="Retrieval Confidence Over Time"
            )
            
            fig2 = px.bar(
                retrieval_df,
                x="top_source",
                y="num_sources",
                title="Top Sources by Frequency"
            )
        else:
            fig1, fig2 = None, None
            
        if not generation_df.empty:
            fig3 = px.histogram(
                generation_df,
                x="response_length",
                title="Response Length Distribution"
            )
        else:
            fig3 = None
            
        return {
            "retrieval_confidence": fig1,
            "source_frequency": fig2,
            "response_length": fig3
        }
    
    def save_to_file(self, path: str = "analytics.json"):
        """Persist analytics data"""
        with open(path, 'w') as f:
            json.dump({
                "retrievals": self.sources,
                "generations": self.responses
            }, f)