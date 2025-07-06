# src/core/retrieval.py
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
from pydantic import BaseModel
import logging
from datetime import datetime
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QueryType(str, Enum):
    """Enumeration of different query types the retriever can handle."""
    TREND = "trend"
    SPECIFIC = "specific"
    COMPARATIVE = "comparative"
    SENTIMENT = "sentiment"

@dataclass
class RetrievedComplaint:
    """Dataclass representing a retrieved complaint with relevant metadata."""
    complaint_id: str
    text: str
    product: str
    date: datetime
    similarity_score: float
    metadata: Dict = None

class QueryAnalyzer:
    """Analyzes queries to determine their type for appropriate processing."""
    def __init__(self):
        # This would be initialized with an ML model in production
        pass

    def analyze_query(self, query: str) -> QueryType:
        """
        Determines the type of query based on keywords.
        
        Args:
            query: The input query string
            
        Returns:
            QueryType enum value
        """
        query = query.lower()
        if any(word in query for word in ["trend", "over time", "last month", "recent"]):
            return QueryType.TREND
        elif any(word in query for word in ["compare", "vs", "versus", "difference"]):
            return QueryType.COMPARATIVE
        elif any(word in query for word in ["sentiment", "feel", "angry", "happy"]):
            return QueryType.SENTIMENT
        else:
            return QueryType.SPECIFIC

class HierarchicalRetriever:
    """Hierarchical complaint retriever with multi-stage processing."""
    def __init__(self, vector_store, graph_store=None):
        """
        Initialize the retriever with required stores.
        
        Args:
            vector_store: Initialized VectorStoreManager instance
            graph_store: Optional graph store for advanced relationships
        """
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.query_analyzer = QueryAnalyzer()
        self.last_query_type = None

    async def retrieve(
        self,
        query_text: str,
        products: Optional[List[str]] = None,
        date_range: Optional[List[datetime]] = None,
        top_k: int = 10
    ) -> List[RetrievedComplaint]:
        """
        Main retrieval method with hierarchical processing.
        
        Args:
            query_text: Search query string
            products: Optional list of products to filter by
            date_range: Optional date range [start, end]
            top_k: Number of results to return
            
        Returns:
            List of RetrievedComplaint objects
            
        Raises:
            ValueError: For invalid inputs
            HTTPException: For processing errors
        """
        try:
            if not query_text:
                raise ValueError("Query text cannot be empty")
                
            if date_range and len(date_range) != 2:
                raise ValueError("Date range must contain exactly 2 dates")

            # Analyze query type
            query_type = self.query_analyzer.analyze_query(query_text)
            self.last_query_type = query_type

            # First-stage vector retrieval
            base_results = self.vector_store.similarity_search(
                query_text,
                k=top_k*3,  # Get more results for hierarchical processing
                filter_by={"product": products} if products else None
            )

            # Second-stage processing based on query type
            if query_type == QueryType.TREND:
                processed = self._process_trend_query(base_results, date_range)
            elif query_type == QueryType.COMPARATIVE:
                processed = self._process_comparative_query(base_results, query_text)
            elif query_type == QueryType.SENTIMENT:
                processed = self._process_sentiment_query(base_results)
            else:
                processed = self._process_specific_query(base_results)

            return processed[:top_k]  # Return only the requested number of results

        except Exception as e:
            logger.error(f"Retrieval error for query '{query_text}': {str(e)}")
            raise

    def _process_trend_query(self, results: List[Dict], date_range: Optional[List[datetime]]) -> List[RetrievedComplaint]:
        """Process trend-based queries with date filtering."""
        filtered = results
        if date_range:
            filtered = [
                r for r in results 
                if date_range[0] <= datetime.strptime(r['date_received'], '%Y-%m-%d') <= date_range[1]
            ]
        return [self._to_retrieved_complaint(r) for r in filtered]

    def _process_comparative_query(self, results: List[Dict], query_text: str) -> List[RetrievedComplaint]:
        """Process comparative queries using graph relationships."""
        if self.graph_store:
            compared = self.graph_store.compare_entities(results, query_text)
            return [self._to_retrieved_complaint(r) for r in compared]
        return [self._to_retrieved_complaint(r) for r in results]  # Fallback to vector results

    def _process_sentiment_query(self, results: List[Dict]) -> List[RetrievedComplaint]:
        """Filter and sort by sentiment scores."""
        scored = sorted(results, key=lambda x: x.get('sentiment_score', 0), reverse=True)
        return [self._to_retrieved_complaint(r) for r in scored]

    def _process_specific_query(self, results: List[Dict]) -> List[RetrievedComplaint]:
        """Return most relevant specific complaints."""
        return [self._to_retrieved_complaint(r) for r in sorted(results, key=lambda x: x['score'], reverse=True)]

    def _to_retrieved_complaint(self, result: Dict) -> RetrievedComplaint:
        """Convert raw result dictionary to RetrievedComplaint dataclass."""
        try:
            return RetrievedComplaint(
                complaint_id=result.get('id', ''),
                text=result.get('text', ''),
                product=result.get('product', ''),
                date=datetime.strptime(result.get('date_received', '1970-01-01'), '%Y-%m-%d'),
                similarity_score=result.get('score', 0),
                metadata={
                    'sub_product': result.get('sub_product'),
                    'issue': result.get('issue'),
                    'company': result.get('company'),
                    'sentiment': result.get('sentiment')
                }
            )
        except ValueError as e:
            logger.warning(f"Error parsing complaint date: {str(e)}")
            return RetrievedComplaint(
                complaint_id=result.get('id', ''),
                text=result.get('text', ''),
                product=result.get('product', ''),
                date=datetime(1970, 1, 1),  # Default date on error
                similarity_score=result.get('score', 0),
                metadata=result  # Include all original data as fallback
            )

    def get_last_query_type(self) -> Optional[QueryType]:
        """Returns the type of the last processed query."""
        return self.last_query_type