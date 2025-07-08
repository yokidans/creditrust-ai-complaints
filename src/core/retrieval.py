# src/core/retrieval.py
from typing import List, Dict, Optional, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
import logging
from pydantic import BaseModel, validator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class QueryType(str, Enum):
    """Enumeration of different query types the retriever can handle."""
    TREND = "trend"
    SPECIFIC = "specific"
    COMPARATIVE = "comparative"
    SENTIMENT = "sentiment"

class ComplaintMetadata(BaseModel):
    """Structured metadata for complaints using Pydantic for validation"""
    sub_product: Optional[str] = None
    issue: Optional[str] = None
    company: Optional[str] = None
    sentiment: Optional[float] = None

@dataclass
class RetrievedComplaint:
    """Dataclass representing a retrieved complaint with relevant metadata."""
    complaint_id: str
    text: str
    product: str
    date: datetime
    similarity_score: float
    metadata: ComplaintMetadata

    @classmethod
    def from_dict(cls, data: Dict) -> 'RetrievedComplaint':
        """Factory method to create from dictionary with validation"""
        try:
            date = datetime.strptime(data.get('date_received', ''), '%Y-%m-%d')
        except (ValueError, TypeError):
            date = datetime(1970, 1, 1)
            logger.warning(f"Invalid date format in complaint {data.get('id')}")

        return cls(
            complaint_id=data.get('id', ''),
            text=data.get('text', ''),
            product=data.get('product', ''),
            date=date,
            similarity_score=float(data.get('score', 0)),
            metadata=ComplaintMetadata(**{
                'sub_product': data.get('sub_product'),
                'issue': data.get('issue'),
                'company': data.get('company'),
                'sentiment': data.get('sentiment')
            })
        )

class QueryAnalyzer:
    """Enhanced query analyzer with configurable patterns"""
    def __init__(self):
        self.patterns = {
            QueryType.TREND: ["trend", "over time", "last month", "recent", "historical"],
            QueryType.COMPARATIVE: ["compare", "vs", "versus", "difference", "contrast"],
            QueryType.SENTIMENT: ["sentiment", "feel", "angry", "happy", "frustrated"]
        }

    def analyze_query(self, query: str) -> QueryType:
        """Enhanced query type detection with pattern matching"""
        query_lower = query.lower()
        for qtype, keywords in self.patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return qtype
        return QueryType.SPECIFIC

class HierarchicalRetriever:
    """Enhanced hierarchical retriever with better error handling and logging"""
    
    def __init__(self, vector_store, graph_store=None):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.query_analyzer = QueryAnalyzer()
        self._last_query_type = None

    async def retrieve(
        self,
        query_text: str,
        products: Optional[List[str]] = None,
        date_range: Optional[List[datetime]] = None,
        top_k: int = 10
    ) -> List[RetrievedComplaint]:
        """
        Enhanced retrieval with better validation and error handling
        """
        self._validate_inputs(query_text, date_range)
        
        try:
            query_type = self.query_analyzer.analyze_query(query_text)
            self._last_query_type = query_type

            # Initial broad retrieval
            base_results = await self._initial_retrieval(query_text, products, top_k)
            
            # Query-type specific processing
            processor = {
                QueryType.TREND: self._process_trend_query,
                QueryType.COMPARATIVE: self._process_comparative_query,
                QueryType.SENTIMENT: self._process_sentiment_query,
                QueryType.SPECIFIC: self._process_specific_query
            }[query_type]

            processed = processor(base_results, query_text, date_range)
            return processed[:top_k]

        except Exception as e:
            logger.error(f"Retrieval failed for query '{query_text}': {str(e)}", exc_info=True)
            raise ValueError(f"Could not process query: {str(e)}")

    def _validate_inputs(self, query_text: str, date_range: Optional[List[datetime]]):
        """Validate all input parameters"""
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        if date_range and len(date_range) != 2:
            raise ValueError("Date range must contain exactly 2 dates")
        if date_range and date_range[0] > date_range[1]:
            raise ValueError("Start date must be before end date")

    async def _initial_retrieval(
        self,
        query_text: str,
        products: Optional[List[str]],
        top_k: int
    ) -> List[Dict]:
        """Perform initial vector store retrieval"""
        return await self.vector_store.similarity_search(
            query_text,
            k=top_k*3,  # Over-fetch for hierarchical processing
            filter_by={"product": products} if products else None
        )

    def _process_trend_query(
        self,
        results: List[Dict],
        query_text: str,
        date_range: Optional[List[datetime]]
    ) -> List[RetrievedComplaint]:
        """Process trend queries with date filtering"""
        filtered = results
        if date_range:
            filtered = [
                r for r in results 
                if self._is_within_date_range(r, date_range)
            ]
        return [RetrievedComplaint.from_dict(r) for r in filtered]

    def _process_comparative_query(
        self,
        results: List[Dict],
        query_text: str,
        date_range: Optional[List[datetime]] = None
    ) -> List[RetrievedComplaint]:
        """Process comparative queries using graph relationships"""
        if self.graph_store:
            try:
                compared = self.graph_store.compare_entities(results, query_text)
                return [RetrievedComplaint.from_dict(r) for r in compared]
            except Exception as e:
                logger.warning(f"Graph comparison failed, falling back to vector results: {str(e)}")
        
        # Fallback to vector results
        return [RetrievedComplaint.from_dict(r) for r in results]

    def _process_sentiment_query(
        self,
        results: List[Dict],
        query_text: str = None,
        date_range: Optional[List[datetime]] = None
    ) -> List[RetrievedComplaint]:
        """Filter and sort by sentiment scores"""
        scored = sorted(
            results,
            key=lambda x: float(x.get('sentiment_score', 0)),
            reverse=True
        )
        return [RetrievedComplaint.from_dict(r) for r in scored]

    def _process_specific_query(
        self,
        results: List[Dict],
        query_text: str = None,
        date_range: Optional[List[datetime]] = None
    ) -> List[RetrievedComplaint]:
        """Return most relevant specific complaints"""
        return [
            RetrievedComplaint.from_dict(r) 
            for r in sorted(results, key=lambda x: float(x['score']), reverse=True)
        ]

    def _is_within_date_range(self, result: Dict, date_range: List[datetime]) -> bool:
        """Helper to check if a result is within date range"""
        try:
            result_date = datetime.strptime(result.get('date_received', ''), '%Y-%m-%d')
            return date_range[0] <= result_date <= date_range[1]
        except (ValueError, TypeError):
            logger.warning(f"Invalid date in result: {result.get('date_received')}")
            return False

    @property
    def last_query_type(self) -> Optional[QueryType]:
        """Get the type of the last processed query"""
        return self._last_query_type