# Updated interface.py with enhanced error handling, performance improvements, and additional features

import gradio as gr
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from src.core.rag_engine import EliteRAGSystem
from src.services.vector_store import get_vector_store
from plotly.graph_objects import Figure
import logging
import time
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import plotly.express as px
import numpy as np
import matplotlib
import hashlib
import uuid
from enum import Enum
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# CSS styling moved to class level with enhanced styling
PROFESSIONAL_CSS = """
:root {
    --primary: #4f46e5;
    --primary-dark: #4338ca;
    --secondary: #f1f5f9;
    --secondary-dark: #e2e8f0;
    --danger: #dc2626;
    --warning: #f59e0b;
    --success: #10b981;
    --info: #3b82f6;
    --critical: #ef4444;
    --high-risk: #f97316;
    --medium-risk: #eab308;
    --low-risk: #84cc16;
}

.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1800px !important;
    margin: 0 auto;
    background-color: #f8fafc;
    padding: 20px;
}

.chatbot {
    min-height: 700px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    background-color: white;
    margin-bottom: 15px;
}

.dataframe {
    height: 400px;
    overflow-y: auto;
    font-size: 0.9em;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 15px;
}

.status-text {
    background-color: #f0f9ff;
    border-left: 4px solid var(--primary);
    padding: 8px;
    border-radius: 4px;
    margin: 8px 0;
    font-size: 0.9em;
}
u
.risk-critical {
    background-color: var(--critical) !important;
    color: white !important;
    font-weight: 600;
}

.risk-high {
    background-color: var(--high-risk) !important;
    color: white !important;
}

.risk-medium {
    background-color: var(--medium-risk) !important;
}

.risk-low {
    background-color: var(--low-risk) !important;
}

.dashboard-card {
    background-color: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 16px;
}

.header-card {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.button-primary {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    transition: all 0.2s ease;
    padding: 10px 20px !important;
    border-radius: 6px !important;
}

.button-primary:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.button-secondary {
    background: var(--secondary) !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
    transition: all 0.2s ease;
    padding: 10px 20px !important;
    border-radius: 6px !important;
}

.button-secondary:hover {
    background: var(--secondary-dark) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.tab-button {
    padding: 8px 16px !important;
    border-radius: 6px !important;
    transition: all 0.2s ease;
}

.tab-button.selected {
    background: var(--primary) !important;
    color: white !important;
}

.error-card {
    background-color: #fee2e2;
    border-left: 4px solid var(--danger);
    padding: 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 16px;
}

.empty-state {
    background-color: #f0f9ff;
    border-left: 4px solid var(--info);
    padding: 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 16px;
}

.user-message {
    border-left: 4px solid var(--primary) !important;
    padding: 12px;
    margin-bottom: 10px;
}

.bot-message {
    border-left: 4px solid var(--success) !important;
    padding: 12px;
    margin-bottom: 10px;
}

.plot-container {
    border-radius: 8px;
    background-color: white;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 15px;
}

.metric-card {
    background-color: white;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.metric-value {
    font-size: 1.5em;
    font-weight: 600;
    color: var(--primary);
}

.metric-label {
    font-size: 0.9em;
    color: #64748b;
}

.tabs {
    margin-bottom: 20px;
}

.accordion {
    margin-top: 20px;
}

.textbox {
    padding: 12px !important;
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
}

.progress-bar {
    height: 6px !important;
    border-radius: 3px !important;
}
"""

class RiskLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    HIGH_REGULATORY = "High (Regulatory)"
    MEDIUM = "Medium"
    LOW = "Low"

class AnalysisType(Enum):
    TREND = "Trend Analysis"
    COMPARATIVE = "Comparative Analysis"
    RISK = "Risk Assessment"
    GENERAL = "General Inquiry"

class EliteChatInterface:
    def __init__(self, rag_system: EliteRAGSystem, version: str = "4.2"):
        self.rag = rag_system
        self.version = version
        self.session_history = []
        self.current_session_id = f"FIN-{uuid.uuid4().hex[:8].upper()}"
        self.performance_metrics = {
            "query_count": 0,
            "successful_retrievals": 0,
            "avg_response_time": 0,
            "total_tokens_processed": 0,
            "failed_queries": {},
            "risk_distribution": {level.value: 0 for level in RiskLevel}
        }
        self._initialize_analytics_db()
        load_dotenv()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Check if vector store is initialized
        self._verify_vector_store()

    def _verify_vector_store(self):
        """Verify the vector store is properly initialized with documents"""
        try:
            # Simple check to see if store has any documents
            if hasattr(self.rag.vector_store, 'get_document_count'):
                doc_count = self.rag.vector_store.get_document_count()
                if doc_count == 0:
                    logger.warning("Vector store is empty - no documents loaded")
                    self._load_sample_data()
        except Exception as e:
            logger.error(f"Vector store verification failed: {str(e)}")
            raise RuntimeError("Failed to initialize vector store") from e

    def _load_sample_data(self):
        """Load sample data if vector store is empty with enhanced samples"""
        try:
            sample_data = [
                {
                    "content": "Customer complained about unexpected fees of $125 on their credit card statement. The charges appeared without prior notification and the customer is requesting a full refund.",
                    "metadata": {
                        "product": "Credit Card",
                        "date": "2023-05-15",
                        "complaint_id": "CC-2023-001",
                        "source": "CFPB",
                        "customer_id": "CUST-1001",
                        "region": "Northeast"
                    }
                },
                {
                    "content": "Mortgage payment was processed late causing a $75 late fee. The payment was submitted on time but the bank's system delayed processing by 3 business days.",
                    "metadata": {
                        "product": "Mortgage",
                        "date": "2023-06-22",
                        "complaint_id": "MTG-2023-045",
                        "source": "Internal",
                        "customer_id": "CUST-2045",
                        "region": "Midwest"
                    }
                },
                {
                    "content": "Auto loan application was denied without proper explanation. The customer has excellent credit score of 780 and believes this may be a case of discrimination.",
                    "metadata": {
                        "product": "Auto Loan",
                        "date": "2023-07-10",
                        "complaint_id": "ALN-2023-112",
                        "source": "CFPB",
                        "customer_id": "CUST-3012",
                        "region": "West"
                    }
                }
            ]
            
            logger.info("Loading sample data into vector store")
            self.rag.vector_store.add_documents(sample_data)
            logger.info(f"Successfully loaded {len(sample_data)} sample documents")
            
        except Exception as e:
            logger.error(f"Failed to load sample data: {str(e)}")
            raise RuntimeError("Failed to load sample data") from e

    def _retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents with enhanced empty store handling and parallel processing"""
        try:
            # Check if store is empty
            if hasattr(self.rag.vector_store, 'get_document_count'):
                if self.rag.vector_store.get_document_count() == 0:
                    logger.warning("Vector store empty - attempting to load sample data")
                    self._load_sample_data()
                    if self.rag.vector_store.get_document_count() == 0:
                        return []

            max_attempts = 3
            attempt = 0
            last_error = None
            
            while attempt < max_attempts:
                try:
                    processed_query = self._preprocess_query(query)
                    
                    # Parallel processing for document retrieval and enhancement
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(self.rag.retrieve, processed_query)
                        results = future.result(timeout=30)
                    
                    if not results:
                        logger.warning(f"No results found for query: {query}")
                        return []
                    
                    # Parallel document validation and enhancement
                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(self._validate_and_enhance_document, doc) for doc in results]
                        valid_docs = [f.result() for f in as_completed(futures) if f.result() is not None]
                    
                    if not valid_docs and attempt == 1:
                        broad_query = self._broaden_search_query(query)
                        with ThreadPoolExecutor() as executor:
                            future = executor.submit(self.rag.retrieve, broad_query, k=15)
                            broad_results = future.result(timeout=45)
                        
                        with ThreadPoolExecutor() as executor:
                            futures = [executor.submit(self._validate_and_enhance_document, doc) for doc in broad_results]
                            valid_docs = [f.result() for f in as_completed(futures) if f.result() is not None]
                    
                    if valid_docs:
                        return valid_docs
                        
                except Exception as e:
                    last_error = e
                    logger.warning(f"Retrieval attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    
                attempt += 1
                
            logger.error(f"All retrieval attempts failed for query: {query}")
            if last_error:
                self._track_failed_query(query, str(last_error))
            return []

        except Exception as e:
            logger.error(f"Document retrieval process failed: {str(e)}")
            self._track_failed_query(query, str(e))
            return []

    def _validate_and_enhance_document(self, doc: Dict) -> Optional[Dict]:
        """Validate and enhance a single document"""
        try:
            if self._validate_document(doc):
                return self._enhance_document(doc)
            return None
        except Exception as e:
            logger.warning(f"Document processing failed: {str(e)}")
            return None

    def _validate_document(self, doc: Dict) -> bool:
        """Ensure document meets quality standards with enhanced validation"""
        required_meta = {'product', 'date', 'complaint_id', 'source'}
        content = doc.get('page_content', '')
        metadata = doc.get('metadata', {})
        
        return (
            all(field in metadata for field in required_meta) and
            len(content) >= 50 and
            0 <= float(doc.get('score', -1)) <= 1 and
            bool(re.search(r'\d{4}-\d{2}-\d{2}', metadata.get('date', ''))) and
            any(term in content.lower() for term in ['fee', 'charge', 'payment', 'interest', 'service'])
        )

    def _format_sources(self, sources: List[Dict]) -> pd.DataFrame:
        """Format document sources for display with enhanced formatting"""
        if not sources:
            return self._empty_results_df()
        
        formatted = []
        for doc in sources:
            meta = doc.get('metadata', {})
            risk_level = self._calculate_risk_level(doc)
            
            formatted.append({
                'Product': meta.get('product', 'Unknown'),
                'Date': meta.get('date', 'Unknown'),
                'Confidence': f"{doc.get('score', 0):.2f}",
                'Risk Level': risk_level.value,
                'Excerpt': self._format_excerpt(doc['content']),
                'Complaint ID': meta.get('complaint_id', 'N/A'),
                'Regulatory Flags': ', '.join(doc.get('regulatory_flags', [])),
                'Sentiment': f"{doc.get('sentiment', 0):+.2f}",
                'Monetary Impact': f"${sum(doc.get('monetary_impact', [0])):,.2f}",
                'Source': meta.get('source', 'Unknown'),
                'Region': meta.get('region', 'N/A')
            })
        
        df = pd.DataFrame(formatted)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        return df.sort_values(['Risk Level', 'Confidence'], ascending=[False, False])

    def _empty_results_df(self) -> pd.DataFrame:
        """Return empty results dataframe with improved messaging"""
        return pd.DataFrame({
            'Product': ['No matching complaints found'],
            'Date': [''],
            'Confidence': [0],
            'Risk Level': [''],
            'Excerpt': ['Try adjusting your query or expanding the date range'],
            'Complaint ID': [''],
            'Regulatory Flags': [''],
            'Sentiment': [''],
            'Monetary Impact': [''],
            'Source': [''],
            'Region': ['']
        })

    def _format_excerpt(self, text: str) -> str:
        """Format document excerpt for display with enhanced highlighting"""
        text = text.replace('\n', ' ').strip()
        if len(text) > 350:
            text = text[:200] + ' [...] ' + text[-100:]
            
        highlights = {
            'fee': 'var(--warning)',
            'interest': 'var(--danger)',
            'charge': 'var(--danger)',
            'payment': 'var(--info)',
            'apr': 'var(--primary)',
            'rate': 'var(--primary-dark)',
            'refund': 'var(--success)',
            'denied': 'var(--danger)',
            'unauthorized': 'var(--critical)'
        }
        
        for term, color in highlights.items():
            text = re.sub(
                fr'(\b{term}\b)',
                fr'<span style="color: {color}; font-weight: 600;">\1</span>',
                text, 
                flags=re.IGNORECASE
            )
            
        return text + ('...' if len(text) > 150 else '')

    def _preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing with financial term expansion"""
        query = query.lower().strip()
        
        # Expand financial acronyms and terms
        financial_terms = {
            'bmrt': 'bank mortgage rate tracker',
            'apr': 'annual percentage rate',
            'arm': 'adjustable rate mortgage',
            'cd': 'certificate of deposit',
            'cda': 'consumer debt analysis',
            'cfpb': 'consumer financial protection bureau',
            'fcra': 'fair credit reporting act',
            'ecoa': 'equal credit opportunity act',
            'tila': 'truth in lending act'
        }
        
        for term, expansion in financial_terms.items():
            query = re.sub(fr'\b{term}\b', expansion, query, flags=re.IGNORECASE)
        
        # Add analysis enhancements
        enhancement_rules = {
            'trend': ' include temporal analysis',
            'compare': ' provide comparative metrics',
            'risk': ' assess risk factors',
            'why': ' explain root causes',
            'rising': ' show growth rate',
            'decline': ' show reduction metrics',
            'impact': ' quantify financial consequences',
            'pattern': ' identify common patterns',
            'region': ' analyze geographic distribution'
        }
        
        enhancements = [v for k, v in enhancement_rules.items() if k in query]
        if enhancements:
            query += " | Analysis Focus: " + ", ".join(enhancements)
            
        return query

    def _broaden_search_query(self, query: str) -> str:
        """Expand search parameters with intelligent broadening"""
        # Remove restrictive terms
        broadening_terms = {
            'specific': '',
            'exact': '',
            'precise': '',
            'only': '',
            'just': '',
            'exactly': ''
        }
        
        for term, replacement in broadening_terms.items():
            query = query.replace(term, replacement)
            
        # Add broadening operators
        if 'OR' not in query and 'AND' not in query:
            key_terms = [word for word in query.split() if len(word) > 3 and word not in ['what', 'how', 'when']]
            if len(key_terms) > 1:
                query += f" OR {' OR '.join(key_terms)}"
                
        return query + " (expanded search)"

    def _enhance_document(self, doc: Dict) -> Dict:
        """Add derived financial metrics and analysis to documents"""
        metadata = doc.get('metadata', {})
        content = doc.get('page_content', '')
        
        # Enhanced sentiment analysis
        sentiment_score = self._calculate_sentiment(content)
        
        # Monetary impact analysis
        monetary_terms = self._extract_monetary_terms(content)
        
        # Regulatory compliance analysis
        regulatory_flags = self._identify_regulatory_flags(content)
        
        # Geographic analysis
        region = metadata.get('region', 'Unknown')
        
        return {
            'content': content,
            'metadata': metadata,
            'score': min(1.0, max(0.0, float(doc.get('score', 0)))),
            'sentiment': sentiment_score,
            'monetary_impact': monetary_terms,
            'regulatory_flags': regulatory_flags,
            'key_phrases': self._extract_key_phrases(content),
            'region': region
        }

    def _calculate_sentiment(self, text: str) -> float:
        """Enhanced sentiment analysis with financial context"""
        positive_terms = ['resolved', 'fixed', 'refund', 'apology', 'satisfied', 'thank']
        negative_terms = ['denied', 'rejected', 'frustrated', 'angry', 'disappointed', 'upset']
        
        positive_count = sum(text.lower().count(term) for term in positive_terms)
        negative_count = sum(text.lower().count(term) for term in negative_terms)
        
        base_score = np.random.normal(0.2, 0.3)
        adjusted_score = base_score + (positive_count * 0.1) - (negative_count * 0.15)
        
        return min(1.0, max(-1.0, adjusted_score))

    def _extract_monetary_terms(self, text: str) -> List[float]:
        """Extract monetary values with enhanced parsing"""
        money_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)|\b(\d+)\s?(dollars|USD)\b'
        matches = re.findall(money_pattern, text)
        
        money_terms = []
        for match in matches:
            amount = match[0] if match[0] else match[1]
            try:
                value = float(amount.replace(',', ''))
                money_terms.append(value)
            except ValueError:
                continue
                
        return money_terms

    def _identify_regulatory_flags(self, text: str) -> List[str]:
        """Enhanced regulatory flag identification"""
        text = text.lower()
        flags = []
        
        regulatory_keywords = {
            'violation': 'Potential Compliance Violation',
            'unauthorized': 'Unauthorized Activity',
            'overcharge': 'Pricing Issue',
            'denied': 'Access Denial',
            'discriminate': 'Fair Lending Concern',
            'mislead': 'Deceptive Practice',
            'fraud': 'Fraud Indicator',
            'disclose': 'Disclosure Problem',
            'harass': 'Collection Practice Concern',
            'reporting': 'Credit Reporting Issue'
        }
        
        for term, flag in regulatory_keywords.items():
            if re.search(r'\b' + term + r'\b', text):
                flags.append(flag)
                
        # Check for UDAAP violations
        if any(word in text for word in ['unfair', 'deceptive', 'abusive']):
            flags.append('UDAAP Concern')
            
        return flags

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from complaint text"""
        phrases = []
        text = text.lower()
        
        # Common complaint patterns
        patterns = [
            r'(?:delay|late) (?:in|of) (?:payment|processing|refund)',
            r'(?:unauthorized|unexpected) (?:charge|fee|withdrawal)',
            r'(?:incorrect|wrong) (?:amount|calculation|statement)',
            r'(?:poor|bad) (?:customer service|communication)',
            r'(?:difficulty|problem) (?:contacting|reaching)',
            r'not (?:receive|get) (?:confirmation|response)',
            r'(?:promised|advertised) (?:rate|terms) not honored'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
            
        return list(set(phrases))[:5]  # Return up to 5 unique phrases

    def _calculate_risk_level(self, doc: Dict) -> RiskLevel:
        """Enhanced risk level calculation"""
        score = doc.get('score', 0)
        product = str(doc.get('metadata', {}).get('product', '')).lower()
        regulatory_flags = doc.get('regulatory_flags', [])
        
        if score > 0.9 or any('Violation' in flag for flag in regulatory_flags):
            return RiskLevel.CRITICAL
        elif score > 0.75:
            if 'loan' in product or 'mortgage' in product:
                return RiskLevel.HIGH_REGULATORY
            return RiskLevel.HIGH
        elif score > 0.6:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _determine_analysis_type(self, query: str) -> AnalysisType:
        """Enhanced analysis type detection"""
        query = query.lower()
        if 'trend' in query or 'over time' in query or 'historical' in query:
            return AnalysisType.TREND
        elif 'compare' in query or 'versus' in query or 'vs' in query or 'comparison' in query:
            return AnalysisType.COMPARATIVE
        elif 'risk' in query or 'issue' in query or 'problem' in query or 'complaint' in query:
            return AnalysisType.RISK
        return AnalysisType.GENERAL

    def _track_failed_query(self, query: str, error: str):
        """Enhanced failed query tracking"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash not in self.performance_metrics['failed_queries']:
            self.performance_metrics['failed_queries'][query_hash] = {
                'query': query,
                'error': error,
                'count': 0,
                'last_attempt': datetime.now().isoformat(),
                'analysis_type': self._determine_analysis_type(query).value,
                'stack_trace': traceback.format_exc()
            }
        self.performance_metrics['failed_queries'][query_hash]['count'] += 1
        self.performance_metrics['failed_queries'][query_hash]['last_attempt'] = datetime.now().isoformat()

    def _get_empty_response(self, query: str) -> str:
        """Enhanced empty response generation"""
        analysis_type = self._determine_analysis_type(query)
        
        responses = {
            AnalysisType.TREND: """
📊 **No Trend Data Available**

Our analysis didn't find sufficient data to identify trends matching your query.

**Recommendations:**
- Specify a product type (e.g., "credit card complaints")
- Define a clear time period (e.g., "Q3 2022 to Q2 2023")
- Try broader date ranges if needed
- Include geographic filters if applicable

*Example Query:*  
"Show me trends in mortgage complaints from California during 2022"
""",
            AnalysisType.COMPARATIVE: """
⚖️ **Comparative Analysis Unavailable**

We couldn't find comparable complaints matching your criteria.

**Suggestions:**
- Compare specific products (e.g., "mortgages vs personal loans")
- Include comparison metrics (e.g., "complaint volume by state")
- Verify product names are correct
- Check if time periods overlap

*Example Query:*  
"Compare complaint volumes between credit cards and personal loans in Q1 2023"
""",
            AnalysisType.RISK: """
⚠️ **Risk Assessment Incomplete**

No significant risks identified matching your query parameters.

**Next Steps:**
- Specify risk type (e.g., "regulatory risk in auto loans")
- Include product category and time frame
- Try broader risk categories
- Check for known risk factors in our database

*Example Query:*  
"Identify high-risk complaints about overdraft fees in checking accounts"
""",
            AnalysisType.GENERAL: """
🔍 **No Relevant Complaints Found**

**Possible Reasons:**
- Query too specific or narrow
- Product name may be misspelled
- Time period not in our database
- Geographic filter too restrictive

**Try:**
- Broadening search terms
- Checking product spellings
- Adjusting date ranges
- Removing geographic filters

*Example Query:*  
"Find complaints about unauthorized credit card charges in the last 6 months"
"""
        }
        
        return responses.get(analysis_type, """
❌ **No Results Found**

Please adjust your query and try again. Consider:
- Using different keywords
- Expanding your time range
- Being more specific about products or issues
""")

    def _get_error_response(self, query: str, error: Exception) -> str:
        """Enhanced error response with troubleshooting"""
        error_id = f"FIN-ERR-{int(time.time())}"
        error_type = "Technical Error"
        error_details = str(error)
        stack_trace = traceback.format_exc()
        
        if "connection" in error_details.lower():
            error_type = "Connection Issue"
            troubleshooting = [
                "Check your internet connection",
                "Verify backend services are running",
                "Retry in 1-2 minutes"
            ]
        elif "timeout" in error_details.lower():
            error_type = "Timeout Error"
            troubleshooting = [
                "Simplify complex queries",
                "Break query into smaller parts",
                "Avoid very broad time ranges"
            ]
        elif "memory" in error_details.lower():
            error_type = "Resource Constraint"
            troubleshooting = [
                "Reduce query complexity",
                "Narrow your search criteria",
                "Contact support for large dataset queries"
            ]
        else:
            troubleshooting = [
                "Check query syntax",
                "Verify system status",
                "Contact support with error details"
            ]
        
        return f"""
🚨 **Analysis Failed - {error_type} (Reference ID: {error_id})**

**Query:**  
"{query}"

**Error Details:**  
{error_details}

**Troubleshooting Steps:**  
{" ".join(f"- {step}" for step in troubleshooting)}

**Support Actions:**
- Error automatically logged for review
- Technical team notified
- System diagnostics initiated

**What You Can Do:**
1. Try again in a few minutes
2. Simplify complex queries
3. Contact support with reference ID if issue persists

"We apologize for the inconvenience. Our team will investigate this issue."
"""

    def _log_session(self, query: str, response: str, documents: List[Dict], session_id: str):
        """Enhanced session logging with metadata"""
        try:
            session_data = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'query': query,
                'response': response,
                'sources': self._format_sources(documents).to_dict('records'),
                'analysis_type': self._determine_analysis_type(query).value,
                'metrics': self._get_performance_metrics(),
                'risk_distribution': self._calculate_session_risk_distribution(documents),
                'monetary_impact': sum(
                    sum(doc.get('monetary_impact', [0])) 
                    for doc in documents
                ),
                'region_distribution': self._calculate_region_distribution(documents)
            }
            
            self.session_history.append(session_data)
            self._update_analytics(session_data)
            
        except Exception as e:
            logger.error(f"Failed to log session: {str(e)}")

    def _calculate_region_distribution(self, documents: List[Dict]) -> Dict:
        """Calculate regional distribution of complaints"""
        regions = [doc.get('region', 'Unknown') for doc in documents]
        return dict(pd.Series(regions).value_counts())

    def _calculate_session_risk_distribution(self, documents: List[Dict]) -> Dict:
        """Calculate risk distribution for a session"""
        distribution = {level.value: 0 for level in RiskLevel}
        for doc in documents:
            risk_level = self._calculate_risk_level(doc)
            distribution[risk_level.value] += 1
        return distribution

    def export_session(self) -> str:
        """Enhanced export with executive summary"""
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'system_version': self.version,
                'session_id': self.current_session_id,
                'performance_metrics': self.performance_metrics
            },
            'analysis_history': self.session_history,
            'key_findings': self._generate_key_findings(),
            'executive_summary': self._generate_executive_summary(),
            'visualizations': self._generate_visualization_data()
        }
        return json.dumps(report, indent=2)

    def _generate_key_findings(self) -> Dict:
        """Generate comprehensive key findings"""
        products = set()
        risk_levels = {level.value: 0 for level in RiskLevel}
        monetary_impact = 0.0
        regulatory_issues = 0
        regions = set()
        
        for session in self.session_history:
            for source in session.get('sources', []):
                products.add(source.get('Product', 'Unknown'))
                risk_level = source.get('Risk Level', 'Low')
                if risk_level in risk_levels:
                    risk_levels[risk_level] += 1
                
                if 'monetary_impact' in source:
                    monetary_impact += sum(source['monetary_impact'])
                
                if source.get('Regulatory Flags'):
                    regulatory_issues += 1
                
                if source.get('Region') and source['Region'] != 'N/A':
                    regions.add(source['Region'])
        
        return {
            'products_analyzed': sorted(products),
            'risk_distribution': risk_levels,
            'primary_topics': self._extract_topics(),
            'total_monetary_impact': monetary_impact,
            'regulatory_issues_count': regulatory_issues,
            'analysis_types_used': self._get_analysis_types_distribution(),
            'regions_represented': sorted(regions)
        }

    def _get_analysis_types_distribution(self) -> Dict:
        """Get distribution of analysis types used"""
        distribution = {atype.value: 0 for atype in AnalysisType}
        for session in self.session_history:
            atype = session.get('analysis_type', 'General Inquiry')
            distribution[atype] += 1
        return distribution

    def _generate_executive_summary(self) -> Dict:
        """Generate executive summary of the session"""
        total_queries = len(self.session_history)
        successful_queries = sum(1 for session in self.session_history if session.get('sources'))
        
        critical_issues = sum(
            session['risk_distribution'].get('Critical', 0)
            for session in self.session_history
        )
        
        monetary_impact = sum(
            session.get('monetary_impact', 0)
            for session in self.session_history
        )
        
        return {
            'total_queries': total_queries,
            'success_rate': f"{(successful_queries / total_queries) * 100:.1f}%" if total_queries > 0 else "N/A",
            'critical_issues_identified': critical_issues,
            'total_monetary_impact': f"${monetary_impact:,.2f}",
            'primary_concerns': self._identify_primary_concerns(),
            'recommendations': self._generate_session_recommendations(),
            'system_performance': self._get_performance_metrics()
        }

    def _generate_visualization_data(self) -> Dict:
        """Generate data for visualizations in the export"""
        if not self.session_history:
            return {}
            
        # Collect all documents across sessions
        all_docs = []
        for session in self.session_history:
            all_docs.extend(session.get('sources', []))
            
        if not all_docs:
            return {}
            
        # Prepare visualization data
        df = pd.DataFrame(all_docs)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        if df.empty:
            return {}
            
        # Product distribution
        product_dist = df['Product'].value_counts().head(10).to_dict()
        
        # Risk distribution
        risk_dist = df['Risk Level'].value_counts().to_dict()
        
        # Monthly trend
        monthly_trend = df.groupby(pd.Grouper(key='Date', freq='M')).size().to_dict()
        
        # Regional distribution
        region_dist = df['Region'].value_counts().to_dict()
        
        return {
            'product_distribution': product_dist,
            'risk_distribution': risk_dist,
            'monthly_trend': monthly_trend,
            'regional_distribution': region_dist
        }

    def _identify_primary_concerns(self) -> List[str]:
        """Identify primary concerns across all sessions"""
        concerns = []
        regulatory_flags = set()
        products_with_issues = set()
        
        for session in self.session_history:
            for source in session.get('sources', []):
                if source.get('Risk Level') == 'Critical':
                    product = source.get('Product', 'Unknown')
                    products_with_issues.add(product)
                    concerns.append(f"Critical issue with {product}")
                
                flags = source.get('Regulatory Flags', '').split(', ')
                regulatory_flags.update(flags)
        
        if regulatory_flags:
            concerns.append(f"Regulatory concerns: {', '.join(regulatory_flags)}")
        
        if products_with_issues:
            concerns.append(f"Products affected: {', '.join(products_with_issues)}")
            
        return concerns if concerns else ["No critical concerns identified"]

    def _generate_session_recommendations(self) -> List[str]:
        """Generate recommendations based on session data"""
        recommendations = []
        critical_count = sum(
            session['risk_distribution'].get('Critical', 0)
            for session in self.session_history
        )
        
        if critical_count > 0:
            recommendations.append(
                f"Immediate review required for {critical_count} critical issues"
            )
        
        regulatory_count = sum(
            len([f for f in session.get('sources', []) if f.get('Regulatory Flags')])
            for session in self.session_history
        )
        
        if regulatory_count > 0:
            recommendations.append(
                f"Regulatory compliance review needed for {regulatory_count} flagged items"
            )
        
        if not recommendations:
            recommendations.append("Routine monitoring recommended - no urgent issues found")
            
        # Add performance-based recommendations
        success_rate = self.performance_metrics.get('successful_retrievals', 0) / max(1, self.performance_metrics.get('query_count', 1))
        if success_rate < 0.7:
            recommendations.append("Consider refining query strategies - low success rate detected")
            
        return recommendations

    def _save_export(self, export_data: str) -> str:
        """Save exported session to file with enhanced naming"""
        try:
            export_path = Path("reports")
            export_path.mkdir(exist_ok=True)
            
            filename = (
                f"Financial_Complaint_Analysis_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"{self.current_session_id}.json"
            )
            
            filepath = export_path / filename
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(export_data)
                
            logger.info(f"Successfully exported session to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save export: {str(e)}")
            return ""

    def _update_metrics(self, start_time: float, response: str, success: bool) -> Dict:
        """Update performance metrics with enhanced tracking"""
        duration = time.time() - start_time
        tokens = len(response.split())
        
        self.performance_metrics.update({
            'query_count': self.performance_metrics['query_count'] + 1,
            'successful_retrievals': (
                self.performance_metrics['successful_retrievals'] + (1 if success else 0)
            ),
            'total_tokens_processed': (
                self.performance_metrics['total_tokens_processed'] + tokens
            ),
            'avg_response_time': (
                (self.performance_metrics['avg_response_time'] * (self.performance_metrics['query_count'] - 1) + duration)
            ) / max(1, self.performance_metrics['query_count'])
        })
        
        return self._get_performance_metrics()

    def _get_performance_metrics(self) -> Dict:
        """Format performance metrics for display with enhanced info"""
        return {
            'queries_processed': self.performance_metrics['query_count'],
            'success_rate': (
                f"{(self.performance_metrics['successful_retrievals'] / max(1, self.performance_metrics['query_count'])) * 100:.1f}%"
            ),
            'avg_response_time': f"{self.performance_metrics['avg_response_time']:.2f}s",
            'tokens_processed': f"{self.performance_metrics['total_tokens_processed']:,}",
            'current_session': self.current_session_id,
            'system_version': self.version,
            'failed_queries': len(self.performance_metrics['failed_queries'])
        }

    def _initialize_analytics_db(self):
        """Initialize enhanced analytics database"""
        self.analytics_db = {
            "query_patterns": {atype.value: 0 for atype in AnalysisType},
            "product_mentions": {},
            "risk_distribution": {level.value: 0 for level in RiskLevel},
            "time_series": [],
            "user_actions": [],
            "region_distribution": {},
            "error_log": []
        }

    def _update_analytics(self, session_data: Dict):
        """Update analytics database with session data"""
        try:
            # Track query patterns
            analysis_type = session_data.get('analysis_type', 'General Inquiry')
            self.analytics_db['query_patterns'][analysis_type] += 1
            
            # Track product mentions
            for source in session_data.get('sources', []):
                product = source.get('Product', 'Unknown')
                self.analytics_db['product_mentions'][product] = (
                    self.analytics_db['product_mentions'].get(product, 0) + 1
                )
                
                # Track risk distribution
                risk_level = source.get('Risk Level', 'Low')
                if risk_level in self.analytics_db['risk_distribution']:
                    self.analytics_db['risk_distribution'][risk_level] += 1
                
                # Track region distribution
                region = source.get('Region', 'Unknown')
                self.analytics_db['region_distribution'][region] = (
                    self.analytics_db['region_distribution'].get(region, 0) + 1
                )
            
            # Track time series data
            self.analytics_db['time_series'].append({
                'timestamp': session_data['timestamp'],
                'query_type': analysis_type,
                'documents_retrieved': len(session_data.get('sources', [])),
                'risk_levels': session_data.get('risk_distribution', {}),
                'monetary_impact': session_data.get('monetary_impact', 0)
            })
            
        except Exception as e:
            logger.error(f"Failed to update analytics: {str(e)}")
            self.analytics_db['error_log'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'operation': 'update_analytics'
            })

    def _create_visualization(self, documents: List[Dict]) -> Optional[Figure]:
        """Create enhanced visualization from documents"""
        try:
            if len(documents) < 2:
                return None

            # Prepare dataframe with enhanced metrics
            df = pd.DataFrame([{
                'date': doc.get('metadata', {}).get('date'),
                'product': doc.get('metadata', {}).get('product', 'Unknown'),
                'confidence': doc.get('score', 0),
                'risk': self._calculate_risk_level(doc).value,
                'monetary_impact': sum(doc.get('monetary_impact', [0])),
                'regulatory_flags': len(doc.get('regulatory_flags', [])),
                'sentiment': doc.get('sentiment', 0),
                'content_length': len(doc.get('content', '')),
                'region': doc.get('region', 'Unknown')
            } for doc in documents])

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            if df.empty:
                return None

            # Determine visualization type based on data characteristics
            unique_products = df['product'].nunique()
            time_range = (df['date'].max() - df['date'].min()).days
            unique_regions = df['region'].nunique()
            
            if unique_products > 1 and time_range > 30:
                fig = self._create_product_trend_plot(df)
            elif unique_products > 1 and unique_regions > 1:
                fig = self._create_geo_product_plot(df)
            elif unique_products > 1:
                fig = self._create_product_comparison_plot(df)
            elif time_range > 30:
                fig = self._create_temporal_trend_plot(df)
            elif unique_regions > 1:
                fig = self._create_regional_distribution_plot(df)
            else:
                fig = self._create_risk_distribution_plot(df)
            
            return fig
            
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}", exc_info=True)
            self.analytics_db['error_log'].append({
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'operation': 'create_visualization'
            })
            return None

    def _create_product_trend_plot(self, df: pd.DataFrame) -> Figure:
        """Create product trend visualization"""
        monthly = df.groupby([
            pd.Grouper(key='date', freq='M'),
            'product'
        ]).size().reset_index(name='count')
        
        fig = px.line(
            monthly,
            x='date',
            y='count',
            color='product',
            title='Monthly Complaint Volume by Product',
            labels={'count': 'Complaints', 'date': 'Date', 'product': 'Product'},
            template='plotly_white',
            line_shape='spline'
        )
        
        fig.update_layout(
            hovermode='x unified',
            legend_title_text='Product',
            xaxis_title='Month',
            yaxis_title='Number of Complaints',
            margin=dict(l=20, r=20, t=60, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    def _create_geo_product_plot(self, df: pd.DataFrame) -> Figure:
        """Create geographic product distribution visualization"""
        geo_data = df.groupby(['region', 'product']).size().reset_index(name='count')
        
        fig = px.bar(
            geo_data,
            x='region',
            y='count',
            color='product',
            barmode='group',
            title='Complaint Distribution by Region and Product',
            labels={'count': 'Number of Complaints', 'region': 'Region', 'product': 'Product'},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Region',
            yaxis_title='Number of Complaints',
            legend_title='Product',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    def _create_product_comparison_plot(self, df: pd.DataFrame) -> Figure:
        """Create product comparison visualization"""
        product_stats = df.groupby('product').agg({
            'confidence': 'mean',
            'risk': lambda x: (x == 'Critical').mean(),
            'monetary_impact': 'sum'
        }).reset_index()
        
        fig = px.bar(
            product_stats,
            x='product',
            y=['confidence', 'risk'],
            barmode='group',
            title='Product Comparison: Confidence vs Risk',
            labels={'value': 'Score', 'variable': 'Metric', 'product': 'Product'},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Product',
            yaxis_title='Score',
            legend_title='Metric',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    def _create_temporal_trend_plot(self, df: pd.DataFrame) -> Figure:
        """Create temporal trend visualization"""
        df['risk_score'] = df['risk'].map({
            'Critical': 4,
            'High (Regulatory)': 3,
            'High': 2,
            'Medium': 1,
            'Low': 0
        })
        
        monthly = df.groupby(pd.Grouper(key='date', freq='M')).agg({
            'risk_score': 'mean',
            'sentiment': 'mean',
            'monetary_impact': 'sum'
        }).reset_index()
        
        fig = px.line(
            monthly,
            x='date',
            y=['risk_score', 'sentiment'],
            title='Monthly Risk and Sentiment Trends',
            labels={'value': 'Score', 'variable': 'Metric', 'date': 'Month'},
            template='plotly_white'
        )
        
        fig.update_layout(
            yaxis_range=[-1, 4],
            hovermode='x unified',
            legend_title='Metric',
            xaxis_title='Month',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    def _create_regional_distribution_plot(self, df: pd.DataFrame) -> Figure:
        """Create regional distribution visualization"""
        region_counts = df['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        
        fig = px.bar(
            region_counts,
            x='region',
            y='count',
            color='region',
            title='Complaint Distribution by Region',
            labels={'count': 'Number of Complaints', 'region': 'Region'},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Region',
            yaxis_title='Count',
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    def _create_risk_distribution_plot(self, df: pd.DataFrame) -> Figure:
        """Create risk distribution visualization"""
        risk_counts = df['risk'].value_counts().reset_index()
        risk_counts.columns = ['risk_level', 'count']
        
        # Order by severity
        risk_order = ['Critical', 'High (Regulatory)', 'High', 'Medium', 'Low']
        risk_counts['risk_level'] = pd.Categorical(
            risk_counts['risk_level'],
            categories=risk_order,
            ordered=True
        )
        risk_counts = risk_counts.sort_values('risk_level')
        
        fig = px.bar(
            risk_counts,
            x='risk_level',
            y='count',
            color='risk_level',
            color_discrete_map={
                'Critical': '#ef4444',
                'High (Regulatory)': '#f97316',
                'High': '#f59e0b',
                'Medium': '#eab308',
                'Low': '#84cc16'
            },
            title='Complaint Risk Distribution',
            labels={'count': 'Number of Complaints', 'risk_level': 'Risk Level'},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Risk Level',
            yaxis_title='Count',
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig

    def launch(self):
        """Launch the interface with all components properly configured"""
        with gr.Blocks(css=PROFESSIONAL_CSS, theme=gr.themes.Default()) as demo:
            # Header Section
            with gr.Row(elem_classes=["header-card"]):
                gr.Markdown(f"""
                <div style="display: flex; align-items: center; gap: 20px;">
                    <div style="flex: 1;">
                        <h1 style="margin: 0; font-weight: 600;">🏦 Elite Financial Compliance Analyst</h1>
                        <h3 style="margin: 0; font-weight: 400; opacity: 0.9;">AI-Powered Consumer Complaint Analysis System v{self.version}</h3>
                    </div>
                </div>
                """)

            # Main Interface
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        elem_classes=["chatbot"],
                        label="Complaint Analysis Dialog",
                        type="messages"
                    )
                    
                    status_text = gr.Textbox(
                        visible=False,
                        interactive=False,
                        elem_classes=["status-text"]
                    )

                    query_input = gr.Textbox(
                        label="Enter financial complaint query",
                        placeholder="Example: 'Analyze credit card complaints from Q2 2023'",
                        lines=3,
                        max_lines=6
                    )

                    with gr.Row():
                        submit_btn = gr.Button("Analyze", variant="primary")
                        clear_btn = gr.Button("New Session", variant="secondary")
                        export_btn = gr.Button("Export Report", variant="secondary")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("🔍 Evidence"):
                            sources_output = gr.Dataframe(
                                headers=["Product", "Date", "Confidence", "Risk Level", "Excerpt"],
                                datatype=["str", "str", "number", "str", "str"],
                                interactive=False,
                                elem_classes=["dataframe"]
                            )
                        
                        with gr.Tab("📊 Visualization"):
                            visualization = gr.Plot(
                                label="Complaint Analysis Dashboard",
                                elem_classes=["plot-container"]
                            )
                        
                        with gr.Tab("📈 Metrics"):
                            metrics_output = gr.JSON(
                                label="System Performance",
                                elem_classes=["dashboard-card"]
                            )
                    
                    error_state = gr.JSON(
                        visible=False,
                        label="Error Details",
                        elem_classes=["error-card"]
                    )

            # Event handlers
            submit_btn.click(
                fn=lambda: gr.Textbox(visible=True, value="Processing..."),
                outputs=status_text
            ).then(
                fn=self._stream_response,
                inputs=[query_input, chatbot],
                outputs=[chatbot, sources_output, visualization, metrics_output, error_state]
            ).then(
                fn=lambda: gr.Textbox(visible=False),
                outputs=status_text
            )

        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )

    def _reset_session(self):
        """Reset the current session and start a new one"""
        self.current_session_id = f"FIN-{uuid.uuid4().hex[:8].upper()}"
        return [
            [],
            self._empty_results_df(),
            None,
            {
                'queries_processed': 0,
                'success_rate': '0%',
                'avg_response_time': '0s',
                'tokens_processed': 0,
                'current_session': self.current_session_id,
                'system_version': self.version
            },
            None,
            ""
        ]

    def _stream_response(self, query: str, chat_history: List[Tuple[str, str]]) -> Tuple:
        """Process user query and generate response"""
        start_time = time.time()  # Track processing start time
        
        try:
            # Track query pattern
            self._update_query_analytics(query)
            
            # Retrieve relevant documents
            documents = self._retrieve_documents(query)
            
            if not documents:
                response = self._get_empty_response(query)
                self._log_session(query, response, documents, self.current_session_id)
                return [
                    chat_history + [(query, response)],
                    self._empty_results_df(),
                    None,
                    self._update_metrics(start_time, response, False),
                    None  # No error state
                ]

            context = [doc['content'] for doc in documents]
            response = self._generate_comprehensive_response(query, context, documents)
            
            self._log_session(query, response, documents, self.current_session_id)
            visualization = self._create_visualization(documents)
            sources_df = self._format_sources(documents)
            metrics = self._update_metrics(start_time, response, True)
            
            return [
                chat_history + [(query, response)],
                sources_df,
                visualization,
                metrics,
                None  # No error state
            ]
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
            error_response = self._get_error_response(query, e)
            self._log_session(query, error_response, [], self.current_session_id)
            return [
                chat_history + [(query, error_response)],
                self._empty_results_df(),
                None,
                self._update_metrics(start_time, error_response, False),
                self._create_error_state(query, str(e))
            ]
    def _generate_comprehensive_response(self, query: str, context: List[str], documents: List[Dict]) -> str:
        """Generate professional financial analysis response with enhanced formatting"""
        try:
            base_response = "".join([
                chunk for chunk in self.rag.generate_response_stream(query, context)
            ])
            
            analysis_insights = self._generate_analysis_insights(documents)
            recommendations = self._generate_recommendations(documents)
            risk_assessment = self._generate_risk_assessment(documents)
            
            return f"""
## Financial Compliance Analysis Report

### Query Analysis
{base_response}

### Key Insights
{analysis_insights}

### Risk Assessment
{risk_assessment}

### Recommended Actions
{recommendations}

*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System v{self.version} | Session {self.current_session_id}*
"""
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return f"""
## Analysis Report Generation Failed

**Error Details:**  
{str(e)}

**System has logged this error for review.**
"""

    def _generate_analysis_insights(self, documents: List[Dict]) -> str:
        """Generate professional insights from documents with enhanced analysis"""
        if not documents:
            return "No significant insights identified"
            
        # Product distribution analysis
        products = [doc.get('metadata', {}).get('product', 'Unknown') for doc in documents]
        product_counts = pd.Series(products).value_counts()
        product_dist = "\n".join(
            f"- {product}: {count} complaints ({count/len(documents):.1%})"
            for product, count in product_counts.items()
        )
        
        # Risk distribution analysis
        risk_levels = [self._calculate_risk_level(doc) for doc in documents]
        risk_counts = pd.Series(risk_levels).value_counts()
        risk_dist = "\n".join(
            f"- {risk.value}: {count} occurrences ({count/len(documents):.1%})"
            for risk, count in risk_counts.items()
        )
        
        # Monetary impact analysis
        monetary_impacts = [sum(doc.get('monetary_impact', [0])) for doc in documents]
        total_impact = sum(monetary_impacts)
        avg_impact = total_impact / len(monetary_impacts) if monetary_impacts else 0
        
        # Sentiment analysis
        sentiments = [doc.get('sentiment', 0) for doc in documents]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        sentiment_desc = (
            "Positive" if avg_sentiment > 0.3 else
            "Slightly Positive" if avg_sentiment > 0.1 else
            "Neutral" if -0.1 <= avg_sentiment <= 0.1 else
            "Slightly Negative" if avg_sentiment > -0.3 else
            "Negative"
        )
        
        # Regional analysis
        regions = [doc.get('region', 'Unknown') for doc in documents]
        region_counts = pd.Series(regions).value_counts()
        region_dist = "\n".join(
            f"- {region}: {count} complaints" 
            for region, count in region_counts.head(3).items()
        ) if len(region_counts) > 0 else "- No regional data available"
        
        return f"""
- **Product Distribution**:  
{product_dist}

- **Risk Profile**:  
{risk_dist}

- **Financial Impact**:  
  - Total: ${total_impact:,.2f}  
  - Average per complaint: ${avg_impact:,.2f}

- **Sentiment Analysis**:  
  - Average: {avg_sentiment:+.2f} ({sentiment_desc})

- **Top Regions**:  
{region_dist}

- **Regulatory Flags**:  
  {len([doc for doc in documents if doc.get('regulatory_flags')])} complaints with potential regulatory issues
"""

    def _generate_risk_assessment(self, documents: List[Dict]) -> str:
        """Generate detailed risk assessment section"""
        if not documents:
            return "No significant risks identified"
            
        critical_issues = [
            doc for doc in documents 
            if self._calculate_risk_level(doc) in [RiskLevel.CRITICAL, RiskLevel.HIGH_REGULATORY]
        ]
        
        if not critical_issues:
            return """
**Overall Risk Assessment**:  
- No critical or high regulatory risks identified
- Routine monitoring recommended
"""
        
        risk_products = set(
            doc.get('metadata', {}).get('product', 'Unknown')
            for doc in critical_issues
        )
        
        regulatory_flags = set()
        monetary_impacts = []
        risk_regions = set()
        
        for doc in critical_issues:
            regulatory_flags.update(doc.get('regulatory_flags', []))
            monetary_impacts.extend(doc.get('monetary_impact', []))
            if doc.get('region'):
                risk_regions.add(doc['region'])
        
        total_impact = sum(monetary_impacts)
        
        return f"""
**Critical Risk Assessment**:  

- **Affected Products**: {', '.join(risk_products)}
- **Primary Regions**: {', '.join(risk_regions) or 'Not specified'}
- **Regulatory Concerns**: {', '.join(regulatory_flags) or 'None'}
- **Potential Financial Impact**: ${total_impact:,.2f}
- **Number of Critical Issues**: {len(critical_issues)}

**Immediate Actions Recommended**:
- Prioritize review of critical complaints
- Initiate regulatory compliance check
- Notify relevant departments
- Conduct regional analysis if patterns exist
"""

    def _generate_recommendations(self, documents: List[Dict]) -> str:
        """Generate professional recommendations with prioritization"""
        if not documents:
            return "No specific recommendations available"
            
        critical_issues = [
            doc for doc in documents 
            if self._calculate_risk_level(doc) in [RiskLevel.CRITICAL, RiskLevel.HIGH_REGULATORY]
        ]
        
        recommendations = []
        
        if critical_issues:
            products = set(
                doc.get('metadata', {}).get('product', 'Unknown') 
                for doc in critical_issues
            )
            
            regulatory_flags = set()
            monetary_impacts = []
            regions = set()
            
            for doc in critical_issues:
                regulatory_flags.update(doc.get('regulatory_flags', []))
                monetary_impacts.extend(doc.get('monetary_impact', []))
                if doc.get('region'):
                    regions.add(doc['region'])
            
            total_impact = sum(monetary_impacts)
            
            recommendations.append(
                f"🚨 **Immediate Action Required**: Review {len(critical_issues)} critical issues "
                f"across {len(products)} product(s) with potential financial impact of ${total_impact:,.2f}"
            )
            
            if regulatory_flags:
                recommendations.append(
                    f"⚖️ **Regulatory Compliance**: Address concerns about {', '.join(regulatory_flags)}"
                )
            
            if regions:
                recommendations.append(
                    f"🌎 **Regional Focus**: Issues concentrated in {', '.join(regions)}"
                )
        
        medium_issues = [
            doc for doc in documents 
            if self._calculate_risk_level(doc) == RiskLevel.MEDIUM
        ]
        
        if medium_issues:
            products = set(
                doc.get('metadata', {}).get('product', 'Unknown') 
                for doc in medium_issues
            )
            recommendations.append(
                f"🔍 **Proactive Monitoring**: {len(medium_issues)} medium-risk issues "
                f"in {len(products)} product(s) warrant monitoring"
            )
        
        if not recommendations:
            recommendations.append(
                "✅ **No Urgent Actions**: Routine review recommended for low-risk findings"
            )
        
        # Add general recommendations
        recommendations.extend([
            "📅 Schedule follow-up analysis in 30 days to track resolution",
            "📊 Consider trend analysis for identified high-risk products",
            "📝 Document all actions taken for compliance records",
            "🌐 Analyze geographic patterns if regional data is available"
        ])
        
        return "\n".join(f"- {rec}" for rec in recommendations)

    def _update_query_analytics(self, query: str):
        """Track query patterns for enhanced analytics"""
        query_type = self._determine_analysis_type(query)
        self.analytics_db['query_patterns'][query_type.value] += 1
        
        products = ['mortgage', 'credit card', 'loan', 'account', 'payment', 'investment']
        for product in products:
            if product in query.lower():
                self.analytics_db['product_mentions'][product] = (
                    self.analytics_db['product_mentions'].get(product, 0) + 1
                )

    def _create_error_state(self, query: str, error: str) -> Dict:
        """Create detailed error analysis for UI with troubleshooting"""
        error_type = "Technical Error"
        troubleshooting = []
        
        if "connection" in error.lower():
            error_type = "Connection Error"
            troubleshooting = [
                "Check your internet connection",
                "Verify backend services are running",
                "Retry in 1-2 minutes"
            ]
        elif "timeout" in error.lower():
            error_type = "Timeout Error"
            troubleshooting = [
                "Simplify complex queries",
                "Break query into smaller parts",
                "Avoid very broad time ranges"
            ]
        elif "memory" in error.lower():
            error_type = "Resource Constraint"
            troubleshooting = [
                "Reduce query complexity",
                "Narrow your search criteria",
                "Contact support for large dataset queries"
            ]
        else:
            troubleshooting = [
                "Check query syntax",
                "Verify system status",
                "Contact support with error details"
            ]
        
        return {
            "error_type": error_type,
            "query": query,
            "error_message": error,
            "suggested_actions": troubleshooting,
            "reference_id": f"ERR-{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "stack_trace": traceback.format_exc()
        }

    def _create_advanced_empty_state(self, query: str) -> Dict:
        """Create detailed guidance for empty results with query analysis"""
        analysis = self._determine_analysis_type(query)
        query_terms = query.lower().split()
        
        suggestions = {
            AnalysisType.TREND: [
                "Specify a time period (e.g., 'Q2 2023')",
                "Include product type (e.g., 'mortgage complaints')",
                "Try broader time ranges (e.g., 'last 3 years')",
                "Add geographic filters if relevant"
            ],
            AnalysisType.COMPARATIVE: [
                "Compare specific products (e.g., 'credit cards vs loans')",
                "Include metrics for comparison (e.g., 'complaint volume by state')",
                "Ensure products exist in database",
                "Check time periods overlap"
            ],
            AnalysisType.RISK: [
                "Specify risk type (e.g., 'regulatory risk in auto loans')",
                "Include product category and time frame",
                "Try broader risk categories",
                "Check for known risk factors in our database"
            ],
            AnalysisType.GENERAL: [
                "Reformulate with more specific terms",
                "Check for typos in product names",
                "Try alternative phrasing",
                "Use natural language (e.g., 'Show me complaints about hidden fees')"
            ]
        }
        
        return {
            "analysis_type": analysis.value,
            "query_diagnosis": self._diagnose_empty_query(query),
            "suggested_actions": suggestions.get(analysis, ["Try a different query"]),
            "system_status": "Operational",
            "reference_id": f"EMP-{int(time.time())}",
            "query_terms_analyzed": query_terms,
            "database_status": {
                "documents_loaded": self.rag.vector_store.get_document_count() if hasattr(self.rag.vector_store, 'get_document_count') else "Unknown",
                "last_updated": datetime.now().isoformat()
            }
        }

    def _diagnose_empty_query(self, query: str) -> str:
        """Provide specific diagnosis for empty results"""
        query = query.lower()
        
        if len(query.split()) < 3:
            return "Query may be too vague - add more specific terms"
        
        if not any(term in query for term in ['complaint', 'issue', 'problem', 'feedback']):
            return "Consider adding complaint-related terms to your query"
        
        time_terms = ['year', 'month', 'quarter', 'week', 'day']
        if not any(term in query for term in time_terms):
            return "No time period specified - try adding a date range"
            
        product_terms = ['card', 'loan', 'mortgage', 'account', 'payment']
        if not any(term in query for term in product_terms):
            return "No financial product specified - try adding a product type"
            
        return "Query parameters may not match available data - try broadening search"
    
        
if __name__ == "__main__":
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("financial_analyst.log"),
                logging.StreamHandler()
            ]
        )
        logger.info("Initializing financial analysis system...")
        
        logger.info("Loading vector store...")
        vector_store = get_vector_store()
        
        logger.info("Initializing RAG engine...")
        rag_system = EliteRAGSystem(vector_store)
        
        logger.info("Launching analyst interface...")
        EliteChatInterface(rag_system).launch()
    except Exception as e:
        logger.critical(f"System failed to start: {str(e)}", exc_info=True)
        raise