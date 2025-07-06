# creditrust-ai-complaints/src/core/rag_engine.py
from typing import Dict, List, Optional
import logging
from src.utils.logger import get_logger
from src.services.vector_store import VectorStoreManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

logger = get_logger(__name__)

class RAGEngine:
    """Orchestrates the RAG pipeline: retrieval + generation."""
    
    def __init__(self, vector_store: VectorStoreManager, llm_config: Dict):
        self.vector_store = vector_store
        self.llm_config = llm_config
        self.llm = self._initialize_llm()
        self.prompt_template = self._create_prompt_template()
        
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            return ChatOpenAI(
                model_name=self.llm_config['model_name'],
                temperature=self.llm_config['temperature'],
                max_tokens=self.llm_config['max_tokens']
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
            
    def _create_prompt_template(self):
        """Create the prompt template for answer generation."""
        return PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are a financial complaints analyst for CrediTrust. 
            Answer the user's question based on the following complaint excerpts.
            Provide a concise summary of key issues and patterns.
            
            Question: {question}
            
            Relevant Complaints:
            {context}
            
            Answer:
            """
        )
        
    def generate_response(self, question: str, product_filter: Optional[str] = None, analyze_sentiment: bool = False) -> Dict:
        """
        Generate an answer to a question using retrieved complaints.
        
        Args:
            question: The user's question
            product_filter: Optional product category filter
            analyze_sentiment: Whether to perform sentiment analysis
            
        Returns:
            Dictionary containing answer, sources, and optional sentiment
        """
        try:
            # Retrieve relevant complaints
            filter_by = {'product': product_filter} if product_filter else None
            retrieved = self.vector_store.similarity_search(question, k=5, filter_by=filter_by)
            
            if not retrieved:
                return {
                    'answer': "No relevant complaints found.",
                    'sources': []
                }
            
            # Format context for LLM
            context = "\n\n".join([
                f"Complaint ID: {item['source_id']}\n"
                f"Product: {item['product']}\n"
                f"Issue: {item['issue']}\n"
                f"Text: {item['text']}\n"
                f"Date: {item['date_received']}"
                for item in retrieved
            ])
            
            # Generate answer
            chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            answer = chain.run(question=question, context=context)
            
            # Prepare sources
            sources = [{
                'complaint_id': item['source_id'],
                'product': item['product'],
                'issue': item['issue'],
                'text_excerpt': item['text'],
                'similarity_score': item['score']
            } for item in retrieved]
            
            # Optional sentiment analysis
            sentiment = None
            if analyze_sentiment:
                sentiment = self._analyze_sentiment(context)
            
            return {
                'answer': answer,
                'sources': sources,
                'sentiment': sentiment
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    def _analyze_sentiment(self, text: str) -> Dict:
        """Perform basic sentiment analysis on complaint text."""
        # This could be enhanced with a dedicated sentiment model
        positive_words = ['good', 'happy', 'satisfied', 'pleased', 'helpful']
        negative_words = ['bad', 'angry', 'frustrated', 'disappointed', 'terrible']
        
        positive_count = sum(text.lower().count(word) for word in positive_words)
        negative_count = sum(text.lower().count(word) for word in negative_words)
        
        if negative_count > positive_count:
            return {'overall': 'negative', 'score': -1}
        elif positive_count > negative_count:
            return {'overall': 'positive', 'score': 1}
        else:
            return {'overall': 'neutral', 'score': 0}