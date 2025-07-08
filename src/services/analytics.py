# src/services/analytics.py
import logging
from typing import List, Dict
import pandas as pd
import plotly.express as px
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ComplaintAnalytics:
    """Handles analytics and visualization of complaint data"""
    
    def __init__(self):
        logger.info("Initializing ComplaintAnalytics service")

    def generate_trend_plot(self, complaints: List[Dict]) -> Dict:
        """Generate trend visualization data"""
        try:
            df = pd.DataFrame(complaints)
            if 'date_received' not in df.columns:
                return {}
                
            df['date'] = pd.to_datetime(df['date_received'])
            trend_data = df.resample('M', on='date').size().reset_index(name='count')
            
            fig = px.line(
                trend_data,
                x='date',
                y='count',
                title='Complaint Trends Over Time'
            )
            return fig.to_dict()
            
        except Exception as e:
            logger.error(f"Error generating trend plot: {str(e)}")
            return {}

    def generate_sentiment_sunburst(self, complaints: List[Dict]) -> Dict:
        """Generate sentiment sunburst visualization"""
        try:
            df = pd.DataFrame(complaints)
            if not all(col in df.columns for col in ['product', 'issue', 'sentiment']):
                return {}
                
            fig = px.sunburst(
                df,
                path=['product', 'issue'],
                values='sentiment',
                title='Complaint Sentiment Distribution'
            )
            return fig.to_dict()
            
        except Exception as e:
            logger.error(f"Error generating sentiment sunburst: {str(e)}")
            return {}