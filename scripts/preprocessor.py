# creditrust-ai-complaints/src/services/preprocessor.py
import re
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    """Handles cleaning and preprocessing of complaint narratives."""
    
    def __init__(self):
        # Common boilerplate phrases to remove
        self.boilerplate_phrases = [
            r"i am writing to file a complaint",
            r"this is a complaint regarding",
            r"please be advised that",
            r"i am requesting your assistance with",
            r"i am reporting an issue with"
        ]
        
    def clean_text(self, text: str) -> str:
        """Cleans and normalizes complaint narrative text."""
        if not text or not isinstance(text, str):
            return ""
            
        try:
            # Lowercase
            text = text.lower()
            
            # Remove boilerplate
            for phrase in self.boilerplate_phrases:
                text = re.sub(phrase, "", text, flags=re.IGNORECASE)
            
            # Remove special characters except basic punctuation
            text = re.sub(r"[^a-zA-Z0-9\s.,;:!?']", " ", text)
            
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return ""