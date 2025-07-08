# src/utils/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = {
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "device": "cpu",
        "normalize": True
    },
    "retrieval": {
        "top_k": 5,
        "score_threshold": 0.7,
        "rerank": False,
        "rerank_model": ""
    },
    "vector_store": {
        "path": "data/embeddings/default_faiss_index.index"
    }
}

def find_config_file(config_file: str) -> Optional[Path]:
    """Search for config file in multiple locations"""
    search_paths = [
        Path("src/config") / config_file,  # Development
        Path("config") / config_file,      # Production
        Path("/etc/creditrust") / config_file,  # System-wide
        Path.home() / ".creditrust" / config_file  # User-specific
    ]
    for path in search_paths:
        if path.exists():
            return path
    return None

def load_config(config_file: str = "model_config.yaml") -> Dict[str, Any]:
    """
    Load configuration with fallback to defaults
    
    Args:
        config_file: Name of the config file to load
        
    Returns:
        Merged configuration (file values override defaults)
    """
    config_path = find_config_file(config_file)
    if not config_path:
        logger.warning(f"Config file {config_file} not found, using defaults")
        return DEFAULT_CONFIG
        
    try:
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
            
        # Deep merge with defaults
        config = {**DEFAULT_CONFIG, **user_config}
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {str(e)}")
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return DEFAULT_CONFIG