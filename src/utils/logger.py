# src/utils/logger.py
import logging
from rich.logging import RichHandler
from typing import Optional

def get_logger(name: str, level: Optional[int] = logging.INFO) -> logging.Logger:
    """
    Creates and configures a logger with Rich formatting.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: logging.INFO)
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(show_path=False)]
    )
    logger = logging.getLogger(name)
    return logger