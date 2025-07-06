# src/config.py
import os
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# --------------------------
# Configuration Models
# --------------------------

@dataclass
class APIConfig:
    title: str = "CreditRust AI API"
    description: str = "Financial Compliance Analysis API"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class AuthConfig:
    scheme: str = "Bearer"
    master_key: str = field(default_factory=lambda: os.getenv("API_MASTER_KEY"))
    token_expiry_minutes: int = 1440  # 24 hours

@dataclass
class DatabaseConfig:
    url: str = field(default_factory=lambda: os.getenv("DB_URL"))
    max_connections: int = 20
    min_connections: int = 5

@dataclass
class CacheConfig:
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL"))
    default_ttl: int = 3600  # 1 hour
    max_size: int = 1000

@dataclass
class LLMServiceConfig:
    model: str = "openai/gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    http_referer: str = "https://api.creditrust.ai"
    x_title: str = "CreditRust AI Production"
    max_context_length: int = 8000  # Tokens
    system_prompt: str = """
    You are CreditRust AI, a financial compliance expert specializing in consumer complaints.
    Your responses should be:
    - Professional yet empathetic
    - Factually accurate with citations when possible
    - Structured with clear sections
    """

# --------------------------
# Domain Models
# --------------------------

class UserSession(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    conversation: List[Dict] = []
    metadata: Dict = {}

class AnalysisRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    metadata: Optional[Dict] = None

# --------------------------
# Service Configurations
# --------------------------

class ServiceConfig:
    def __init__(self):
        self.api = APIConfig()
        self.auth = AuthConfig()
        self.db = DatabaseConfig()
        self.cache = CacheConfig()
        self.llm = LLMServiceConfig()

    @property
    def fastapi_kwargs(self) -> Dict:
        return {
            "title": self.api.title,
            "description": self.api.description,
            "version": self.api.version,
            "docs_url": self.api.docs_url,
            "redoc_url": self.api.redoc_url,
            "openapi_url": self.api.openapi_url
        }

    @property
    def cors_config(self) -> Dict:
        return {
            "allow_origins": self.api.cors_origins,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        }

# --------------------------
# Initialized Configurations
# --------------------------

config = ServiceConfig()

# Helper function for environment variable validation
def get_env_var(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} is required")
    return value