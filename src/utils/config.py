from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "CrediTrust AI Complaint Analysis"
    openai_api_key: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"
    chroma_db_path: str = "data/chroma_db"
    embedding_model: str = "all-mpnet-base-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()