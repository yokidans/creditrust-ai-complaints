# src/utils/cache.py
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

class RedisCache:  # Changed back to RedisCache to match imports
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.enabled = False
        if os.getenv("DISABLE_REDIS", "").lower() == "true":
            self.enabled = False
            logger.info("Redis disabled by configuration")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def initialize(self):
        """Initialize Redis connection with retry"""
        if not self.enabled:
            return
            
        try:
            self.client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                decode_responses=True
            )
            await self.client.ping()
            self.enabled = True
            logger.info("Redis connected successfully")
        except Exception as e:
            self.enabled = False
            logger.warning(f"Redis connection failed: {str(e)}")

    async def get(self, key: str) -> Optional[str]:
        if not self.enabled or not self.client:
            return None
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"Cache get failed: {str(e)}")
            return None

    async def set(self, key: str, value: str, expire: int = 3600) -> None:
        if not self.enabled or not self.client:
            return
        try:
            await self.client.setex(key, expire, value)
        except Exception as e:
            logger.error(f"Cache set failed: {str(e)}")