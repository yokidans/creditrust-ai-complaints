import os
import re
import json
import asyncio
import logging
from typing import Optional, List, Dict, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from getpass import getpass
from pathlib import Path
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from fastapi import FastAPI, HTTPException, Depends, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uuid
from datetime import datetime

# --------------------------
# Configuration Setup
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_path=False)]
)
logger = logging.getLogger(__name__)
console = Console()

#app = FastAPI(title="CreditRust AI API")

app = FastAPI()
security = HTTPBearer()

# Add this endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Creditrust AI Complaints API"}

# --------------------------
# Domain Models
# --------------------------
class UserSession(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    conversation: List[Dict] = []
    metadata: Dict = {}

@dataclass
class GenerationConfig:
    model: str = "openai/gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    system_prompt: str = """
    You are CreditRust AI, a financial compliance expert specializing in consumer complaints.
    Your responses should be:
    - Professional yet empathetic
    - Factually accurate with citations when possible
    - Structured with clear sections
    """
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str = "https://api.creditrust.ai"
    x_title: str = "CreditRust AI Production"
    max_context_length: int = 8000  # Tokens

    def __post_init__(self):
        if not 0 <= self.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive")

# --------------------------
# Security & Authentication
# --------------------------
class AuthHandler:
    def __init__(self):
        self.valid_tokens = {
            os.getenv("API_MASTER_KEY"): "admin",
            # Add other tokens here or load from DB
        }

    async def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
        if credentials.credentials not in self.valid_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        return self.valid_tokens[credentials.credentials]

auth_handler = AuthHandler()

# --------------------------
# Conversation Management
# --------------------------
class ConversationManager:
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}

    def create_session(self, user_id: str, metadata: Dict = None) -> UserSession:
        session_id = str(uuid.uuid4())
        session = UserSession(
            id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        return self.sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str):
        if session_id in self.sessions:
            self.sessions[session_id].conversation.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })

    def prune_conversation(self, session_id: str, max_tokens: int):
        """Trim conversation to fit context window"""
        # Implementation would tokenize and trim oldest messages
        pass

conv_manager = ConversationManager()

# --------------------------
# Advanced Prompt Engineering
# --------------------------
class PromptEngineer:
    @staticmethod
    def financial_compliance_prompt(text: str, context: List[Dict] = None) -> Tuple[str, List[Dict]]:
        base_prompt = f"""
        Analyze this financial complaint with the following structure:

        1. **Issue Identification**:
        - Primary concern
        - Secondary concerns
        - Regulatory implications

        2. **Compliance Analysis**:
        - Relevant regulations (CFPB, FDCPA, etc.)
        - Potential violations
        - Risk level (Low/Medium/High)

        3. **Resolution Recommendation**:
        - Immediate actions
        - Long-term solutions
        - Customer communication points

        Complaint Details:
        {text}
        """

        messages = [
            {"role": "system", "content": GenerationConfig().system_prompt},
            *([] if not context else context),
            {"role": "user", "content": base_prompt}
        ]

        return base_prompt, messages

    @staticmethod
    def summarize_conversation(conversation: List[Dict]) -> str:
        return f"""
        Summarize this conversation for our records:

        Key Points:
        - Identify main complaint themes
        - Extract regulatory references
        - Note suggested resolutions

        Conversation:
        {json.dumps(conversation, indent=2)}
        """

# --------------------------
# Core LLM Service (Enhanced)
# --------------------------
class LLMService:
    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=self.config.base_url,
            default_headers={
                "HTTP-Referer": self.config.http_referer,
                "X-Title": self.config.x_title,
            }
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        reraise=True
    )
    async def generate(self, prompt: str, context: List[Dict] = None) -> str:
        messages = self._build_messages(prompt, context)
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise self._transform_error(e)

    async def generate_stream(self, prompt: str, context: List[Dict] = None) -> AsyncGenerator[str, None]:
        messages = self._build_messages(prompt, context)
        try:
            stream = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Stream generation failed: {str(e)}")
            raise self._transform_error(e)

    def _build_messages(self, prompt: str, context: List[Dict] = None) -> List[Dict]:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _transform_error(self, error: Exception) -> Exception:
        error_msg = str(error)
        if "401" in error_msg:
            return HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API credentials")
        elif "429" in error_msg:
            return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded")
        elif "404" in error_msg:
            return HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
        return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "LLM service error")

# --------------------------
# FastAPI Endpoints
# --------------------------
class AnalysisRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    metadata: Optional[Dict] = None

@app.post("/api/v1/analyze", tags=["Analysis"])
async def analyze_complaint(
    request: AnalysisRequest,
    user: str = Depends(auth_handler.verify_token)
):
    """Endpoint for complaint analysis"""
    llm = LLMService()
    
    # Get or create session
    session = (conv_manager.get_session(request.session_id) if request.session_id
              else conv_manager.create_session(user, request.metadata))
    
    # Generate professional analysis
    prompt, messages = PromptEngineer.financial_compliance_prompt(request.text, session.conversation)
    response = await llm.generate(prompt, messages)
    
    # Update conversation
    conv_manager.add_message(session.id, "user", request.text)
    conv_manager.add_message(session.id, "assistant", response)
    
    return {
        "analysis": response,
        "session_id": session.id,
        "tokens_used": len(response.split())  # Approximation
    }

@app.post("/api/v1/chat", tags=["Conversation"])
async def chat_stream(
    request: AnalysisRequest,
    user: str = Depends(auth_handler.verify_token)
):
    """Streaming chat endpoint"""
    llm = LLMService()
    session = (conv_manager.get_session(request.session_id) if request.session_id
              else conv_manager.create_session(user, request.metadata))
    
    prompt, messages = PromptEngineer.financial_compliance_prompt(request.text, session.conversation)
    
    async def stream_response():
        full_response = []
        async for chunk in llm.generate_stream(prompt, messages):
            yield chunk
            full_response.append(chunk)
        
        # Save conversation after streaming completes
        full_text = "".join(full_response)
        conv_manager.add_message(session.id, "user", request.text)
        conv_manager.add_message(session.id, "assistant", full_text)
    
    return StreamingResponse(stream_response(), media_type="text/event-stream")

@app.get("/api/v1/sessions/{session_id}", tags=["Sessions"])
async def get_session(
    session_id: str,
    user: str = Depends(auth_handler.verify_token)
):
    """Retrieve conversation history"""
    session = conv_manager.get_session(session_id)
    if not session or session.user_id != user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return session

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)