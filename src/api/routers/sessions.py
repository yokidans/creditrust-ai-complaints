from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
from uuid import UUID
import json

# Advanced imports
from src.services.session import SessionManager
from src.models.schemas import (
    UserSession,
    SessionAnalytics,
    ConversationSummary
)
from src.api.dependencies import get_current_user
from src.utils.cache import RedisCache
from src.services.llm import LLMService

router = APIRouter(
    prefix="/sessions",
    tags=["Intelligent Sessions"],
    dependencies=[Depends(get_current_user)]
)

class SessionCreateRequest(BaseModel):
    context_memory: str = Field(
        default="hybrid",
        description="Memory strategy (hybrid, semantic, episodic)"
    )
    retention_period: timedelta = Field(
        default=timedelta(days=30),
        description="Auto-purge period for inactive sessions"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Custom session metadata"
    )

@router.post("/v2", response_model=UserSession)
async def create_enhanced_session(
    request: SessionCreateRequest,
    user: dict = Depends(get_current_user),
    session_manager: SessionManager = Depends(SessionManager)
):
    """
    Creates an intelligent session with:
    - Adaptive memory management
    - Context-aware retention
    - Multi-modal metadata support
    """
    session = await session_manager.create_enhanced_session(
        user_id=user["id"],
        memory_strategy=request.context_memory,
        retention=request.retention_period,
        metadata=request.metadata
    )
    
    # Initialize with regulatory context
    await session_manager.initialize_regulatory_context(
        session.id,
        user["jurisdiction"]
    )
    
    return session

@router.get("/v2/{session_id}/analytics", response_model=SessionAnalytics)
async def get_session_analytics(
    session_id: UUID,
    session_manager: SessionManager = Depends(SessionManager),
    llm: LLMService = Depends(LLMService)
):
    """
    Advanced session analytics including:
    - Conversation sentiment trajectory
    - Regulatory topic clustering
    - Resolution effectiveness scoring
    """
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Generate real-time analytics
    analytics = await session_manager.generate_analytics(session_id)
    
    # LLM-powered insights
    insights_prompt = f"""
    Analyze this compliance session and provide executive insights:
    
    {json.dumps(analytics.dict(), indent=2)}
    """
    analytics.insights = await llm.generate(insights_prompt)
    
    return analytics

@router.post("/v2/{session_id}/summarize", response_model=ConversationSummary)
async def summarize_session(
    session_id: UUID,
    detail_level: str = Query("executive"),
    session_manager: SessionManager = Depends(SessionManager),
    llm: LLMService = Depends(LLMService)
):
    """
    AI-powered session summarization with:
    - Dynamic detail adjustment (executive, technical, legal)
    - Action item extraction
    - Compliance violation tagging
    """
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Generate summary based on requested detail level
    summary = await session_manager.generate_summary(
        session_id,
        detail_level=detail_level
    )
    
    # Post-process with LLM for coherence
    refined = await llm.refine_summary(summary, style=detail_level)
    
    return ConversationSummary(
        **summary.dict(),
        refined_summary=refined
    )

@router.get("/v2/{session_id}/context")
async def get_session_context(
    session_id: UUID,
    depth: int = Query(3),
    session_manager: SessionManager = Depends(SessionManager)
):
    """
    Retrieves expanded session context with:
    - Multi-hop memory recall
    - Related regulatory references
    - Temporal context alignment
    """
    return await session_manager.get_context_graph(
        session_id,
        depth=depth
    )