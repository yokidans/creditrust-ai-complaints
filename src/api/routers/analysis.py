from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import json
from uuid import uuid4
import httpx

# Enhanced imports
from src.core.retrieval import HierarchicalRetriever
from src.services.llm import LLMService
from src.services.cache import RedisCache
from src.utils.analytics import ComplaintAnalytics
from src.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    RegulatoryAssessment,
    RiskProfile
)
from src.api.dependencies import get_current_user
from src.utils.logging import logger

router = APIRouter(
    prefix="/analysis",
    tags=["Advanced Analysis"],
    dependencies=[Depends(get_current_user)]
)

class AdvancedAnalysisRequest(AnalysisRequest):
    analysis_framework: str = Field(
        default="CFPB",
        description="Regulatory framework to apply (CFPB, FDCPA, GDPR, etc.)"
    )
    risk_assessment: bool = Field(
        default=True,
        description="Include detailed risk scoring"
    )
    comparative_analysis: bool = Field(
        default=False,
        description="Compare against historical complaints"
    )

@router.post("/v2/analyze", response_model=AnalysisResponse)
async def advanced_analyze(
    request: AdvancedAnalysisRequest,
    llm: LLMService = Depends(LLMService),
    retriever: HierarchicalRetriever = Depends(HierarchicalRetriever),
    cache: RedisCache = Depends(RedisCache)
):
    """
    Elite-level complaint analysis with:
    - Dynamic framework adaptation
    - Real-time regulatory updates
    - Multi-dimensional risk scoring
    """
    # Generate cache key with all parameters
    cache_key = f"analysis:v2:{hash(frozenset(request.dict().items()))}"
    
    if cached := await cache.get(cache_key):
        return JSONResponse(content=json.loads(cached))

    # Hybrid retrieval (semantic + temporal + product-based)
    results = await retriever.hybrid_retrieve(
        query_text=request.text,
        products=request.products,
        date_range=request.date_range,
        framework=request.analysis_framework
    )

    # Generate regulatory impact assessment
    regulatory_assessment = await llm.generate_regulatory_assessment(
        request.text, 
        framework=request.analysis_framework
    )

    # Dynamic prompt engineering based on request parameters
    prompt = await llm.build_analysis_prompt(
        request.text,
        results,
        include_risk=request.risk_assessment,
        comparative=request.comparative_analysis
    )

    analysis = await llm.generate(prompt)

    response = AnalysisResponse(
        analysis=analysis,
        insights=results,
        regulatory_assessment=regulatory_assessment,
        metadata={
            "framework": request.analysis_framework,
            "retrieval_strategy": retriever.last_strategy
        }
    )

    # Cache with framework-aware TTL
    await cache.set(cache_key, response.json(), expire=3600)
    
    return response

@router.post("/v2/stream")
async def streaming_analysis(
    request: AdvancedAnalysisRequest,
    llm: LLMService = Depends(LLMService)
):
    """
    Real-time streaming analysis with:
    - Progressive disclosure of insights
    - Live regulatory flagging
    - Dynamic confidence scoring
    """
    async def generate_insights():
        # Phase 1: Immediate issue identification
        yield json.dumps({"phase": "identification"})
        id_prompt = llm.build_identification_prompt(request.text)
        async for chunk in llm.generate_stream(id_prompt):
            yield chunk

        # Phase 2: Regulatory context
        yield json.dumps({"phase": "regulatory"})
        reg_prompt = llm.build_regulatory_prompt(request.text, request.analysis_framework)
        async for chunk in llm.generate_stream(reg_prompt):
            yield chunk

        # Phase 3: Resolution recommendations
        yield json.dumps({"phase": "resolution"})
        res_prompt = llm.build_resolution_prompt(request.text)
        async for chunk in llm.generate_stream(res_prompt):
            yield chunk

    return StreamingResponse(
        generate_insights(),
        media_type="application/x-ndjson"
    )

@router.get("/v2/regulatory/updates")
async def get_regulatory_updates(
    framework: str = Query("CFPB"),
    since: datetime = Query(None)
):
    """
    Real-time regulatory update monitoring integrated with:
    - CFPB API
    - FINRA feeds
    - SEC Edgar system
    """
    async with httpx.AsyncClient() as client:
        # In a real implementation, this would call regulatory APIs
        updates = await client.get(
            f"https://api.regulatory.updates/v1/{framework}",
            params={"since": since.isoformat() if since else None}
        )
        return updates.json()