# src/api/main.py
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import json
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from jinja2 import TemplateNotFound, Template

# Load environment variables first
load_dotenv()

# ==============================================
# FastAPI App Configuration
# ==============================================
app = FastAPI(
    title="CrediTrust AI Complaint Analysis API",
    description="API for analyzing financial customer complaints",
    version="0.1.0",
    # Documentation endpoints configuration
    docs_url="/docs",              # Primary Swagger UI
    redoc_url="/redoc",            # Primary ReDoc
    openapi_url="/openapi.json",   # OpenAPI schema
    # API-prefixed docs configuration
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
    swagger_ui_init_oauth={
        "clientId": "your-client-id",
        "appName": "CrediTrust AI"
    }
)

# ==============================================
# Path Configuration
# ==============================================
BASE_DIR = Path(r"C:\Users\tefer\creditrust-ai-complaints")
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Verify critical paths exist
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

print(f"Template verification:")
print(f"• index.html: {(TEMPLATES_DIR/'pages'/'index.html').exists()}")
print(f"• navbar.html: {(TEMPLATES_DIR/'components'/'navbar.html').exists()}")
print(f"• styles.css: {(STATIC_DIR/'css'/'styles.css').exists()}")

# ==============================================
# Template Configuration with Fallbacks
# ==============================================
class SafeTemplates(Jinja2Templates):
    """
    Enhanced template system with fallbacks for missing templates
    Includes special handling for documentation pages
    """
    def _get_template(self, name: str):
        try:
            return super()._get_template(name)
        except TemplateNotFound:
            fallbacks = {
                "errors/404.html": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>404 Not Found</title>
                    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
                </head>
                <body class="bg-gray-100">
                    <div class="min-h-screen flex items-center justify-center">
                        <div class="bg-white p-8 rounded-lg shadow-md text-center">
                            <h1 class="text-4xl font-bold text-red-600 mb-4">404</h1>
                            <p class="text-lg text-gray-600 mb-6">{{ message }}</p>
                            <a href="/" class="text-blue-600 hover:underline">Return to Home</a>
                        </div>
                    </div>
                </body>
                </html>
                """,
                "components/footer.html": """
                <footer class="bg-gray-800 text-white py-8">
                    <div class="container mx-auto px-6 text-center">
                        <p>© 2023 CrediTrust AI. All rights reserved.</p>
                    </div>
                </footer>
                """,
                "pages/dashboard.html": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Dashboard</title>
                    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
                </head>
                <body class="bg-gray-50">
                    <h1 class="text-3xl font-bold p-6">Complaint Dashboard</h1>
                    <div class="container mx-auto px-6">
                        <p>Dashboard functionality coming soon</p>
                    </div>
                </body>
                </html>
                """
            }
            
            if name in fallbacks:
                return Template(fallbacks[name])
            raise

templates = SafeTemplates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ==============================================
# Middleware and Security
# ==============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Important for documentation
)

security = HTTPBearer()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================
# Documentation Endpoints
# ==============================================
@app.get("/api/docs", include_in_schema=False)
async def redirect_swagger():
    """Redirect /api/docs to /docs for backward compatibility"""
    return RedirectResponse(url="/docs")

@app.get("/api/redoc", include_in_schema=False)
async def redirect_redoc():
    """Redirect /api/redoc to /redoc for backward compatibility"""
    return RedirectResponse(url="/redoc")

@app.get("/api/openapi.json", include_in_schema=False)
async def redirect_openapi():
    """Redirect /api/openapi.json to /openapi.json"""
    return RedirectResponse(url="/openapi.json")

@app.get("/api/docs-status", include_in_schema=False)
async def docs_status():
    """Endpoint to verify documentation availability"""
    return {
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "schema": "/openapi.json"
        },
        "redirects": {
            "api_swagger": "/api/docs",
            "api_redoc": "/api/redoc"
        }
    }

# ==============================================
# HTML Routes
# ==============================================
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    """Main landing page with analytics dashboard"""
    return templates.TemplateResponse(
        "pages/index.html",
        {
            "request": request,
            "page_title": "CrediTrust AI - Financial Compliance Intelligence",
            "features": [
                {"title": "Real-time Regulatory Analysis", "icon": "⚖️", "desc": "Instant compliance checks"},
                {"title": "Predictive Risk Scoring", "icon": "📊", "desc": "AI-powered risk assessment"},
                {"title": "Automated Remediation", "icon": "🤖", "desc": "Smart recommendations"},
                {"title": "Cross-Institution Insights", "icon": "🔍", "desc": "Pattern detection"}
            ],
            "stats": {
                "cases_analyzed": "12.4M+",
                "institutions": "1.2K+",
                "compliance_accuracy": "98.7%",
                "response_time": "2.4s"
            }
        }
    )

@app.get("/dashboard", response_class=HTMLResponse)
async def analysis_dashboard(request: Request):
    """Interactive analysis dashboard"""
    return templates.TemplateResponse(
        "pages/dashboard.html",
        {
            "request": request,
            "page_title": "Complaint Analysis Dashboard",
            "sample_complaints": [
                {"id": 1, "text": "Unauthorized charges...", "type": "Fraud"},
                {"id": 2, "text": "Payment delay...", "type": "Servicing"},
                {"id": 3, "text": "Incorrect balance...", "type": "Account Error"}
            ]
        }
    )

# ==============================================
# Error Handling
# ==============================================
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    """
    Custom 404 handler that:
    - Returns JSON for API routes
    - Returns HTML for frontend routes
    - Excludes documentation paths
    """
    # Skip documentation paths
    if request.url.path in ["/docs", "/redoc", "/openapi.json",
                          "/api/docs", "/api/redoc", "/api/openapi.json"]:
        raise exc
        
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=404,
            content={"detail": "Not Found"}
        )
    return templates.TemplateResponse(
        "errors/404.html",
        {"request": request, "message": "The page you requested was not found"},
        status_code=404
    )

# ==============================================
# Service Initialization (Mocked for example)
# ==============================================
try:
    from src.services.analytics import ComplaintAnalytics
    from src.models.llm_service import LLMService, GenerationConfig
    from src.utils.cache import RedisCache
    from src.models.embedding_service import EmbeddingService
    from src.services.vector_store import VectorStoreManager
    from src.core.retrieval import HierarchicalRetriever
except ImportError as e:
    logger.warning(f"Service imports partially failed: {str(e)}")
    # Mock implementations
    class ComplaintAnalytics: pass
    class GenerationConfig: 
        def __init__(self, model, system_prompt): pass
    class LLMService: 
        def __init__(self, config): pass
    class RedisCache:
        enabled = False
        async def get(self, key): return None
        async def set(self, key, value): pass
        async def initialize(self): pass
    class EmbeddingService: pass
    class VectorStoreManager: 
        def __init__(self, model): pass
    class HierarchicalRetriever: 
        def __init__(self, vector_store): pass

# Initialize services
services_initialized = True
embedding_service = None
vector_store = None
retriever = None
analytics = None
cache = None
llm_service = None

# ==============================================
# API Endpoints (Example)
# ==============================================
class QueryRequest(BaseModel):
    text: str
    products: Optional[List[str]] = None
    date_range: Optional[List[datetime]] = None

@app.post("/analyze")
async def analyze_complaints(request: QueryRequest):
    """Analyze financial complaints"""
    return {"analysis": "Sample response"}

# ==============================================
# Main Application Entry Point
# ==============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)