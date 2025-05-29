"""
ML API Application

This module provides the main FastAPI application for the ML module.
"""

import os
import logging
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, List, Optional

# Import project modules
from .model_service import router as model_router
from ..models.model_registry import get_registry, ModelRegistry
from ..monitoring.api import router as monitoring_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SMOOPs ML API",
    description="API for SMOOPs ML module",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include routers
app.include_router(model_router, prefix="/api/models", tags=["Models"])
app.include_router(monitoring_router, prefix="/api/monitoring", tags=["Monitoring"])

# Mount static files for reports
reports_dir = os.path.join(os.getcwd(), "reports")
os.makedirs(reports_dir, exist_ok=True)
app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")


# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """Root endpoint for health check"""
    return {"status": "ok", "message": "SMOOPs ML API is running"}


# Registry endpoints
@app.get("/api/registry/symbols", tags=["Registry"])
async def list_symbols(registry: ModelRegistry = Depends(get_registry)) -> Dict[str, List[str]]:
    """List all available symbols in the registry"""
    symbols = registry.get_symbols()
    return {"symbols": symbols}


@app.get("/api/registry/versions/{symbol}", tags=["Registry"])
async def list_versions(
    symbol: str,
    registry: ModelRegistry = Depends(get_registry)
) -> Dict[str, List[str]]:
    """List all available versions for a symbol"""
    versions = registry.get_versions(symbol)
    return {"symbol": symbol, "versions": versions}


@app.get("/api/registry/metadata/{symbol}", tags=["Registry"])
async def get_metadata(
    symbol: str,
    version: Optional[str] = None,
    registry: ModelRegistry = Depends(get_registry)
) -> Dict[str, Any]:
    """Get metadata for a specific model version"""
    try:
        metadata = registry.get_metadata(symbol, version)
        return {
            "symbol": symbol,
            "version": version or "latest",
            "metadata": metadata
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/registry/versions/{symbol}/{version}", tags=["Registry"])
async def delete_version(
    symbol: str,
    version: str,
    force: bool = False,
    registry: ModelRegistry = Depends(get_registry)
) -> Dict[str, Any]:
    """Delete a specific model version"""
    try:
        success = registry.delete_version(symbol, version, force)
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Could not delete version {version} for {symbol}. It may be the only version."
            )
        return {"status": "success", "message": f"Deleted model version {version} for {symbol}"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/registry/compare/{symbol}", tags=["Registry"])
async def compare_versions(
    symbol: str,
    version1: str,
    version2: str,
    registry: ModelRegistry = Depends(get_registry)
) -> Dict[str, Any]:
    """Compare two model versions"""
    try:
        comparison = registry.compare_versions(symbol, version1, version2)
        return comparison
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )


# Create app factory function
def create_app() -> FastAPI:
    """Create FastAPI application"""
    return app


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.environ.get("ML_API_PORT", 8000))
    
    # Run the application
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True) 