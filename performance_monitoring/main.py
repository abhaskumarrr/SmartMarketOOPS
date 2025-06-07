"""Performance Monitoring Main Module"""

from fastapi import FastAPI

def create_monitoring_app() -> FastAPI:
    """Create monitoring FastAPI application"""
    app = FastAPI(title="Performance Monitoring")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "monitoring"}
    
    return app
