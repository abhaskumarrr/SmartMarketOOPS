"""Risk Management Main Module"""

from fastapi import FastAPI

def create_risk_app() -> FastAPI:
    """Create risk management FastAPI application"""
    app = FastAPI(title="Risk Management")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "risk"}
    
    return app
