"""ML Position Manager Main Module"""

from fastapi import FastAPI

def create_position_app() -> FastAPI:
    """Create position manager FastAPI application"""
    app = FastAPI(title="ML Position Manager")
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "position_manager"}
    
    return app
