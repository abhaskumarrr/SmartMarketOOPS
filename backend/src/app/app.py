from fastapi import FastAPI, Depends
from starlette.middleware.cors import CORSMiddleware
import os
import sys
from sqlalchemy.orm import Session

# Import routers
from .auth import router as auth_router
from .core_strategy_service import router as core_strategy_router
from .model_service import router as model_router
from .database import SessionLocal, engine, Base

# Create all tables
Base.metadata.create_all(bind=engine)

# --- FastAPI App Initialization ---

app = FastAPI(
    title="SmartMarketOOPS - Enhanced Trading API",
    description="Provides high-probability trading signals and enhanced ML predictions.",
    version="2.0.0"
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routers ---

app.include_router(auth_router)
app.include_router(core_strategy_router, tags=["strategy"])
app.include_router(model_router, tags=["models"])


# --- Root Endpoint ---

@app.get("/")
async def root():
        return {
        "status": "ok", 
        "message": "Enhanced Trading API is running.",
        "endpoints": {
            "auth": "/auth",
            "strategy": "/strategy",
            "models": "/models",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SmartMarketOOPS API",
        "timestamp": "2025-01-27T16:30:00Z"
    }

