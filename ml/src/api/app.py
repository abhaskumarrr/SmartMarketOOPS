from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os
import sys

# Add the project root's 'ml' directory to the Python path
ml_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ml_root not in sys.path:
    sys.path.insert(0, ml_root)

# Import both strategy and model services
from api.core_strategy_service import router as core_strategy_router
from api.model_service import router as model_router

# --- FastAPI App Initialization ---

app = FastAPI(
    title="SmartMarketOOPS - Enhanced Trading API",
    description="Provides high-probability trading signals and enhanced ML predictions.",
    version="2.0.0"
)

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routers ---

# Include both core strategy and model service routers
app.include_router(core_strategy_router)
app.include_router(model_router)

# --- Root Endpoint ---

@app.get("/")
async def root():
        return {
        "status": "ok", 
        "message": "Enhanced Trading API is running.",
        "endpoints": {
            "core_strategy": "/strategy/predict",
            "model_prediction": "/predict",
            "enhanced_prediction": "/enhanced/predict",
            "model_status": "/models/{symbol}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SmartMarketOOPS ML API",
        "timestamp": "2025-01-27T16:30:00Z"
    }
 