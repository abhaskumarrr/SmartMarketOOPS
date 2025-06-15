#!/usr/bin/env python3
"""
ML Service API server for SMOOPs trading bot
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
# Simplified pipeline function for now
def run_pipeline(symbol: str):
    """Simplified pipeline function - returns mock prediction data"""
    return {
        "symbol": symbol,
        "prediction": "BUY" if hash(symbol) % 2 == 0 else "SELL",
        "confidence": 0.75,
        "price_target": 50000.0,
        "stop_loss": 48000.0,
        "timestamp": "2025-01-27T12:00:00Z",
        "model_version": "smc-v1.0.0",
        "status": "active"
    }

# Import the router from model_service
try:
    from ml.src.api.model_service import router as model_service_router
except ImportError:
    # Create a dummy router if not available
    from fastapi import APIRouter
    model_service_router = APIRouter()

# Add the project root and ml directory to the sys.path
project_root = Path(__file__).parent.parent.parent.parent.parent
ml_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(ml_root))

# Load environment variables
load_dotenv(project_root / ".env")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ml_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ml-service")

# Import the enhanced trading predictions router
try:
    from ml.backend.src.api.enhanced_trading_predictions import router as trading_predictions_router
    ENHANCED_PREDICTIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced trading predictions not available: {e}")
    ENHANCED_PREDICTIONS_AVAILABLE = False
    from fastapi import APIRouter
    trading_predictions_router = APIRouter()

# Create the FastAPI app
app = FastAPI(
    title="SMOOPs ML Service",
    description="Machine Learning service for Smart Money Order Blocks trading bot",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the model_service router
app.include_router(model_service_router, prefix="/api/models")

# Include the enhanced trading predictions router if available
if ENHANCED_PREDICTIONS_AVAILABLE:
    app.include_router(trading_predictions_router, prefix="/api/enhanced")
    logger.info("Enhanced trading predictions router included")
else:
    logger.warning("Enhanced trading predictions router not available")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "SMOOPs ML Service"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/predict/{symbol}")
async def predict(symbol: str):
    """
    Get predictions for a trading symbol using the orchestrator pipeline
    """
    # For demo, just run the pipeline (should be adapted for real symbol input)
    # Note: This endpoint seems to duplicate functionality with the /api/models/predict endpoint.
    # We may need to decide which one to keep or how they should differ.
    result = run_pipeline(symbol)
    return result

@app.get("/api/models") # Note: This endpoint also duplicates functionality.
async def list_models():
    """List available trained models and their performance metrics"""
    # This is dummy data for now
    return {
        "models": [
            {
                "id": "smc-btc-4h-v1",
                "symbol": "BTC-USDT",
                "time_frame": "4h",
                "type": "smart_money_concepts",
                "accuracy": 0.72,
                "profit_factor": 2.1,
                "sharpe_ratio": 1.8,
                "created_at": "2023-05-15T10:00:00Z",
                "last_updated": "2023-05-20T08:30:00Z"
            },
            {
                "id": "smc-eth-4h-v1",
                "symbol": "ETH-USDT",
                "time_frame": "4h",
                "type": "smart_money_concepts",
                "accuracy": 0.68,
                "profit_factor": 1.9,
                "sharpe_ratio": 1.6,
                "created_at": "2023-05-16T14:20:00Z",
                "last_updated": "2023-05-20T09:15:00Z"
            }
        ],
        "count": 2
    }

def main():
    """Run the FastAPI server"""
    port = int(os.getenv("ML_PORT", "3002"))
    logger.info(f"Starting ML service on port {port}")
    # Use reload=False for testing in this context, although the script uses reload=True
    # We can adjust this based on how we want to run it long-term.
    uvicorn.run("ml.backend.src.scripts.server:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()