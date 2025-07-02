from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---

app = FastAPI(
    title="SmartMarketOOPS - Simple ML API",
    description="Simple ML predictions for trading signals",
    version="1.0.0"
)

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---

class PredictionInput(BaseModel):
    """Input data for prediction"""
    symbol: str = Field("BTCUSD", description="Trading symbol")
    features: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

class PredictionOutput(BaseModel):
    """Output data for prediction"""
    symbol: str
    predictions: List[float]
    confidence: float
    predicted_direction: str
    prediction_time: str
    model_version: str

# --- API Endpoints ---

@app.get("/")
async def root():
    return {
        "status": "ok", 
        "message": "Simple ML API is running (v2).",
        "service": "simple_app",
        "endpoints": {
            "predict": "/predict",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SmartMarketOOPS Simple ML API",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput) -> Dict[str, Any]:
    """
    Make a simple prediction using basic logic
    """
    try:
        # Simple prediction logic based on feature values
        features = np.array(input_data.features)
        
        # Calculate simple moving average-like prediction
        avg_feature = np.mean(features)
        trend = features[-1] - features[0] if len(features) > 1 else 0
        
        # Generate probabilities for buy/hold/sell
        if trend > 0.1:
            # Strong upward trend
            probabilities = [0.1, 0.2, 0.7]  # [sell, hold, buy]
            direction = "buy"
        elif trend < -0.1:
            # Strong downward trend
            probabilities = [0.7, 0.2, 0.1]  # [sell, hold, buy]
            direction = "sell"
        else:
            # Sideways/neutral
            probabilities = [0.2, 0.6, 0.2]  # [sell, hold, buy]
            direction = "hold"
        
        confidence = max(probabilities)
        
        response = {
            "symbol": input_data.symbol,
            "predictions": probabilities,
            "confidence": confidence,
            "predicted_direction": direction,
            "prediction_time": pd.Timestamp.now().isoformat(),
            "model_version": "simple-v1.0"
        }
        
        logger.info(f"Generated prediction for {input_data.symbol}: {direction} (confidence: {confidence:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/models/{symbol}")
async def get_model_info(symbol: str) -> Dict[str, Any]:
    """
    Get information about the model for a specific symbol
    """
    return {
        "symbol": symbol,
        "model_type": "simple_predictor",
        "version": "v1.0",
        "status": "active",
        "description": "Simple prediction model using basic feature analysis"
    } 