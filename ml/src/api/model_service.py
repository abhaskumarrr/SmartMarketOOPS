"""
Model Service API

This module provides API endpoints for serving model predictions.
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
from ..data.unified_data_processor import UnifiedDataProcessor
from typing import Dict, Any, List, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import json

# Import project modules
from ..models import ModelFactory
from ..models.model_registry import get_registry, ModelRegistry
from .enhanced_model_service import EnhancedModelService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'default_config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Create API router
router = APIRouter()

# Model registry path from config
MODEL_REGISTRY_PATH = config['model_registry']['path']


# Define a small threshold for determining 'Neutral' movement (must match loader.py)
LOG_RETURN_THRESHOLD = 0.0005

class PredictionInput(BaseModel):
    """Input data for prediction"""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    features: List[float] = Field(..., description="Pre-engineered and scaled feature values for prediction")
    sequence_length: Optional[int] = Field(60, description="Length of input sequence")


class PredictionOutput(BaseModel):
    """Output data for prediction"""
    symbol: str
    predictions: List[float]
    confidence: Optional[float]
    predicted_direction: Optional[str]
    prediction_time: str
    model_version: str


class ModelService:
    """Service for loading and running models"""

    def __init__(self, registry: ModelRegistry = Depends(get_registry)):
        self.registry = registry
        self.models = {}
        self.min_params = {}
        self.max_params = {}
        self.model_info = {}
        self.data_processor = UnifiedDataProcessor() # Initialize UnifiedDataProcessor

    def load_model(self, symbol: str, model_version: Optional[str] = None) -> bool:
        """
        Load a model for a specific symbol using ModelRegistry.

        Args:
            symbol: Trading symbol
            model_version: Specific model version to load (default: latest)

        Returns:
            True if model was loaded successfully
        """
        try:
            model, metadata, preprocessor = self.registry.load_model(
                symbol=symbol,
                version=model_version,
                return_metadata=True,
                return_preprocessor=True
            )

            self.models[symbol] = model
            self.min_params[symbol] = preprocessor['feature_scaler'].min_ # Assuming min_ and max_ are attributes of the scaler
            self.max_params[symbol] = preprocessor['feature_scaler'].max_
            self.model_info[symbol] = {
                'version': metadata.get('version', 'unknown'),
                'type': metadata.get('model_type', 'unknown'),
                'config': metadata.get('config', {})
            }
            logger.info(f"Model and normalization parameters for {symbol} loaded successfully (version: {self.model_info[symbol]['version']})")
            return True
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"Model for {symbol} not found: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading model for {symbol}: {str(e)}")

    def predict(self, symbol: str, features: List[float], sequence_length: int = 60) -> Dict[str, Any]:
        """
        Make predictions using the loaded model.

        Args:
            symbol: Trading symbol
            features: Pre-engineered and scaled feature values for prediction
            sequence_length: Length of input sequence

        Returns:
            Dictionary with prediction results
        """
        # Check if model is loaded
        if symbol not in self.models:
            loaded = self.load_model(symbol)

        # Check again if model was successfully loaded after attempting to load
        if symbol not in self.models:
             raise HTTPException(status_code=500, detail=f"Model for {symbol} could not be loaded.") # Ensure model is available

        model = self.models[symbol]

        try:
            try:
            # Convert features to DataFrame
            df = pd.DataFrame([features])

            # Transform data using UnifiedDataProcessor
            # This assumes the input features are raw OHLCV and the processor will engineer them
            # For a single prediction, we need to ensure the processor can handle it correctly
            processed_features = self.data_processor.transform_inference_data(df)

            # Reshape for model input
            # The processed_features is already (1, sequence_length, num_features)
            features_tensor = torch.tensor(processed_features, dtype=torch.float32)


            # Make prediction
            with torch.no_grad():
                # model expects input of shape (batch_size, seq_len, input_size)
                raw_predictions = model(features_tensor).cpu().numpy()[0]

            # Postprocess predictions and calculate confidence/direction
            # raw_predictions are logits from the classification model
            # Apply Softmax to get probabilities
            probabilities = torch.softmax(torch.tensor(raw_predictions, dtype=torch.float32), dim=-1).numpy()

            # Get the maximum probability as the confidence score
            confidence_score = float(np.max(probabilities))

            # Get the predicted class index (0, 1, or 2)
            predicted_class_index = int(np.argmax(probabilities))

            # Map class index to direction string
            # Ensure this mapping matches the target_direction definition in loader.py
            # 0: Down, 1: Neutral, 2: Up
            direction_map = {0: "down", 1: "neutral", 2: "up"}
            predicted_direction = direction_map.get(predicted_class_index, "neutral") # Use .get for safety

            # The 'predictions' field in PredictionOutput should likely represent the probabilities for each class
            # rather than a single value from inverse transform (which is not applicable for classification).
            # Let's return the probabilities as the 'predictions' list.
            predictions_list = probabilities.tolist()

            # Get model version from loaded info
            model_version = self.model_info.get(symbol, {}).get('version', 'unknown')

            # Create response dictionary
            response_data = {
                "symbol": symbol,
                "predictions": predictions_list, # Return probabilities for each class
                "confidence": confidence_score,
                "predicted_direction": predicted_direction,
                "prediction_time": pd.Timestamp.now().isoformat(), # Use current time for prediction
                "model_version": model_version
            }

            return response_data

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Create model service instances
model_service = ModelService()
enhanced_model_service = EnhancedModelService()


def get_model_service():
    """Dependency for getting model service instance"""
    return model_service


def get_enhanced_model_service():
    """Dependency for getting enhanced model service instance"""
    return enhanced_model_service


@router.post("/predict", response_model=PredictionOutput)
async def predict(
    input_data: PredictionInput,
    service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Make a prediction for the given input data.
    """
    result = service.predict(
        symbol=input_data.symbol,
        features=input_data.features,
        sequence_length=input_data.sequence_length
    )
    return result


@router.get("/models/{symbol}")
async def get_model_info(
    symbol: str,
    service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Get information about the model for a specific symbol.
    """
    if symbol not in service.model_info:
        # Try to load the model
        loaded = service.load_model(symbol)
        if not loaded:
            raise HTTPException(status_code=404, detail=f"Model for {symbol} not found")

    return service.model_info[symbol]


@router.post("/models/{symbol}/load")
async def load_model(
    symbol: str,
    model_version: Optional[str] = None,
    service: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Load a model for a specific symbol.
    """
    success = service.load_model(symbol, model_version)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model for {symbol} not found")

    return {"status": "success", "model_info": service.model_info[symbol]}


# Enhanced API Endpoints with Signal Quality System

class EnhancedPredictionOutput(BaseModel):
    """Enhanced output data for prediction with signal quality metrics"""
    symbol: str
    prediction: float
    confidence: float
    signal_valid: bool
    quality_score: float
    recommendation: str
    market_regime: str
    regime_strength: float
    model_predictions: Dict[str, Dict[str, float]]
    confidence_breakdown: Dict[str, float]
    prediction_time: str
    enhanced: bool


@router.post("/enhanced/predict", response_model=EnhancedPredictionOutput)
async def enhanced_predict(
    input_data: PredictionInput,
    service: EnhancedModelService = Depends(get_enhanced_model_service)
) -> Dict[str, Any]:
    """
    Make an enhanced prediction using the signal quality system.
    """
    result = service.predict(
        symbol=input_data.symbol,
        features=input_data.features,
        sequence_length=input_data.sequence_length
    )

    # Format for enhanced output model
    return {
        "symbol": input_data.symbol,
        "prediction": result.get('prediction', 0.5),
        "confidence": result.get('confidence', 0.0),
        "signal_valid": result.get('signal_valid', False),
        "quality_score": result.get('quality_score', 0.0),
        "recommendation": result.get('recommendation', 'NEUTRAL'),
        "market_regime": result.get('market_regime', 'unknown'),
        "regime_strength": result.get('regime_strength', 0.0),
        "model_predictions": result.get('model_predictions', {}),
        "confidence_breakdown": result.get('confidence_breakdown', {}),
        "prediction_time": result.get('timestamp', pd.Timestamp.now().isoformat()),
        "enhanced": result.get('enhanced', True)
    }


@router.get("/enhanced/models/{symbol}/status")
async def get_enhanced_model_status(
    symbol: str,
    service: EnhancedModelService = Depends(get_enhanced_model_service)
) -> Dict[str, Any]:
    """
    Get comprehensive status of the enhanced model system for a symbol.
    """
    return service.get_model_status(symbol)


@router.post("/enhanced/models/{symbol}/load")
async def load_enhanced_model(
    symbol: str,
    model_version: Optional[str] = None,
    service: EnhancedModelService = Depends(get_enhanced_model_service)
) -> Dict[str, Any]:
    """
    Load an enhanced model for a specific symbol.
    """
    success = service.load_model(symbol, model_version)
    if not success:
        raise HTTPException(status_code=404, detail=f"Enhanced model for {symbol} not found")

    return {
        "status": "success",
        "model_status": service.get_model_status(symbol),
        "enhanced": True
    }


@router.post("/enhanced/models/{symbol}/performance")
async def update_model_performance(
    symbol: str,
    prediction: float,
    actual_outcome: float,
    confidence: float,
    service: EnhancedModelService = Depends(get_enhanced_model_service)
) -> Dict[str, Any]:
    """
    Update model performance with actual trading results.
    """
    service.update_performance(symbol, prediction, actual_outcome, confidence)
    return {"status": "success", "message": "Performance updated"}


@router.get("/enhanced/models")
async def list_enhanced_models(
    service: EnhancedModelService = Depends(get_enhanced_model_service)
) -> Dict[str, Any]:
    """
    List all available enhanced models.
    """
    return {
        "models": service.get_available_models(),
        "enhanced": True
    }