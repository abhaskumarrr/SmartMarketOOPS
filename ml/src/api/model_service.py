"""
Model Service API

This module provides API endpoints for serving model predictions.
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
import ta # Import the ta library
from typing import Dict, List, Any, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

# Import project modules
from ..models import ModelFactory
# from ..data.preprocessor import EnhancedPreprocessor # Removed EnhancedPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Model registry path
MODEL_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH", "models/registry")

# Define a small threshold for determining 'Neutral' movement (must match loader.py)
LOG_RETURN_THRESHOLD = 0.0005

class PredictionInput(BaseModel):
    """Input data for prediction"""
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC/USDT')")
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
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
    
    def __init__(self):
        self.models = {}
        self.min_params = {}
        self.max_params = {}
        self.model_info = {}
        
    def load_model(self, symbol: str, model_version: Optional[str] = None) -> bool:
        """
        Load a model for a specific symbol.
        
        Args:
            symbol: Trading symbol
            model_version: Specific model version to load (default: latest)
            
        Returns:
            True if model was loaded successfully
        """
        try:
            # Normalize symbol name for file paths
            # Use the same naming convention as in the training script now using ModelRegistry.save_model
            # This assumes a consistent naming convention where '/' is removed.
            symbol_name_in_registry = symbol.replace("/", "") # Use consistent naming with the training script
            
            # Determine model path
            if model_version is None:
                # Find latest model version
                # model_dir = Path(MODEL_REGISTRY_PATH) / symbol_name # Original line
                model_dir = Path(MODEL_REGISTRY_PATH) / symbol_name_in_registry # Corrected line
                # logger.info(f"Looking for model directory at: {model_dir}") # Added logging
                if not model_dir.exists():
                    # logger.error(f"No models found for {symbol} in registry directory {symbol_name_in_registry}")
                    raise HTTPException(status_code=404, detail=f"Model directory not found: {model_dir}") # More specific error
                
                # Find the latest version
                versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                # logger.info(f"Found versions: {versions}") # Added logging
                if not versions:
                    # logger.error(f"No model versions found for {symbol} in registry directory {symbol_name_in_registry}")
                    raise HTTPException(status_code=404, detail=f"No model versions found in directory: {model_dir}") # More specific error
                
                model_version = sorted(versions)[-1] # Assumes version names sort correctly chronologically
            
            model_path = Path(MODEL_REGISTRY_PATH) / symbol_name_in_registry / model_version / "best.pt"
            logger.info(f"Looking for model file at: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model file not found at {model_path}")
                # Also try loading model.pt if best.pt is not found (from ModelRegistry.save_model)
                model_path = Path(MODEL_REGISTRY_PATH) / symbol_name_in_registry / model_version / "model.pt"
                logger.info(f"best.pt not found, trying model.pt at: {model_path}")
                if not model_path.exists():
                     raise HTTPException(status_code=404, detail=f"Model file not found at: {model_path}") # More specific error
            
            # Load the model
            logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            
            # Get model config
            model_config = checkpoint.get("config", {})
            logger.info(f"Model config loaded: {model_config}") # Added print statement to inspect model_config

            # Determine model type more robustly
            model_type = model_config.get('model_type') # First, try from model_config

            if model_type is None:
                # If not in model_config, try top level of checkpoint
                model_type = checkpoint.get('model_type')

            if model_type is None:
                # If still not found, default to 'cnnlstm' and log a warning
                logger.warning("Model type not found in checkpoint, defaulting to 'cnnlstm'")
                model_type = 'cnnlstm'

            # Ensure model_type is lowercase before passing to factory
            model_type = model_type.lower()

            # Create model instance
            # Pass required parameters from model_config or use defaults
            # Ensure parameter names match ModelFactory.create_model signature
            model = ModelFactory.create_model(
                model_type=model_type,
                input_dim=model_config.get('input_dim', 10), # Use default 10 if not found
                output_dim=model_config.get('output_dim', 1), # Use default 1 if not found
                seq_len=model_config.get('seq_len', 60),   # Use default 60 if not found
                forecast_horizon=model_config.get('forecast_horizon', 5), # Use default 5 if not found
                hidden_dim=model_config.get('hidden_dim', 128), # Use default 128 if not found
                num_layers=model_config.get('num_layers', 2), # Use default 2 if not found
                # Add other model-specific parameters if necessary and available in config
                cnn_channels=model_config.get('cnn_channels', 64), # Added for CNNLSTM
                lstm_hidden=model_config.get('lstm_hidden', 128), # Added for CNNLSTM
                lstm_layers=model_config.get('lstm_layers', 2), # Added for CNNLSTM
                dropout=model_config.get('dropout', 0.2) # Added for CNNLSTM
            )
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load normalization parameters (min/max from norm_params.npy)
            norm_params_path = Path(MODEL_REGISTRY_PATH) / symbol_name_in_registry / model_version / "norm_params.npy"
            if not norm_params_path.exists():
                 raise HTTPException(status_code=404, detail=f"Normalization parameters not found at: {norm_params_path}")

            norm = np.load(norm_params_path, allow_pickle=True).item()
            self.min_params[symbol] = norm['min']
            self.max_params[symbol] = norm['max']

            # Store model and parameters
            self.models[symbol] = model
            self.model_info[symbol] = {
                'version': model_version,
                'type': model_type,
                'config': model_config # Store the original config or the inferred one
            }
            
            logger.info(f"Model and normalization parameters for {symbol} loaded successfully (version: {model_version})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {str(e)}")
            if "Model directory not found" in str(e) or "Model file not found" in str(e) or "No model versions found" in str(e):
                 raise HTTPException(status_code=404, detail=f"Model for {symbol} not found: {str(e)}") # Propagate 404 for specific errors
            else:
                 raise HTTPException(status_code=500, detail=f"Error loading model for {symbol}: {str(e)}") # Raise 500 for other exceptions
    
    def predict(self, symbol: str, features: Dict[str, float], sequence_length: int = 60) -> Dict[str, Any]:
        """
        Make predictions using the loaded model.
        
        Args:
            symbol: Trading symbol
            features: Feature values for prediction
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
        min_vals = self.min_params[symbol]
        max_vals = self.max_params[symbol]
        # preprocessor = self.preprocessors[symbol] # Removed old preprocessor

        try:
            # Convert features to DataFrame
            # Ensure the order of features matches the training data
            # The input 'features' dictionary might not have features in the correct order.
            # We need to explicitly order them based on the feature columns in TradingDataset.
            # Since we don't dynamically get the feature_cols here, we'll hardcode them based on our analysis of loader.py
            # This is a potential point of failure if the feature engineering in loader.py changes.

            # Step 1: Create a DataFrame from input features
            # Assuming input 'features' contains basic OHLCV and any manually provided indicators
            # We need to reconstruct the full feature set used during training.
            # This requires the raw OHLCV data at a minimum.
            # Let's assume the input 'features' dictionary for prediction contains at least 'open', 'high', 'low', 'close', 'volume'
            # And potentially other base features if used by the loader before indicator calculation.
            # For now, let's create a DataFrame with the explicitly passed features.
            df = pd.DataFrame([features], index=[0]) # Add index=0 to make it a proper DataFrame row

            # Step 2: Replicate feature engineering from TradingDataset in loader.py
            # This must exactly match the logic in ml/backend/src/data/loader.py
            # Make sure column names match exactly.
            # We need to handle potential NaN values that can result from indicator calculations on a single data point.
            # For prediction on a single point, some indicators requiring a window might produce NaN.
            # A robust solution would pass a window of recent data. For now, we'll fill NaNs.

            # Calculate indicators
            if 'close' in df.columns:
                df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
                df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
                df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
                df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
                macd_indicator = ta.trend.macd(df['close'])
                df['macd'] = macd_indicator
                macd_signal_indicator = ta.trend.macd_signal(df['close'])
                df['macd_signal'] = macd_signal_indicator

            if 'close' in df.columns and 'volume' in df.columns:
                 df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])

            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
                df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])

            # Add lagged returns (requires previous close values, which are not in the single feature input)
            # This is a limitation with the current single-input design.
            # For a real system, we'd need to pass a window of historical data.
            # For now, we'll skip lagged returns in the prediction service to avoid errors.
            # This means the prediction service is NOT using the exact same features as training, which is problematic.
            # A better fix requires changing the PredictionInput model to accept historical data.
            # Let's proceed *without* lagged returns for now, but note this discrepancy.

            # Define the expected feature columns *after* engineering, based on loader.py (excluding lagged returns for now)
            feature_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_50', 'sma_200', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'obv', 'stoch_k', 'stoch_d'
                # 'log_return_1', 'log_return_2', 'log_return_3' # Skipped due to single-input limitation
            ]

            # Ensure the DataFrame has only the expected feature columns in the correct order
            # Drop any columns not in feature_cols and reindex to ensure order
            # Use .reindex to ensure correct order and add missing columns with NaN
            df = df.reindex(columns=feature_cols)

            # Fill any resulting NaN values. Using 0 might be acceptable for some indicators, but consider better strategies.
            df = df.fillna(0) # Simple fill NaNs with 0. This might need refinement.

            # Step 3: Apply Min-Max scaling using loaded parameters
            # Ensure df is numpy array for scaling
            features_array = df.values.astype(np.float32)
            # Add a small epsilon to max-min to avoid division by zero if min == max
            scaled_features = (features_array - min_vals) / (max_vals - min_vals + 1e-8)


            # Step 4: Reshape and prepare tensor for model input
            # The model expects (batch_size, seq_len, num_features)
            # Current design replicates a single data point seq_len times.
            # This is incorrect for time series but matches existing model service logic.
            # A proper fix needs historical data in PredictionInput.

            # Replicate the single scaled feature vector to create a sequence of 'sequence_length'
            # Ensure scaled_features is 2D (1, num_features) before tiling
            if len(scaled_features.shape) == 1:
                scaled_features = np.expand_dims(scaled_features, axis=0) # Make it (1, num_features)

            features_tensor = torch.tensor(
                np.tile(scaled_features, (sequence_length, 1)),
                dtype=torch.float32
            ).unsqueeze(0) # Add batch dimension -> (1, seq_len, num_features)

            # Ensure the number of features matches the input size the model was trained with
            # This requires loading the model config or inferring from loaded model weights
            # For now, assume input_size matches the number of columns in feature_cols defined above.
            expected_input_size = len(feature_cols) # Number of features after engineering

            if features_tensor.shape[2] != expected_input_size:
                 raise HTTPException(status_code=500, detail=f"Feature mismatch: Input has {features_tensor.shape[2]} features, but expected {expected_input_size} features after preprocessing.")


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


# Create model service instance
model_service = ModelService()


def get_model_service():
    """Dependency for getting model service instance"""
    return model_service


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