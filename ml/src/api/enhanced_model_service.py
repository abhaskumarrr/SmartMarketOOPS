"""
Enhanced Model Service with Integrated Signal Quality System
Integrates Transformer models and ensemble-based signal generation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import torch
from datetime import datetime

from ..models.model_registry import ModelRegistry, get_registry
from ..ensemble.signal_quality_system import EnhancedSignalQualitySystem, SignalQualityMetrics
from ..ensemble.multi_model_ensemble import EnsembleConfig
from ..utils.config import MODEL_CONFIG
from fastapi import Depends

logger = logging.getLogger(__name__)


class EnhancedModelService:
    """
    Enhanced model service that integrates:
    - Traditional model loading and prediction
    - Enhanced Transformer models
    - Multi-model ensemble system
    - Advanced signal quality analysis
    """

    def __init__(self, model_registry: ModelRegistry = Depends(get_registry)):
        """Initialize the enhanced model service"""
        self.model_registry = model_registry
        self.loaded_models: Dict[str, Any] = {}
        self.signal_systems: Dict[str, EnhancedSignalQualitySystem] = {}

        # Configuration
        self.config = MODEL_CONFIG
        self.ensemble_enabled = self.config.get('ensemble', {}).get('enabled', True)

        logger.info("Enhanced Model Service initialized")

    def load_model(self, symbol: str, model_version: Optional[str] = None) -> bool:
        """
        Load a model for the given symbol using ModelRegistry.

        Args:
            symbol: Trading symbol
            model_version: Optional specific version to load

        Returns:
            True if successful, False otherwise
        """
        try:
            model, metadata, preprocessor = self.model_registry.load_model(
                symbol=symbol,
                version=model_version,
                return_metadata=True,
                return_preprocessor=True
            )

            if model is None:
                logger.error(f"Failed to load model for {symbol}")
                return False

            # Store loaded model, metadata, and preprocessor
            self.loaded_models[symbol] = {
                'model': model,
                'metadata': metadata,
                'preprocessor': preprocessor,
                'loaded_at': datetime.now()
            }

            # Initialize enhanced signal quality system if ensemble is enabled
            if self.ensemble_enabled:
                self._initialize_signal_system(symbol, model, metadata, preprocessor)

            logger.info(f"Model loaded successfully for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return False

    def _initialize_signal_system(self, symbol: str, model: Any, metadata: Dict[str, Any]):
        """Initialize the enhanced signal quality system for a symbol"""
        try:
            # Create ensemble configuration
            ensemble_config = self._create_ensemble_config(symbol, model, metadata)

            # Initialize signal quality system
            signal_system = EnhancedSignalQualitySystem(ensemble_config)
            self.signal_systems[symbol] = signal_system

            logger.info(f"Signal quality system initialized for {symbol}")

        except Exception as e:
            logger.error(f"Failed to initialize signal system for {symbol}: {e}")

    def _create_ensemble_config(self, symbol: str, model: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble configuration for the signal quality system"""

        # Get model path for ensemble members
        model_dir = Path(self.config['model_dir']) / 'registry' / symbol
        latest_version = metadata.get('version', 'latest')
        model_path = model_dir / latest_version / 'model.pt'

        # Base ensemble configuration
        ensemble_models = {}
        ensemble_weights = {}

        # Add models based on configuration
        model_configs = self.config.get('ensemble', {}).get('models', {})

        for model_name, model_config in model_configs.items():
            if model_config.get('enabled', True):
                ensemble_models[model_name] = {
                    'model_path': str(model_path) if model_name in ['enhanced_transformer', 'cnn_lstm'] else None,
                    'config': {
                        'input_dim': metadata.get('input_dim', 20),
                        'output_dim': metadata.get('output_dim', 1),
                        'seq_len': metadata.get('sequence_length', 100),
                        'forecast_horizon': metadata.get('forecast_horizon', 1)
                    }
                }
                ensemble_weights[model_name] = model_config.get('weight', 0.25)

        # Create configuration
        config = {
            'ensemble': {
                'models': ensemble_models,
                'weights': ensemble_weights,
                'confidence_threshold': self.config.get('ensemble', {}).get('confidence_threshold', 0.7),
                'min_models_required': self.config.get('ensemble', {}).get('min_models_required', 2),
                'voting_method': self.config.get('ensemble', {}).get('voting_method', 'confidence_weighted'),
                'dynamic_weights': self.config.get('ensemble', {}).get('dynamic_weights', True)
            },
            'confidence': {
                'window_size': 100,
                'decay_factor': 0.95
            },
            'regime': {
                'adx_period': 14,
                'bb_period': 20,
                'trending_threshold': 25
            },
            'min_confidence_threshold': self.config.get('signal_quality', {}).get('confidence_threshold', 0.7),
            'favorable_regimes': [
                'trending_bullish',
                'trending_bearish',
                'breakout_bullish',
                'breakout_bearish'
            ]
        }

        return config

    async def predict(self, symbol: str, features: List[float], sequence_length: int = 60) -> Dict[str, Any]:
        """
        Generate enhanced predictions using ensemble system or fallback to traditional model

        Args:
            symbol: Trading symbol
            features: Pre-engineered and scaled feature values for prediction
            sequence_length: Length of input sequence

        Returns:
            Enhanced prediction results
        """
        try:
            # Check if model is loaded
            if symbol not in self.loaded_models:
                success = self.load_model(symbol)
                if not success:
                    raise ValueError(f"Could not load model for {symbol}")

            # Use enhanced signal quality system if available
            if symbol in self.signal_systems and self.ensemble_enabled:
                return self._predict_with_ensemble(symbol, features, sequence_length)
            else:
                return self._predict_traditional(symbol, features, sequence_length)

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return {
                'error': str(e),
                'prediction': 0.5,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def _predict_with_ensemble(self, symbol: str, features: List[float], sequence_length: int) -> Dict[str, Any]:
        """Generate prediction using enhanced signal quality system"""

        # Convert features to market data format
        market_data = np.array(features).reshape(1, -1) # Assuming features are already OHLCV + indicators

        # Generate timestamps
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=len(market_data),
            freq='1H'
        ).tolist()

        # Get signal quality system
        signal_system = self.signal_systems[symbol]

        # Generate enhanced signal
        signal_metrics = signal_system.generate_signal(market_data, timestamps)

        # Format response
        return {
            'prediction': signal_metrics.ensemble_prediction.prediction,
            'confidence': signal_metrics.confidence_metrics.final_confidence,
            'signal_valid': signal_metrics.signal_valid,
            'quality_score': signal_metrics.quality_score,
            'recommendation': signal_metrics.recommendation,
            'market_regime': signal_metrics.regime_analysis.current_regime.value,
            'regime_strength': signal_metrics.regime_analysis.regime_strength,
            'model_predictions': {
                name: {
                    'prediction': pred.prediction,
                    'confidence': pred.confidence
                }
                for name, pred in signal_metrics.components['model_predictions'].items()
            },
            'confidence_breakdown': signal_metrics.confidence_metrics.components,
            'timestamp': datetime.now().isoformat(),
            'enhanced': True
        }

    def _predict_traditional(self, symbol: str, features: List[float], sequence_length: int) -> Dict[str, Any]:
        """Generate prediction using traditional model"""

        model_info = self.loaded_models[symbol]
        model = model_info['model']
        preprocessor = model_info['preprocessor']

        # Prepare input data
        feature_values = features

        # Create sequence (repeat last values if needed)
        if len(feature_values) < sequence_length:
            feature_values = feature_values + [feature_values[-1]] * (sequence_length - len(feature_values))

        input_data = np.array(feature_values[-sequence_length:]).reshape(1, sequence_length, -1)

        # Make prediction
        with torch.no_grad():
            if hasattr(model, 'predict'):
                prediction = model.predict(input_data)
                if hasattr(prediction, 'prediction'):
                    pred_value = prediction.prediction
                    confidence = getattr(prediction, 'confidence', 0.5)
                else:
                    pred_value = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
                    confidence = 0.5
            else:
                # Direct model inference
                input_tensor = torch.FloatTensor(input_data)
                output = model(input_tensor)
                pred_value = torch.sigmoid(output).item() if output.numel() == 1 else output.cpu().numpy()
                confidence = 0.5

        return {
            'prediction': float(pred_value),
            'confidence': float(confidence),
            'signal_valid': True,  # Traditional models always considered valid
            'quality_score': confidence,
            'recommendation': 'BUY' if pred_value > 0.6 else 'SELL' if pred_value < 0.4 else 'NEUTRAL',
            'timestamp': datetime.now().isoformat(),
            'enhanced': False
        }

    def update_performance(self, symbol: str, prediction: float, actual_outcome: float, confidence: float):
        """Update performance metrics for the enhanced system"""
        if symbol in self.signal_systems:
            self.signal_systems[symbol].update_performance(prediction, actual_outcome, confidence)

    def get_model_status(self, symbol: str) -> Dict[str, Any>:
        """Get comprehensive model status"""
        if symbol not in self.loaded_models:
            return {'status': 'not_loaded'}

        model_info = self.loaded_models[symbol]
        status = {
            'status': 'loaded',
            'model_type': model_info['metadata'].get('model_type', 'unknown'),
            'version': model_info['metadata'].get('version', 'unknown'),
            'loaded_at': model_info['loaded_at'].isoformat(),
            'enhanced': symbol in self.signal_systems
        }

        # Add ensemble status if available
        if symbol in self.signal_systems:
            ensemble_status = self.signal_systems[symbol].get_system_status()
            status['ensemble_status'] = ensemble_status

        return status

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.loaded_models.keys())

    def is_model_loaded(self, symbol: str) -> bool:
        """Check if model is loaded for symbol"""
        return symbol in self.loaded_models
