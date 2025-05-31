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

from ..models.model_registry import ModelRegistry
from ..ensemble.signal_quality_system import EnhancedSignalQualitySystem, SignalQualityMetrics
from ..ensemble.multi_model_ensemble import EnsembleConfig
from ..data.transformer_preprocessor import TransformerPreprocessor
from ..data.real_market_data_service import get_market_data_service, MarketDataPoint
from ..utils.config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class EnhancedModelService:
    """
    Enhanced model service that integrates:
    - Traditional model loading and prediction
    - Enhanced Transformer models
    - Multi-model ensemble system
    - Advanced signal quality analysis
    """

    def __init__(self):
        """Initialize the enhanced model service"""
        self.model_registry = ModelRegistry()
        self.loaded_models: Dict[str, Any] = {}
        self.signal_systems: Dict[str, EnhancedSignalQualitySystem] = {}
        self.preprocessors: Dict[str, TransformerPreprocessor] = {}
        self.market_data_service = None

        # Configuration
        self.config = MODEL_CONFIG
        self.ensemble_enabled = self.config.get('ensemble', {}).get('enabled', True)
        self.use_real_data = self.config.get('use_real_market_data', True)

        logger.info("Enhanced Model Service initialized")

    async def initialize_market_data_service(self):
        """Initialize real market data service"""
        if self.use_real_data and self.market_data_service is None:
            try:
                self.market_data_service = await get_market_data_service()
                logger.info("✅ Real market data service initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize market data service: {e}")
                self.use_real_data = False

    async def get_real_market_features(self, symbol: str) -> Dict[str, float]:
        """Get real market features for a symbol"""
        if not self.use_real_data or not self.market_data_service:
            return self._generate_synthetic_features(symbol)

        try:
            # Get latest market data
            market_data = await self.market_data_service.get_latest_data(symbol)

            if market_data:
                features = {
                    'open': market_data.open,
                    'high': market_data.high,
                    'low': market_data.low,
                    'close': market_data.close,
                    'volume': market_data.volume,
                    'timestamp': market_data.timestamp.timestamp()
                }

                # Add spread if available
                if market_data.spread:
                    features['spread'] = market_data.spread

                # Add funding rate if available
                if market_data.funding_rate:
                    features['funding_rate'] = market_data.funding_rate

                # Add open interest if available
                if market_data.open_interest:
                    features['open_interest'] = market_data.open_interest

                return features
            else:
                logger.warning(f"No real market data available for {symbol}, using synthetic")
                return self._generate_synthetic_features(symbol)

        except Exception as e:
            logger.error(f"Error getting real market data for {symbol}: {e}")
            return self._generate_synthetic_features(symbol)

    def _generate_synthetic_features(self, symbol: str) -> Dict[str, float]:
        """Generate synthetic market features as fallback"""
        import random

        # Base prices for different symbols
        base_prices = {
            'BTCUSDT': 45000,
            'ETHUSDT': 2500,
            'SOLUSDT': 100,
            'ADAUSDT': 0.5
        }

        base_price = base_prices.get(symbol, 100)
        volatility = base_price * 0.02

        close = base_price + random.uniform(-volatility, volatility)
        open_price = close * (1 + random.uniform(-0.01, 0.01))
        high = max(open_price, close) * (1 + random.uniform(0, 0.02))
        low = min(open_price, close) * (1 - random.uniform(0, 0.02))
        volume = random.uniform(1000000, 5000000)

        return {
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'timestamp': datetime.now().timestamp()
        }

    def load_model(self, symbol: str, model_version: Optional[str] = None) -> bool:
        """
        Load a model for the given symbol

        Args:
            symbol: Trading symbol
            model_version: Optional specific version to load

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load traditional model
            model, metadata, preprocessor = self.model_registry.load_model(
                symbol=symbol,
                version=model_version,
                return_metadata=True,
                return_preprocessor=True
            )

            if model is None:
                logger.error(f"Failed to load model for {symbol}")
                return False

            # Store loaded model
            self.loaded_models[symbol] = {
                'model': model,
                'metadata': metadata,
                'preprocessor': preprocessor,
                'loaded_at': datetime.now()
            }

            # Initialize enhanced signal quality system if ensemble is enabled
            if self.ensemble_enabled:
                self._initialize_signal_system(symbol, model, metadata)

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

            # Initialize transformer preprocessor
            preprocessor = TransformerPreprocessor(
                sequence_length=metadata.get('sequence_length', 100),
                forecast_horizon=metadata.get('forecast_horizon', 1),
                scaling_method='standard',
                feature_engineering=True,
                multi_timeframe=True,
                attention_features=True
            )
            self.preprocessors[symbol] = preprocessor

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

    async def predict(self, symbol: str, features: Dict[str, float], sequence_length: int = 60) -> Dict[str, Any]:
        """
        Generate enhanced predictions using ensemble system or fallback to traditional model

        Args:
            symbol: Trading symbol
            features: Feature dictionary (optional if using real data)
            sequence_length: Length of input sequence

        Returns:
            Enhanced prediction results
        """
        try:
            # Initialize market data service if needed
            await self.initialize_market_data_service()

            # Get real market features if enabled
            if self.use_real_data:
                real_features = await self.get_real_market_features(symbol)
                # Merge with provided features (provided features take precedence)
                combined_features = {**real_features, **features}
                logger.info(f"Using real market data for {symbol}: close=${real_features.get('close', 0):.2f}")
            else:
                combined_features = features

            # Check if model is loaded
            if symbol not in self.loaded_models:
                success = self.load_model(symbol)
                if not success:
                    raise ValueError(f"Could not load model for {symbol}")

            # Use enhanced signal quality system if available
            if symbol in self.signal_systems and self.ensemble_enabled:
                return self._predict_with_ensemble(symbol, combined_features, sequence_length)
            else:
                return self._predict_traditional(symbol, combined_features, sequence_length)

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return {
                'error': str(e),
                'prediction': 0.5,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

    def _predict_with_ensemble(self, symbol: str, features: Dict[str, float], sequence_length: int) -> Dict[str, Any]:
        """Generate prediction using enhanced signal quality system"""

        # Convert features to market data format
        market_data = self._features_to_market_data(features, sequence_length)

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

    def _predict_traditional(self, symbol: str, features: Dict[str, float], sequence_length: int) -> Dict[str, Any]:
        """Generate prediction using traditional model"""

        model_info = self.loaded_models[symbol]
        model = model_info['model']
        preprocessor = model_info['preprocessor']

        # Prepare input data
        feature_values = list(features.values())

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

    def _features_to_market_data(self, features: Dict[str, float], sequence_length: int) -> np.ndarray:
        """Convert feature dictionary to market data format for ensemble system"""

        # Extract OHLCV data from features
        ohlcv_keys = ['open', 'high', 'low', 'close', 'volume']

        # Create base OHLCV data
        base_data = []
        for key in ohlcv_keys:
            if key in features:
                base_data.append(features[key])
            elif key == 'volume':
                base_data.append(1000.0)  # Default volume
            else:
                # Use close price as fallback
                base_data.append(features.get('close', 100.0))

        # Create sequence by repeating with small variations
        market_data = []
        for i in range(sequence_length):
            # Add small random variations to simulate historical data
            variation = 1 + np.random.normal(0, 0.001)  # 0.1% variation
            row = [val * variation for val in base_data]
            market_data.append(row)

        return np.array(market_data)

    def update_performance(self, symbol: str, prediction: float, actual_outcome: float, confidence: float):
        """Update performance metrics for the enhanced system"""
        if symbol in self.signal_systems:
            self.signal_systems[symbol].update_performance(prediction, actual_outcome, confidence)

    def get_model_status(self, symbol: str) -> Dict[str, Any]:
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
