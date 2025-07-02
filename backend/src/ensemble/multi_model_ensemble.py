"""
Multi-Model Ensemble Framework for Enhanced Signal Quality
Implements Subtask 25.1: Multi-Model Ensemble Framework
Combines Transformer, CNN-LSTM, SMC analysis, and technical indicators
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Container for model prediction with metadata"""
    prediction: Union[float, np.ndarray]
    confidence: float
    model_name: str
    timestamp: Optional[pd.Timestamp] = None
    features_used: Optional[List[str]] = None


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model"""
    models: Dict[str, Dict[str, Any]]
    weights: Dict[str, float]
    confidence_threshold: float = 0.7
    min_models_required: int = 2
    voting_method: str = 'weighted'  # 'weighted', 'majority', 'confidence_weighted'
    dynamic_weights: bool = True
    performance_window: int = 100  # Number of recent predictions for performance tracking


class BaseEnsembleMember(ABC):
    """Abstract base class for ensemble members"""

    @abstractmethod
    def predict(self, data: np.ndarray) -> ModelPrediction:
        """Make prediction with confidence score"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get model name"""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if model is ready for prediction"""
        pass


class TransformerEnsembleMember(BaseEnsembleMember):
    """Transformer model wrapper for ensemble"""

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        self.model = None
        self.model_path = model_path
        self.config = config or {}
        self.name = "enhanced_transformer"

        if model_path and Path(model_path).exists():
            self._load_model()

    def _load_model(self):
        """Load trained Transformer model"""
        try:
            from ml.src.models.transformer_model import EnhancedTransformerModel

            # Load model configuration
            config_path = Path(self.model_path).parent / "model_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
            else:
                model_config = self.config

            # Create and load model
            self.model = EnhancedTransformerModel(**model_config)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()

            logger.info(f"Loaded Transformer model from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load Transformer model: {e}")
            self.model = None

    def predict(self, data: np.ndarray) -> ModelPrediction:
        """Make prediction with confidence"""
        if not self.is_ready():
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name=self.name
            )

        try:
            # Get prediction with confidence
            prediction, confidence = self.model.predict(
                data.reshape(1, *data.shape),
                return_confidence=True
            )

            return ModelPrediction(
                prediction=float(prediction[0]),
                confidence=float(confidence[0]),
                model_name=self.name
            )

        except Exception as e:
            logger.error(f"Transformer prediction error: {e}")
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name=self.name
            )

    def get_name(self) -> str:
        return self.name

    def is_ready(self) -> bool:
        return self.model is not None


class CNNLSTMEnsembleMember(BaseEnsembleMember):
    """CNN-LSTM model wrapper for ensemble"""

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        self.model = None
        self.model_path = model_path
        self.config = config or {}
        self.name = "cnn_lstm"

        if model_path and Path(model_path).exists():
            self._load_model()

    def _load_model(self):
        """Load trained CNN-LSTM model"""
        try:
            from ml.src.models.cnn_lstm_model import CNNLSTMModel

            # Load model
            self.model = CNNLSTMModel(**self.config)
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.model.eval()

            logger.info(f"Loaded CNN-LSTM model from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load CNN-LSTM model: {e}")
            self.model = None

    def predict(self, data: np.ndarray) -> ModelPrediction:
        """Make prediction with confidence"""
        if not self.is_ready():
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name=self.name
            )

        try:
            # CNN-LSTM prediction
            with torch.no_grad():
                input_tensor = torch.FloatTensor(data).unsqueeze(0)
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output).item()

            # Calculate confidence as distance from 0.5
            confidence = abs(prediction - 0.5) * 2

            return ModelPrediction(
                prediction=prediction,
                confidence=confidence,
                model_name=self.name
            )

        except Exception as e:
            logger.error(f"CNN-LSTM prediction error: {e}")
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name=self.name
            )

    def get_name(self) -> str:
        return self.name

    def is_ready(self) -> bool:
        return self.model is not None


class SMCEnsembleMember(BaseEnsembleMember):
    """Smart Money Concepts analysis wrapper for ensemble"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "smc_analyzer"

        # Initialize SMC analyzer
        try:
            from ml.backend.src.strategy.smc_detection import SMCDetection
            self.smc_detector = SMCDetection()
            logger.info("SMC analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SMC analyzer: {e}")
            self.smc_detector = None

    def predict(self, data: np.ndarray) -> ModelPrediction:
        """Analyze SMC patterns and generate signal"""
        if not self.is_ready():
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name=self.name
            )

        try:
            # Convert data to DataFrame format expected by SMC
            # Assuming data is OHLCV format
            if data.shape[-1] >= 5:
                df = pd.DataFrame(data[-100:], columns=['open', 'high', 'low', 'close', 'volume'])

                # Analyze SMC patterns
                order_blocks = self.smc_detector.detect_order_blocks(df)
                fvgs = self.smc_detector.detect_fair_value_gaps(df)
                liquidity = self.smc_detector.detect_liquidity_levels(df)

                # Generate signal based on SMC analysis
                signal_strength = self._calculate_smc_signal(order_blocks, fvgs, liquidity, df)
                confidence = min(signal_strength['confidence'], 1.0)

                return ModelPrediction(
                    prediction=signal_strength['signal'],
                    confidence=confidence,
                    model_name=self.name
                )
            else:
                # Insufficient data
                return ModelPrediction(
                    prediction=0.5,
                    confidence=0.0,
                    model_name=self.name
                )

        except Exception as e:
            logger.error(f"SMC analysis error: {e}")
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name=self.name
            )

    def _calculate_smc_signal(self, order_blocks, fvgs, liquidity, df) -> Dict[str, float]:
        """Calculate signal strength from SMC components"""
        signal_score = 0.5  # Neutral
        confidence_score = 0.0

        current_price = df['close'].iloc[-1]

        # Order block analysis
        if order_blocks:
            recent_ob = order_blocks[-1] if order_blocks else None
            if recent_ob:
                if recent_ob['type'] == 'bullish' and current_price > recent_ob['low']:
                    signal_score += 0.2
                    confidence_score += 0.3
                elif recent_ob['type'] == 'bearish' and current_price < recent_ob['high']:
                    signal_score -= 0.2
                    confidence_score += 0.3

        # FVG analysis
        if fvgs:
            recent_fvg = fvgs[-1] if fvgs else None
            if recent_fvg:
                if recent_fvg['type'] == 'bullish':
                    signal_score += 0.15
                    confidence_score += 0.2
                elif recent_fvg['type'] == 'bearish':
                    signal_score -= 0.15
                    confidence_score += 0.2

        # Liquidity analysis
        if liquidity:
            if liquidity.get('sweep_detected'):
                signal_score += 0.1 if liquidity['direction'] == 'bullish' else -0.1
                confidence_score += 0.2

        # Normalize signal to [0, 1]
        signal_score = max(0, min(1, signal_score))

        return {
            'signal': signal_score,
            'confidence': confidence_score
        }

    def get_name(self) -> str:
        return self.name

    def is_ready(self) -> bool:
        return self.smc_detector is not None


class TechnicalIndicatorEnsembleMember(BaseEnsembleMember):
    """Technical indicators wrapper for ensemble"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "technical_indicators"
        self.indicators = self._initialize_indicators()

    def _initialize_indicators(self):
        """Initialize technical indicators"""
        try:
            import talib
            return {
                'rsi': talib.RSI,
                'macd': talib.MACD,
                'bb': talib.BBANDS,
                'stoch': talib.STOCH
            }
        except ImportError:
            logger.warning("TA-Lib not available, using basic indicators")
            return {}

    def predict(self, data: np.ndarray) -> ModelPrediction:
        """Generate signal from technical indicators"""
        try:
            # Handle both 1D and 2D data
            if data.ndim == 1:
                # Single data point, create minimal sequence
                close_prices = np.array([data[3] if len(data) > 3 else data[-1]] * 20)
            elif data.ndim == 2 and data.shape[1] >= 5:
                close_prices = data[-100:, 3]  # Close prices
            else:
                # Fallback for unexpected data format
                close_prices = np.array([50.0] * 20)  # Default prices

            signals = []
            confidences = []

            # RSI signal
            if len(close_prices) >= 14:
                rsi = self._calculate_rsi(close_prices)
                if rsi is not None:
                    if rsi < 30:
                        signals.append(0.8)  # Oversold - bullish
                        confidences.append(0.7)
                    elif rsi > 70:
                        signals.append(0.2)  # Overbought - bearish
                        confidences.append(0.7)
                    else:
                        signals.append(0.5)  # Neutral
                        confidences.append(0.3)

            # MACD signal
            macd_signal = self._calculate_macd_signal(close_prices)
            if macd_signal is not None:
                signals.append(macd_signal['signal'])
                confidences.append(macd_signal['confidence'])

            # Bollinger Bands signal
            bb_signal = self._calculate_bb_signal(close_prices)
            if bb_signal is not None:
                signals.append(bb_signal['signal'])
                confidences.append(bb_signal['confidence'])

            # Combine signals
            if signals:
                final_signal = np.mean(signals)
                final_confidence = np.mean(confidences)
            else:
                final_signal = 0.5
                final_confidence = 0.0

            return ModelPrediction(
                prediction=final_signal,
                confidence=final_confidence,
                model_name=self.name
            )

        except Exception as e:
            logger.error(f"Technical indicator error: {e}")
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name=self.name
            )

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> Optional[float]:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd_signal(self, prices: np.ndarray) -> Optional[Dict[str, float]]:
        """Calculate MACD signal"""
        if len(prices) < 26:
            return None

        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd = ema12 - ema26
        signal_line = self._ema(np.array([macd]), 9)[0]

        if macd > signal_line:
            return {'signal': 0.7, 'confidence': 0.6}
        else:
            return {'signal': 0.3, 'confidence': 0.6}

    def _calculate_bb_signal(self, prices: np.ndarray, period: int = 20) -> Optional[Dict[str, float]]:
        """Calculate Bollinger Bands signal"""
        if len(prices) < period:
            return None

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        current_price = prices[-1]

        if current_price < lower_band:
            return {'signal': 0.8, 'confidence': 0.7}  # Oversold
        elif current_price > upper_band:
            return {'signal': 0.2, 'confidence': 0.7}  # Overbought
        else:
            return {'signal': 0.5, 'confidence': 0.3}  # Neutral

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def get_name(self) -> str:
        return self.name

    def is_ready(self) -> bool:
        return True


class MultiModelEnsemble:
    """
    Main ensemble class that combines multiple models for enhanced signal quality
    Implements weighted voting, confidence scoring, and dynamic weight adjustment
    """

    def __init__(self, config: EnsembleConfig):
        """
        Initialize the ensemble

        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.models: Dict[str, BaseEnsembleMember] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.prediction_history: List[Dict[str, Any]] = []

        # Initialize models
        self._initialize_models()

        logger.info(f"MultiModelEnsemble initialized with {len(self.models)} models")

    def _initialize_models(self):
        """Initialize all ensemble members"""
        for model_name, model_config in self.config.models.items():
            try:
                if model_name == 'enhanced_transformer':
                    self.models[model_name] = TransformerEnsembleMember(
                        model_path=model_config.get('model_path'),
                        config=model_config.get('config', {})
                    )
                elif model_name == 'cnn_lstm':
                    self.models[model_name] = CNNLSTMEnsembleMember(
                        model_path=model_config.get('model_path'),
                        config=model_config.get('config', {})
                    )
                elif model_name == 'smc_analyzer':
                    self.models[model_name] = SMCEnsembleMember(
                        config=model_config.get('config', {})
                    )
                elif model_name == 'technical_indicators':
                    self.models[model_name] = TechnicalIndicatorEnsembleMember(
                        config=model_config.get('config', {})
                    )

                # Initialize performance history
                self.performance_history[model_name] = []

                logger.info(f"Initialized {model_name}: ready={self.models[model_name].is_ready()}")

            except Exception as e:
                logger.error(f"Failed to initialize {model_name}: {e}")

    def predict(self, data: np.ndarray, return_details: bool = False) -> Union[ModelPrediction, Dict[str, Any]]:
        """
        Generate ensemble prediction

        Args:
            data: Input data for prediction
            return_details: Whether to return detailed prediction breakdown

        Returns:
            Ensemble prediction or detailed results
        """
        # Get predictions from all models
        model_predictions = {}
        ready_models = []

        for model_name, model in self.models.items():
            if model.is_ready():
                try:
                    prediction = model.predict(data)
                    model_predictions[model_name] = prediction
                    ready_models.append(model_name)
                except Exception as e:
                    logger.error(f"Prediction error for {model_name}: {e}")

        # Check if we have minimum required models
        if len(ready_models) < self.config.min_models_required:
            logger.warning(f"Only {len(ready_models)} models ready, minimum required: {self.config.min_models_required}")
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name="ensemble_insufficient_models"
            )

        # Calculate ensemble prediction
        ensemble_result = self._calculate_ensemble_prediction(model_predictions)

        # Store prediction history
        self.prediction_history.append({
            'timestamp': pd.Timestamp.now(),
            'model_predictions': model_predictions,
            'ensemble_result': ensemble_result,
            'ready_models': ready_models
        })

        # Limit history size
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

        if return_details:
            return {
                'ensemble_prediction': ensemble_result,
                'model_predictions': model_predictions,
                'ready_models': ready_models,
                'weights_used': self._get_current_weights(ready_models)
            }

        return ensemble_result

    def _calculate_ensemble_prediction(self, model_predictions: Dict[str, ModelPrediction]) -> ModelPrediction:
        """Calculate weighted ensemble prediction"""
        if not model_predictions:
            return ModelPrediction(
                prediction=0.5,
                confidence=0.0,
                model_name="ensemble_no_predictions"
            )

        # Get current weights
        ready_models = list(model_predictions.keys())
        weights = self._get_current_weights(ready_models)

        if self.config.voting_method == 'weighted':
            return self._weighted_voting(model_predictions, weights)
        elif self.config.voting_method == 'confidence_weighted':
            return self._confidence_weighted_voting(model_predictions, weights)
        elif self.config.voting_method == 'majority':
            return self._majority_voting(model_predictions)
        else:
            return self._weighted_voting(model_predictions, weights)

    def _weighted_voting(self, predictions: Dict[str, ModelPrediction], weights: Dict[str, float]) -> ModelPrediction:
        """Standard weighted voting"""
        total_weight = 0
        weighted_prediction = 0
        weighted_confidence = 0

        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0)
            total_weight += weight
            weighted_prediction += prediction.prediction * weight
            weighted_confidence += prediction.confidence * weight

        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_prediction = 0.5
            final_confidence = 0.0

        return ModelPrediction(
            prediction=final_prediction,
            confidence=final_confidence,
            model_name="ensemble_weighted"
        )

    def _confidence_weighted_voting(self, predictions: Dict[str, ModelPrediction], base_weights: Dict[str, float]) -> ModelPrediction:
        """Confidence-weighted voting"""
        total_weight = 0
        weighted_prediction = 0

        for model_name, prediction in predictions.items():
            base_weight = base_weights.get(model_name, 0)
            confidence_weight = base_weight * prediction.confidence
            total_weight += confidence_weight
            weighted_prediction += prediction.prediction * confidence_weight

        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
            # Ensemble confidence is average of individual confidences weighted by base weights
            final_confidence = sum(pred.confidence * base_weights.get(name, 0)
                                 for name, pred in predictions.items()) / sum(base_weights.values())
        else:
            final_prediction = 0.5
            final_confidence = 0.0

        return ModelPrediction(
            prediction=final_prediction,
            confidence=final_confidence,
            model_name="ensemble_confidence_weighted"
        )

    def _majority_voting(self, predictions: Dict[str, ModelPrediction]) -> ModelPrediction:
        """Simple majority voting (binary classification)"""
        votes = [1 if pred.prediction > 0.5 else 0 for pred in predictions.values()]
        majority_vote = 1 if sum(votes) > len(votes) / 2 else 0

        # Confidence is based on agreement level
        agreement = max(sum(votes), len(votes) - sum(votes)) / len(votes)

        return ModelPrediction(
            prediction=float(majority_vote),
            confidence=agreement,
            model_name="ensemble_majority"
        )

    def _get_current_weights(self, ready_models: List[str]) -> Dict[str, float]:
        """Get current model weights (static or dynamic)"""
        if self.config.dynamic_weights:
            return self._calculate_dynamic_weights(ready_models)
        else:
            # Use static weights from config
            total_weight = sum(self.config.weights.get(model, 0) for model in ready_models)
            if total_weight > 0:
                return {model: self.config.weights.get(model, 0) / total_weight for model in ready_models}
            else:
                # Equal weights if no config weights
                return {model: 1.0 / len(ready_models) for model in ready_models}

    def _calculate_dynamic_weights(self, ready_models: List[str]) -> Dict[str, float]:
        """Calculate dynamic weights based on recent performance"""
        weights = {}

        for model in ready_models:
            # Get recent performance
            recent_performance = self.performance_history.get(model, [])

            if len(recent_performance) >= 10:
                # Use recent average performance
                avg_performance = np.mean(recent_performance[-self.config.performance_window:])
                weights[model] = max(0.1, avg_performance)  # Minimum weight of 0.1
            else:
                # Use static weight if insufficient history
                weights[model] = self.config.weights.get(model, 1.0)

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {model: weight / total_weight for model, weight in weights.items()}

        return weights

    def update_performance(self, model_name: str, performance_score: float):
        """Update model performance history"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []

        self.performance_history[model_name].append(performance_score)

        # Limit history size
        if len(self.performance_history[model_name]) > self.config.performance_window * 2:
            self.performance_history[model_name] = self.performance_history[model_name][-self.config.performance_window:]

    def get_ensemble_confidence(self, data: np.ndarray) -> float:
        """Get ensemble confidence for given input"""
        result = self.predict(data, return_details=True)
        return result['ensemble_prediction'].confidence

    def is_signal_valid(self, data: np.ndarray) -> bool:
        """Check if ensemble signal meets confidence threshold"""
        confidence = self.get_ensemble_confidence(data)
        return confidence >= self.config.confidence_threshold

    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all models in ensemble"""
        status = {}

        for model_name, model in self.models.items():
            recent_performance = self.performance_history.get(model_name, [])
            status[model_name] = {
                'ready': model.is_ready(),
                'recent_performance': np.mean(recent_performance[-10:]) if len(recent_performance) >= 10 else None,
                'prediction_count': len(recent_performance),
                'current_weight': self._get_current_weights([model_name]).get(model_name, 0)
            }

        return status

    def save_ensemble(self, filepath: str):
        """Save ensemble configuration and history"""
        ensemble_data = {
            'config': {
                'models': self.config.models,
                'weights': self.config.weights,
                'confidence_threshold': self.config.confidence_threshold,
                'min_models_required': self.config.min_models_required,
                'voting_method': self.config.voting_method,
                'dynamic_weights': self.config.dynamic_weights,
                'performance_window': self.config.performance_window
            },
            'performance_history': self.performance_history,
            'prediction_history': self.prediction_history[-100:]  # Save last 100 predictions
        }

        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)

        logger.info(f"Ensemble saved to {filepath}")

    @classmethod
    def load_ensemble(cls, filepath: str) -> 'MultiModelEnsemble':
        """Load ensemble from file"""
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)

        # Recreate config
        config = EnsembleConfig(**ensemble_data['config'])

        # Create ensemble
        ensemble = cls(config)

        # Restore history
        ensemble.performance_history = ensemble_data.get('performance_history', {})
        ensemble.prediction_history = ensemble_data.get('prediction_history', [])

        logger.info(f"Ensemble loaded from {filepath}")
        return ensemble
