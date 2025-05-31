"""
Advanced Confidence Scoring System for Enhanced Signal Quality
Implements Subtask 25.2: Advanced Confidence Scoring System
Provides sophisticated confidence algorithms with historical accuracy weighting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceMetrics:
    """Container for confidence-related metrics"""
    base_confidence: float
    historical_accuracy: float
    agreement_score: float
    volatility_adjusted: float
    final_confidence: float
    components: Dict[str, float]


class HistoricalAccuracyTracker:
    """
    Tracks historical accuracy for each model and calculates dynamic weights
    """
    
    def __init__(self, window_size: int = 100, decay_factor: float = 0.95):
        """
        Initialize accuracy tracker
        
        Args:
            window_size: Number of recent predictions to track
            decay_factor: Exponential decay factor for older predictions
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.model_histories: Dict[str, deque] = {}
        self.model_accuracies: Dict[str, float] = {}
        
    def add_prediction(self, model_name: str, prediction: float, actual: float, confidence: float):
        """
        Add a prediction result for accuracy tracking
        
        Args:
            model_name: Name of the model
            prediction: Model prediction (0-1)
            actual: Actual outcome (0-1)
            confidence: Model confidence (0-1)
        """
        if model_name not in self.model_histories:
            self.model_histories[model_name] = deque(maxlen=self.window_size)
        
        # Calculate prediction accuracy (binary classification)
        predicted_class = 1 if prediction > 0.5 else 0
        actual_class = 1 if actual > 0.5 else 0
        is_correct = predicted_class == actual_class
        
        # Store prediction result with metadata
        self.model_histories[model_name].append({
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'is_correct': is_correct,
            'timestamp': pd.Timestamp.now()
        })
        
        # Update model accuracy
        self._update_model_accuracy(model_name)
    
    def _update_model_accuracy(self, model_name: str):
        """Update exponentially weighted accuracy for a model"""
        history = self.model_histories[model_name]
        
        if not history:
            self.model_accuracies[model_name] = 0.5  # Neutral
            return
        
        # Calculate exponentially weighted accuracy
        weights = np.array([self.decay_factor ** i for i in range(len(history))])
        weights = weights[::-1]  # Most recent gets highest weight
        
        accuracies = np.array([entry['is_correct'] for entry in history])
        
        if len(weights) > 0:
            weighted_accuracy = np.average(accuracies, weights=weights)
            self.model_accuracies[model_name] = weighted_accuracy
        else:
            self.model_accuracies[model_name] = 0.5
    
    def get_model_accuracy(self, model_name: str) -> float:
        """Get current accuracy for a model"""
        return self.model_accuracies.get(model_name, 0.5)
    
    def get_confidence_calibration(self, model_name: str) -> Dict[str, float]:
        """
        Calculate confidence calibration metrics for a model
        
        Returns:
            Dictionary with calibration metrics
        """
        if model_name not in self.model_histories:
            return {'calibration_error': 0.5, 'reliability': 0.5}
        
        history = list(self.model_histories[model_name])
        
        if len(history) < 10:
            return {'calibration_error': 0.5, 'reliability': 0.5}
        
        # Calculate calibration error
        confidences = [entry['confidence'] for entry in history]
        accuracies = [entry['is_correct'] for entry in history]
        
        # Bin predictions by confidence
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(confidences, bins) - 1
        
        calibration_error = 0
        reliability = 0
        
        for i in range(len(bins) - 1):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_confidence = np.mean(np.array(confidences)[bin_mask])
                bin_accuracy = np.mean(np.array(accuracies)[bin_mask])
                bin_size = np.sum(bin_mask)
                
                calibration_error += (bin_size / len(history)) * abs(bin_confidence - bin_accuracy)
                reliability += bin_size * (bin_accuracy ** 2)
        
        reliability = reliability / len(history) if len(history) > 0 else 0.5
        
        return {
            'calibration_error': calibration_error,
            'reliability': reliability
        }


class AdvancedConfidenceScorer:
    """
    Advanced confidence scoring system with multiple components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize confidence scorer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.accuracy_tracker = HistoricalAccuracyTracker(
            window_size=self.config.get('window_size', 100),
            decay_factor=self.config.get('decay_factor', 0.95)
        )
        
        # Confidence scoring weights
        self.weights = {
            'base_confidence': self.config.get('base_confidence_weight', 0.3),
            'historical_accuracy': self.config.get('historical_accuracy_weight', 0.3),
            'agreement_score': self.config.get('agreement_score_weight', 0.2),
            'volatility_adjustment': self.config.get('volatility_adjustment_weight', 0.2)
        }
        
        logger.info("Advanced Confidence Scorer initialized")
    
    def calculate_ensemble_confidence(
        self, 
        model_predictions: Dict[str, Any],
        market_data: Optional[np.ndarray] = None
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence score for ensemble prediction
        
        Args:
            model_predictions: Dictionary of model predictions with confidence
            market_data: Optional market data for volatility analysis
            
        Returns:
            ConfidenceMetrics object with detailed confidence breakdown
        """
        # Extract base confidences
        base_confidences = {}
        predictions = {}
        
        for model_name, pred_obj in model_predictions.items():
            base_confidences[model_name] = pred_obj.confidence
            predictions[model_name] = pred_obj.prediction
        
        # Calculate components
        base_confidence = self._calculate_base_confidence(base_confidences)
        historical_accuracy = self._calculate_historical_accuracy_score(model_predictions.keys())
        agreement_score = self._calculate_agreement_score(predictions)
        volatility_adjusted = self._calculate_volatility_adjustment(market_data) if market_data is not None else 1.0
        
        # Calculate final weighted confidence
        final_confidence = (
            self.weights['base_confidence'] * base_confidence +
            self.weights['historical_accuracy'] * historical_accuracy +
            self.weights['agreement_score'] * agreement_score +
            self.weights['volatility_adjustment'] * volatility_adjusted
        )
        
        # Ensure confidence is in [0, 1] range
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        return ConfidenceMetrics(
            base_confidence=base_confidence,
            historical_accuracy=historical_accuracy,
            agreement_score=agreement_score,
            volatility_adjusted=volatility_adjusted,
            final_confidence=final_confidence,
            components={
                'base_confidence': base_confidence,
                'historical_accuracy': historical_accuracy,
                'agreement_score': agreement_score,
                'volatility_adjustment': volatility_adjusted
            }
        )
    
    def _calculate_base_confidence(self, base_confidences: Dict[str, float]) -> float:
        """Calculate weighted average of base model confidences"""
        if not base_confidences:
            return 0.0
        
        # Weight by historical accuracy
        weighted_sum = 0
        total_weight = 0
        
        for model_name, confidence in base_confidences.items():
            accuracy = self.accuracy_tracker.get_model_accuracy(model_name)
            weight = max(0.1, accuracy)  # Minimum weight of 0.1
            
            weighted_sum += confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_historical_accuracy_score(self, model_names: List[str]) -> float:
        """Calculate confidence based on historical accuracy of models"""
        if not model_names:
            return 0.5
        
        accuracies = [self.accuracy_tracker.get_model_accuracy(name) for name in model_names]
        
        # Use weighted average, giving more weight to better performing models
        weights = np.array(accuracies)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)
        
        return np.average(accuracies, weights=weights)
    
    def _calculate_agreement_score(self, predictions: Dict[str, float]) -> float:
        """Calculate confidence based on agreement between models"""
        if len(predictions) < 2:
            return 0.5
        
        pred_values = list(predictions.values())
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(pred_values)):
            for j in range(i + 1, len(pred_values)):
                # Agreement is inverse of absolute difference
                agreement = 1.0 - abs(pred_values[i] - pred_values[j])
                agreements.append(agreement)
        
        # Average agreement
        avg_agreement = np.mean(agreements) if agreements else 0.5
        
        # Also consider variance (lower variance = higher confidence)
        variance = np.var(pred_values)
        variance_score = max(0, 1.0 - variance * 4)  # Scale variance to [0, 1]
        
        # Combine agreement and variance
        return (avg_agreement + variance_score) / 2
    
    def _calculate_volatility_adjustment(self, market_data: np.ndarray) -> float:
        """Adjust confidence based on market volatility"""
        if market_data is None or len(market_data) < 20:
            return 1.0
        
        # Calculate recent volatility (using close prices)
        if market_data.ndim > 1 and market_data.shape[1] >= 4:
            close_prices = market_data[-20:, 3]  # Last 20 close prices
        else:
            close_prices = market_data[-20:]
        
        # Calculate volatility as standard deviation of returns
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns)
        
        # Higher volatility = lower confidence
        # Normalize volatility to [0, 1] range and invert
        normalized_volatility = min(1.0, volatility * 100)  # Scale factor
        volatility_adjustment = 1.0 - normalized_volatility
        
        return max(0.1, volatility_adjustment)  # Minimum adjustment of 0.1
    
    def update_model_performance(self, model_name: str, prediction: float, actual: float, confidence: float):
        """Update historical performance for a model"""
        self.accuracy_tracker.add_prediction(model_name, prediction, actual, confidence)
    
    def get_model_calibration(self, model_name: str) -> Dict[str, float]:
        """Get confidence calibration metrics for a model"""
        return self.accuracy_tracker.get_confidence_calibration(model_name)
    
    def get_confidence_threshold_recommendation(self, target_precision: float = 0.8) -> float:
        """
        Recommend confidence threshold for desired precision
        
        Args:
            target_precision: Desired precision level
            
        Returns:
            Recommended confidence threshold
        """
        # Analyze historical performance at different confidence levels
        all_predictions = []
        
        for model_name, history in self.accuracy_tracker.model_histories.items():
            for entry in history:
                all_predictions.append({
                    'confidence': entry['confidence'],
                    'is_correct': entry['is_correct']
                })
        
        if len(all_predictions) < 50:
            return 0.7  # Default threshold
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Find threshold that achieves target precision
        for i in range(10, len(all_predictions), 10):
            subset = all_predictions[:i]
            precision = sum(p['is_correct'] for p in subset) / len(subset)
            
            if precision >= target_precision:
                return subset[-1]['confidence']
        
        return 0.8  # Fallback threshold
    
    def save_confidence_data(self, filepath: str):
        """Save confidence scoring data"""
        data = {
            'config': self.config,
            'weights': self.weights,
            'model_accuracies': self.accuracy_tracker.model_accuracies,
            'model_histories': {
                name: list(history) 
                for name, history in self.accuracy_tracker.model_histories.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Confidence data saved to {filepath}")
    
    def load_confidence_data(self, filepath: str):
        """Load confidence scoring data"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.config = data.get('config', {})
            self.weights = data.get('weights', self.weights)
            self.accuracy_tracker.model_accuracies = data.get('model_accuracies', {})
            
            # Restore histories
            for model_name, history in data.get('model_histories', {}).items():
                self.accuracy_tracker.model_histories[model_name] = deque(
                    history, maxlen=self.accuracy_tracker.window_size
                )
            
            logger.info(f"Confidence data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load confidence data: {e}")


class ConfidenceBasedPositionSizer:
    """
    Position sizing based on ensemble confidence
    """
    
    def __init__(self, base_position_size: float = 0.02, max_position_size: float = 0.05):
        """
        Initialize confidence-based position sizer
        
        Args:
            base_position_size: Base position size (fraction of portfolio)
            max_position_size: Maximum position size
        """
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
    
    def calculate_position_size(self, confidence: float, volatility: Optional[float] = None) -> float:
        """
        Calculate position size based on confidence and volatility
        
        Args:
            confidence: Ensemble confidence (0-1)
            volatility: Market volatility (optional)
            
        Returns:
            Position size as fraction of portfolio
        """
        # Scale position size by confidence
        confidence_multiplier = confidence ** 2  # Quadratic scaling for more conservative sizing
        position_size = self.base_position_size * confidence_multiplier
        
        # Adjust for volatility if provided
        if volatility is not None:
            volatility_adjustment = 1.0 / (1.0 + volatility * 10)  # Reduce size in high volatility
            position_size *= volatility_adjustment
        
        # Ensure within bounds
        position_size = max(0.001, min(self.max_position_size, position_size))
        
        return position_size
