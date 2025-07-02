"""
Complete Enhanced Signal Quality System
Integrates all components: ensemble, confidence scoring, and market regime detection
Implements Subtask 25.4: Signal Quality Monitoring and Adaptation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json

from .multi_model_ensemble import MultiModelEnsemble, EnsembleConfig, ModelPrediction
from .confidence_scoring import AdvancedConfidenceScorer, ConfidenceMetrics
from .market_regime_detector import MarketRegimeDetector, RegimeAnalysis, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class SignalQualityMetrics:
    """Container for comprehensive signal quality metrics"""
    ensemble_prediction: ModelPrediction
    confidence_metrics: ConfidenceMetrics
    regime_analysis: RegimeAnalysis
    signal_valid: bool
    quality_score: float
    recommendation: str
    components: Dict[str, Any]


class EnhancedSignalQualitySystem:
    """
    Complete enhanced signal quality system that integrates:
    - Multi-model ensemble
    - Advanced confidence scoring
    - Market regime detection
    - Real-time monitoring and adaptation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enhanced signal quality system
        
        Args:
            config: Configuration dictionary with all component settings
        """
        self.config = config
        
        # Initialize components
        ensemble_config = EnsembleConfig(**config.get('ensemble', {}))
        self.ensemble = MultiModelEnsemble(ensemble_config)
        
        self.confidence_scorer = AdvancedConfidenceScorer(config.get('confidence', {}))
        self.regime_detector = MarketRegimeDetector(config.get('regime', {}))
        
        # System settings
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.7)
        self.favorable_regimes = config.get('favorable_regimes', [
            MarketRegime.TRENDING_BULLISH.value,
            MarketRegime.TRENDING_BEARISH.value,
            MarketRegime.BREAKOUT_BULLISH.value,
            MarketRegime.BREAKOUT_BEARISH.value
        ])
        
        # Performance tracking
        self.signal_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'confidence': self.min_confidence_threshold,
            'regime_strength': 0.5,
            'quality_score': 0.6
        }
        
        logger.info("Enhanced Signal Quality System initialized")
    
    def generate_signal(
        self, 
        market_data: np.ndarray, 
        timestamps: Optional[List[pd.Timestamp]] = None,
        return_details: bool = False
    ) -> Union[SignalQualityMetrics, Dict[str, Any]]:
        """
        Generate enhanced signal with comprehensive quality analysis
        
        Args:
            market_data: OHLCV market data
            timestamps: Optional timestamps for data points
            return_details: Whether to return detailed breakdown
            
        Returns:
            SignalQualityMetrics or detailed results dictionary
        """
        current_time = timestamps[-1] if timestamps else pd.Timestamp.now()
        
        # 1. Get ensemble prediction
        ensemble_result = self.ensemble.predict(market_data[-1], return_details=True)
        ensemble_prediction = ensemble_result['ensemble_prediction']
        model_predictions = ensemble_result['model_predictions']
        
        # 2. Calculate advanced confidence metrics
        confidence_metrics = self.confidence_scorer.calculate_ensemble_confidence(
            model_predictions, market_data
        )
        
        # 3. Detect market regime
        regime_analysis = self.regime_detector.detect_regime(market_data, timestamps)
        
        # 4. Determine signal validity
        signal_valid = self._is_signal_valid(
            ensemble_prediction, confidence_metrics, regime_analysis
        )
        
        # 5. Calculate overall quality score
        quality_score = self._calculate_quality_score(
            ensemble_prediction, confidence_metrics, regime_analysis
        )
        
        # 6. Generate recommendation
        recommendation = self._generate_recommendation(
            ensemble_prediction, confidence_metrics, regime_analysis, signal_valid, quality_score
        )
        
        # 7. Create comprehensive metrics
        signal_metrics = SignalQualityMetrics(
            ensemble_prediction=ensemble_prediction,
            confidence_metrics=confidence_metrics,
            regime_analysis=regime_analysis,
            signal_valid=signal_valid,
            quality_score=quality_score,
            recommendation=recommendation,
            components={
                'ensemble_details': ensemble_result,
                'model_predictions': model_predictions,
                'regime_components': regime_analysis.components,
                'confidence_components': confidence_metrics.components
            }
        )
        
        # 8. Store signal history
        self._store_signal_history(signal_metrics, current_time, market_data)
        
        # 9. Update adaptive thresholds
        self._update_adaptive_thresholds()
        
        if return_details:
            return {
                'signal_metrics': signal_metrics,
                'detailed_breakdown': {
                    'ensemble': ensemble_result,
                    'confidence': confidence_metrics,
                    'regime': regime_analysis,
                    'adaptive_thresholds': self.adaptive_thresholds,
                    'performance_metrics': self.performance_metrics
                }
            }
        
        return signal_metrics
    
    def _is_signal_valid(
        self, 
        ensemble_prediction: ModelPrediction,
        confidence_metrics: ConfidenceMetrics,
        regime_analysis: RegimeAnalysis
    ) -> bool:
        """Determine if signal meets all quality criteria"""
        
        # Check confidence threshold
        confidence_valid = confidence_metrics.final_confidence >= self.adaptive_thresholds['confidence']
        
        # Check regime favorability
        regime_favorable = regime_analysis.current_regime.value in self.favorable_regimes
        regime_strong = regime_analysis.regime_strength >= self.adaptive_thresholds['regime_strength']
        
        # Check ensemble agreement
        ensemble_confident = ensemble_prediction.confidence >= 0.6
        
        # All criteria must be met
        return confidence_valid and regime_favorable and regime_strong and ensemble_confident
    
    def _calculate_quality_score(
        self,
        ensemble_prediction: ModelPrediction,
        confidence_metrics: ConfidenceMetrics,
        regime_analysis: RegimeAnalysis
    ) -> float:
        """Calculate overall signal quality score (0-1)"""
        
        # Component weights
        weights = {
            'ensemble_confidence': 0.25,
            'final_confidence': 0.30,
            'regime_strength': 0.20,
            'regime_confidence': 0.15,
            'historical_accuracy': 0.10
        }
        
        # Component scores
        scores = {
            'ensemble_confidence': ensemble_prediction.confidence,
            'final_confidence': confidence_metrics.final_confidence,
            'regime_strength': regime_analysis.regime_strength,
            'regime_confidence': regime_analysis.confidence,
            'historical_accuracy': confidence_metrics.historical_accuracy
        }
        
        # Calculate weighted score
        quality_score = sum(weights[component] * scores[component] for component in weights.keys())
        
        return max(0.0, min(1.0, quality_score))
    
    def _generate_recommendation(
        self,
        ensemble_prediction: ModelPrediction,
        confidence_metrics: ConfidenceMetrics,
        regime_analysis: RegimeAnalysis,
        signal_valid: bool,
        quality_score: float
    ) -> str:
        """Generate trading recommendation based on signal analysis"""
        
        if not signal_valid:
            return "NO_TRADE - Signal quality insufficient"
        
        if quality_score < self.adaptive_thresholds['quality_score']:
            return "WAIT - Signal quality below threshold"
        
        # Determine trade direction
        if ensemble_prediction.prediction > 0.6:
            direction = "BUY"
        elif ensemble_prediction.prediction < 0.4:
            direction = "SELL"
        else:
            return "NEUTRAL - No clear directional bias"
        
        # Determine position size based on confidence
        if confidence_metrics.final_confidence > 0.8:
            size = "LARGE"
        elif confidence_metrics.final_confidence > 0.7:
            size = "MEDIUM"
        else:
            size = "SMALL"
        
        # Consider regime
        regime_modifier = ""
        if regime_analysis.current_regime in [MarketRegime.VOLATILE, MarketRegime.RANGING]:
            regime_modifier = " (CAUTIOUS - volatile/ranging market)"
        elif regime_analysis.transition_probability > 0.7:
            regime_modifier = " (CAUTIOUS - regime transition likely)"
        
        return f"{direction}_{size}{regime_modifier}"
    
    def _store_signal_history(
        self, 
        signal_metrics: SignalQualityMetrics, 
        timestamp: pd.Timestamp,
        market_data: np.ndarray
    ):
        """Store signal in history for performance tracking"""
        
        signal_record = {
            'timestamp': timestamp,
            'prediction': signal_metrics.ensemble_prediction.prediction,
            'confidence': signal_metrics.confidence_metrics.final_confidence,
            'regime': signal_metrics.regime_analysis.current_regime.value,
            'quality_score': signal_metrics.quality_score,
            'signal_valid': signal_metrics.signal_valid,
            'recommendation': signal_metrics.recommendation,
            'market_price': market_data[-1, 3] if len(market_data) > 0 else 0  # Close price
        }
        
        self.signal_history.append(signal_record)
        
        # Limit history size
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent performance"""
        
        if len(self.signal_history) < 50:
            return  # Need sufficient history
        
        recent_signals = self.signal_history[-50:]
        
        # Calculate recent performance metrics
        valid_signals = [s for s in recent_signals if s['signal_valid']]
        
        if len(valid_signals) < 10:
            return  # Need sufficient valid signals
        
        # Analyze performance by confidence level
        high_conf_signals = [s for s in valid_signals if s['confidence'] > 0.8]
        med_conf_signals = [s for s in valid_signals if 0.6 <= s['confidence'] <= 0.8]
        
        # Adjust confidence threshold based on performance
        if len(high_conf_signals) > 0:
            # If we have many high confidence signals, we can be more selective
            if len(high_conf_signals) / len(valid_signals) > 0.3:
                self.adaptive_thresholds['confidence'] = min(0.85, self.adaptive_thresholds['confidence'] + 0.02)
            else:
                self.adaptive_thresholds['confidence'] = max(0.6, self.adaptive_thresholds['confidence'] - 0.01)
        
        # Adjust quality score threshold
        avg_quality = np.mean([s['quality_score'] for s in valid_signals])
        if avg_quality > 0.75:
            self.adaptive_thresholds['quality_score'] = min(0.8, avg_quality - 0.05)
        elif avg_quality < 0.6:
            self.adaptive_thresholds['quality_score'] = max(0.5, avg_quality + 0.05)
        
        logger.info(f"Updated adaptive thresholds: {self.adaptive_thresholds}")
    
    def update_performance(self, prediction: float, actual_outcome: float, confidence: float):
        """Update system performance with actual trading results"""
        
        # Update ensemble model performances
        for model_name in self.ensemble.models.keys():
            # Simplified: assume all models contributed equally
            # In practice, you'd track individual model contributions
            self.ensemble.update_performance(model_name, abs(prediction - actual_outcome))
        
        # Update confidence scorer
        self.confidence_scorer.update_model_performance(
            "ensemble", prediction, actual_outcome, confidence
        )
        
        # Update performance metrics
        self._update_performance_metrics(prediction, actual_outcome)
    
    def _update_performance_metrics(self, prediction: float, actual_outcome: float):
        """Update overall system performance metrics"""
        
        if 'predictions' not in self.performance_metrics:
            self.performance_metrics['predictions'] = []
            self.performance_metrics['outcomes'] = []
        
        self.performance_metrics['predictions'].append(prediction)
        self.performance_metrics['outcomes'].append(actual_outcome)
        
        # Keep only recent performance
        if len(self.performance_metrics['predictions']) > 200:
            self.performance_metrics['predictions'] = self.performance_metrics['predictions'][-200:]
            self.performance_metrics['outcomes'] = self.performance_metrics['outcomes'][-200:]
        
        # Calculate metrics
        predictions = np.array(self.performance_metrics['predictions'])
        outcomes = np.array(self.performance_metrics['outcomes'])
        
        if len(predictions) >= 10:
            # Accuracy (for binary classification)
            pred_binary = (predictions > 0.5).astype(int)
            outcome_binary = (outcomes > 0.5).astype(int)
            accuracy = np.mean(pred_binary == outcome_binary)
            
            # Mean Absolute Error
            mae = np.mean(np.abs(predictions - outcomes))
            
            # Correlation
            correlation = np.corrcoef(predictions, outcomes)[0, 1] if len(predictions) > 1 else 0
            
            self.performance_metrics.update({
                'accuracy': accuracy,
                'mae': mae,
                'correlation': correlation,
                'sample_count': len(predictions)
            })
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'ensemble_status': self.ensemble.get_model_status(),
            'adaptive_thresholds': self.adaptive_thresholds,
            'performance_metrics': self.performance_metrics,
            'signal_history_length': len(self.signal_history),
            'regime_statistics': self.regime_detector.get_regime_statistics(),
            'recent_signals': self.signal_history[-10:] if self.signal_history else []
        }
    
    def save_system_state(self, filepath: str):
        """Save complete system state"""
        
        system_state = {
            'config': self.config,
            'adaptive_thresholds': self.adaptive_thresholds,
            'performance_metrics': self.performance_metrics,
            'signal_history': self.signal_history[-100:]  # Save recent history
        }
        
        # Save main state
        with open(filepath, 'w') as f:
            json.dump(system_state, f, indent=2, default=str)
        
        # Save component states
        base_path = Path(filepath).parent
        
        self.ensemble.save_ensemble(str(base_path / "ensemble_state.pkl"))
        self.confidence_scorer.save_confidence_data(str(base_path / "confidence_state.json"))
        
        logger.info(f"System state saved to {filepath}")
    
    def load_system_state(self, filepath: str):
        """Load complete system state"""
        
        try:
            with open(filepath, 'r') as f:
                system_state = json.load(f)
            
            self.adaptive_thresholds = system_state.get('adaptive_thresholds', self.adaptive_thresholds)
            self.performance_metrics = system_state.get('performance_metrics', {})
            self.signal_history = system_state.get('signal_history', [])
            
            # Load component states
            base_path = Path(filepath).parent
            
            ensemble_path = base_path / "ensemble_state.pkl"
            if ensemble_path.exists():
                self.ensemble = MultiModelEnsemble.load_ensemble(str(ensemble_path))
            
            confidence_path = base_path / "confidence_state.json"
            if confidence_path.exists():
                self.confidence_scorer.load_confidence_data(str(confidence_path))
            
            logger.info(f"System state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
    
    def get_signal_attribution(self) -> Dict[str, float]:
        """Get attribution of recent signals to different components"""
        
        if len(self.signal_history) < 10:
            return {}
        
        recent_signals = self.signal_history[-50:]
        
        # Analyze which components contributed most to valid signals
        valid_signals = [s for s in recent_signals if s['signal_valid']]
        
        if not valid_signals:
            return {}
        
        attribution = {
            'ensemble_contribution': np.mean([s.get('ensemble_confidence', 0.5) for s in valid_signals]),
            'confidence_contribution': np.mean([s['confidence'] for s in valid_signals]),
            'regime_contribution': np.mean([s['quality_score'] for s in valid_signals]),
            'signal_count': len(valid_signals),
            'success_rate': len(valid_signals) / len(recent_signals)
        }
        
        return attribution
