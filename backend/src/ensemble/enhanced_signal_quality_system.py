"""
Enhanced Signal Quality System for SmartMarketOOPS
Task #25: Enhanced Signal Quality System with Transformer Integration
Implements confidence scoring, signal validation, and quality metrics
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for trading decisions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class SignalQuality(Enum):
    """Signal quality levels"""
    EXCELLENT = "excellent"  # >90% confidence
    GOOD = "good"           # 70-90% confidence
    FAIR = "fair"           # 50-70% confidence
    POOR = "poor"           # <50% confidence


@dataclass
class TradingSignal:
    """Enhanced trading signal with quality metrics"""
    signal_type: SignalType
    confidence: float
    quality: SignalQuality
    timestamp: datetime
    symbol: str
    price: float

    # Signal components
    transformer_prediction: float
    ensemble_prediction: float
    smc_score: float
    technical_score: float

    # Quality metrics
    model_agreement: float
    historical_accuracy: float
    market_condition_score: float
    volatility_adjustment: float

    # Risk metrics
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    risk_reward_ratio: Optional[float] = None

    # Metadata
    features_used: Optional[List[str]] = None
    model_versions: Optional[Dict[str, str]] = None
    processing_time_ms: Optional[float] = None


class EnhancedSignalQualitySystem:
    """
    Advanced signal quality system that integrates Transformer predictions
    with ensemble methods and comprehensive quality assessment
    """

    def __init__(
        self,
        transformer_model,
        ensemble_models: List[Any],
        confidence_threshold: float = 0.6,
        quality_weights: Optional[Dict[str, float]] = None,
        historical_window: int = 100,
        volatility_lookback: int = 20
    ):
        """
        Initialize the enhanced signal quality system

        Args:
            transformer_model: Trained Transformer model
            ensemble_models: List of additional models for ensemble
            confidence_threshold: Minimum confidence for signal generation
            quality_weights: Weights for different quality components
            historical_window: Window for historical accuracy calculation
            volatility_lookback: Lookback period for volatility calculation
        """
        self.transformer_model = transformer_model
        self.ensemble_models = ensemble_models
        self.confidence_threshold = confidence_threshold
        self.historical_window = historical_window
        self.volatility_lookback = volatility_lookback

        # Default quality weights
        self.quality_weights = quality_weights or {
            'transformer_weight': 0.4,
            'ensemble_weight': 0.3,
            'smc_weight': 0.15,
            'technical_weight': 0.15
        }

        # Signal history for quality tracking
        self.signal_history: List[TradingSignal] = []
        self.performance_metrics = {
            'total_signals': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'average_confidence': 0.0,
            'quality_distribution': {q.value: 0 for q in SignalQuality}
        }

        logger.info("Enhanced Signal Quality System initialized")

    def generate_signal(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        current_price: float,
        smc_analysis: Optional[Dict] = None,
        technical_indicators: Optional[Dict] = None
    ) -> Optional[TradingSignal]:
        """
        Generate a high-quality trading signal with comprehensive analysis

        Args:
            market_data: Recent market data for analysis
            symbol: Trading symbol
            current_price: Current market price
            smc_analysis: Smart Money Concepts analysis results
            technical_indicators: Technical indicator values

        Returns:
            TradingSignal if quality meets threshold, None otherwise
        """
        start_time = datetime.now()

        try:
            # 1. Get Transformer prediction
            transformer_pred, transformer_conf = self._get_transformer_prediction(market_data)

            # 2. Get ensemble predictions
            ensemble_pred, ensemble_conf = self._get_ensemble_prediction(market_data)

            # 3. Calculate SMC score
            smc_score = self._calculate_smc_score(smc_analysis) if smc_analysis else 0.5

            # 4. Calculate technical score
            technical_score = self._calculate_technical_score(technical_indicators) if technical_indicators else 0.5

            # 5. Calculate model agreement
            model_agreement = self._calculate_model_agreement(
                transformer_pred, ensemble_pred, smc_score, technical_score
            )

            # 6. Calculate market condition score
            market_condition_score = self._assess_market_conditions(market_data)

            # 7. Calculate volatility adjustment
            volatility_adjustment = self._calculate_volatility_adjustment(market_data)

            # 8. Calculate historical accuracy
            historical_accuracy = self._get_historical_accuracy(symbol)

            # 9. Combine all components for final confidence
            final_confidence = self._calculate_final_confidence(
                transformer_conf, ensemble_conf, model_agreement,
                market_condition_score, volatility_adjustment, historical_accuracy
            )

            # 10. Determine signal type and quality
            signal_type = self._determine_signal_type(
                transformer_pred, ensemble_pred, final_confidence
            )
            signal_quality = self._determine_signal_quality(final_confidence)

            # 11. Check if signal meets quality threshold
            if final_confidence < self.confidence_threshold:
                logger.debug(f"Signal confidence {final_confidence:.3f} below threshold {self.confidence_threshold}")
                return None

            # 12. Calculate risk metrics
            stop_loss, take_profit, position_size, risk_reward = self._calculate_risk_metrics(
                signal_type, current_price, final_confidence, volatility_adjustment
            )

            # 13. Create trading signal
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            signal = TradingSignal(
                signal_type=signal_type,
                confidence=final_confidence,
                quality=signal_quality,
                timestamp=datetime.now(),
                symbol=symbol,
                price=current_price,
                transformer_prediction=transformer_pred,
                ensemble_prediction=ensemble_pred,
                smc_score=smc_score,
                technical_score=technical_score,
                model_agreement=model_agreement,
                historical_accuracy=historical_accuracy,
                market_condition_score=market_condition_score,
                volatility_adjustment=volatility_adjustment,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_reward_ratio=risk_reward,
                processing_time_ms=processing_time
            )

            # 14. Update signal history
            self._update_signal_history(signal)

            logger.info(f"Generated {signal_quality.value} quality {signal_type.value} signal "
                       f"with {final_confidence:.3f} confidence for {symbol}")

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None

    def _get_transformer_prediction(self, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Get prediction and confidence from Transformer model"""
        try:
            # Prepare data for Transformer
            # This would use the TransformerPreprocessor
            processed_data = self._prepare_transformer_input(market_data)

            # Get prediction with confidence
            with torch.no_grad():
                prediction, confidence = self.transformer_model.predict(
                    processed_data, return_confidence=True
                )

            return float(prediction[0]), float(confidence[0])

        except Exception as e:
            logger.warning(f"Transformer prediction failed: {str(e)}")
            return 0.5, 0.3  # Neutral prediction with low confidence

    def _get_ensemble_prediction(self, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Get ensemble prediction from multiple models"""
        predictions = []
        confidences = []

        for model in self.ensemble_models:
            try:
                pred = model.predict(market_data)
                conf = getattr(model, 'confidence', 0.5)
                predictions.append(pred)
                confidences.append(conf)
            except Exception as e:
                logger.warning(f"Ensemble model prediction failed: {str(e)}")
                continue

        if not predictions:
            return 0.5, 0.3

        # Weighted average of predictions
        ensemble_pred = np.mean(predictions)
        ensemble_conf = np.mean(confidences)

        return ensemble_pred, ensemble_conf

    def _calculate_smc_score(self, smc_analysis: Dict) -> float:
        """Calculate Smart Money Concepts score"""
        if not smc_analysis:
            return 0.5

        # Extract SMC components
        order_blocks = smc_analysis.get('order_blocks', [])
        fvg_score = smc_analysis.get('fair_value_gaps', {}).get('score', 0.5)
        liquidity_score = smc_analysis.get('liquidity_levels', {}).get('score', 0.5)
        structure_score = smc_analysis.get('market_structure', {}).get('score', 0.5)

        # Calculate composite SMC score
        smc_score = (fvg_score + liquidity_score + structure_score) / 3

        # Adjust for order block presence
        if order_blocks:
            smc_score = min(smc_score * 1.2, 1.0)

        return smc_score

    def _calculate_technical_score(self, technical_indicators: Dict) -> float:
        """Calculate technical analysis score"""
        if not technical_indicators:
            return 0.5

        scores = []

        # RSI score
        rsi = technical_indicators.get('rsi', 50)
        if rsi < 30:
            scores.append(0.8)  # Oversold - bullish
        elif rsi > 70:
            scores.append(0.2)  # Overbought - bearish
        else:
            scores.append(0.5)  # Neutral

        # MACD score
        macd = technical_indicators.get('macd', {})
        if macd.get('signal') == 'bullish':
            scores.append(0.8)
        elif macd.get('signal') == 'bearish':
            scores.append(0.2)
        else:
            scores.append(0.5)

        # Moving average score
        ma_signal = technical_indicators.get('ma_signal', 'neutral')
        if ma_signal == 'bullish':
            scores.append(0.8)
        elif ma_signal == 'bearish':
            scores.append(0.2)
        else:
            scores.append(0.5)

        return np.mean(scores) if scores else 0.5

    def _calculate_model_agreement(self, transformer_pred: float, ensemble_pred: float,
                                 smc_score: float, technical_score: float) -> float:
        """Calculate agreement between different prediction components"""
        predictions = [transformer_pred, ensemble_pred, smc_score, technical_score]

        # Calculate variance (lower variance = higher agreement)
        variance = np.var(predictions)

        # Convert variance to agreement score (0-1)
        agreement = max(0, 1 - (variance * 4))  # Scale factor of 4

        return agreement

    def _assess_market_conditions(self, market_data: pd.DataFrame) -> float:
        """Assess current market conditions for signal quality"""
        try:
            # Calculate recent volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1]

            # Calculate trend strength
            sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
            sma_50 = market_data['close'].rolling(50).mean().iloc[-1]
            current_price = market_data['close'].iloc[-1]

            trend_strength = abs(current_price - sma_20) / sma_20

            # Calculate volume trend
            volume_ma = market_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            volume_ratio = current_volume / volume_ma

            # Combine factors
            condition_score = 0.5  # Base score

            # Adjust for volatility (moderate volatility is good)
            if 0.01 <= volatility <= 0.03:
                condition_score += 0.2
            elif volatility > 0.05:
                condition_score -= 0.2

            # Adjust for trend strength
            if trend_strength > 0.02:
                condition_score += 0.2

            # Adjust for volume
            if volume_ratio > 1.2:
                condition_score += 0.1

            return max(0, min(1, condition_score))

        except Exception as e:
            logger.warning(f"Market condition assessment failed: {str(e)}")
            return 0.5

    def _calculate_volatility_adjustment(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility adjustment factor"""
        try:
            returns = market_data['close'].pct_change().dropna()
            current_volatility = returns.rolling(self.volatility_lookback).std().iloc[-1]

            # Historical volatility percentile
            historical_volatility = returns.rolling(100).std()
            volatility_percentile = (historical_volatility <= current_volatility).mean()

            # Adjustment factor (lower volatility = higher confidence)
            adjustment = 1 - (volatility_percentile * 0.3)

            return max(0.5, min(1.0, adjustment))

        except Exception as e:
            logger.warning(f"Volatility adjustment calculation failed: {str(e)}")
            return 0.8

    def _get_historical_accuracy(self, symbol: str) -> float:
        """Get historical accuracy for the symbol"""
        if not self.signal_history:
            return 0.7  # Default accuracy

        # Filter signals for this symbol
        symbol_signals = [s for s in self.signal_history[-self.historical_window:]
                         if s.symbol == symbol]

        if not symbol_signals:
            return 0.7

        # Calculate accuracy (this would be updated with actual trade results)
        # For now, use a placeholder calculation
        total_signals = len(symbol_signals)
        high_quality_signals = len([s for s in symbol_signals
                                   if s.quality in [SignalQuality.EXCELLENT, SignalQuality.GOOD]])

        return high_quality_signals / total_signals if total_signals > 0 else 0.7

    def _calculate_final_confidence(self, transformer_conf: float, ensemble_conf: float,
                                  model_agreement: float, market_condition: float,
                                  volatility_adj: float, historical_acc: float) -> float:
        """Calculate final confidence score"""
        # Weighted combination of all factors
        confidence = (
            transformer_conf * self.quality_weights['transformer_weight'] +
            ensemble_conf * self.quality_weights['ensemble_weight'] +
            model_agreement * 0.2 +
            market_condition * 0.15 +
            volatility_adj * 0.1 +
            historical_acc * 0.15
        )

        return max(0, min(1, confidence))

    def _determine_signal_type(self, transformer_pred: float, ensemble_pred: float,
                             confidence: float) -> SignalType:
        """Determine signal type based on predictions"""
        # Average prediction
        avg_pred = (transformer_pred + ensemble_pred) / 2

        if confidence > 0.8:
            if avg_pred > 0.7:
                return SignalType.STRONG_BUY
            elif avg_pred < 0.3:
                return SignalType.STRONG_SELL

        if avg_pred > 0.6:
            return SignalType.BUY
        elif avg_pred < 0.4:
            return SignalType.SELL
        else:
            return SignalType.HOLD

    def _determine_signal_quality(self, confidence: float) -> SignalQuality:
        """Determine signal quality based on confidence"""
        if confidence >= 0.9:
            return SignalQuality.EXCELLENT
        elif confidence >= 0.7:
            return SignalQuality.GOOD
        elif confidence >= 0.5:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR

    def _calculate_risk_metrics(self, signal_type: SignalType, current_price: float,
                              confidence: float, volatility_adj: float) -> Tuple[float, float, float, float]:
        """Calculate risk management metrics"""
        if signal_type == SignalType.HOLD:
            return None, None, None, None

        # Base risk percentages
        base_stop_loss = 0.02  # 2%
        base_take_profit = 0.04  # 4%

        # Adjust based on confidence and volatility
        confidence_multiplier = confidence
        volatility_multiplier = 2 - volatility_adj  # Higher volatility = wider stops

        stop_loss_pct = base_stop_loss * volatility_multiplier
        take_profit_pct = base_take_profit * confidence_multiplier

        # Calculate actual prices
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + take_profit_pct)
        else:  # SELL signals
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - take_profit_pct)

        # Position size based on confidence (higher confidence = larger position)
        max_position_size = 0.1  # 10% of portfolio
        position_size = max_position_size * confidence

        # Risk-reward ratio
        risk_reward_ratio = take_profit_pct / stop_loss_pct

        return stop_loss, take_profit, position_size, risk_reward_ratio

    def _prepare_transformer_input(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare market data for Transformer model input"""
        # This would use the TransformerPreprocessor
        # For now, return a placeholder
        return market_data[['open', 'high', 'low', 'close', 'volume']].values[-100:]

    def _update_signal_history(self, signal: TradingSignal):
        """Update signal history and performance metrics"""
        self.signal_history.append(signal)

        # Keep only recent signals for memory efficiency
        if len(self.signal_history) > self.historical_window * 2:
            self.signal_history = self.signal_history[-self.historical_window:]

        # Update performance metrics
        self.performance_metrics['total_signals'] += 1
        self.performance_metrics['quality_distribution'][signal.quality.value] += 1

        # Update average confidence
        total_confidence = sum(s.confidence for s in self.signal_history)
        self.performance_metrics['average_confidence'] = total_confidence / len(self.signal_history)

    def update_signal_outcome(self, signal_id: str, was_profitable: bool):
        """Update signal outcome for accuracy tracking"""
        # Find signal by timestamp or ID
        for signal in self.signal_history:
            if str(signal.timestamp) == signal_id:
                if was_profitable:
                    self.performance_metrics['correct_predictions'] += 1
                else:
                    if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                        self.performance_metrics['false_positives'] += 1
                    else:
                        self.performance_metrics['false_negatives'] += 1
                break

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        total_signals = self.performance_metrics['total_signals']
        if total_signals == 0:
            return {"message": "No signals generated yet"}

        accuracy = (self.performance_metrics['correct_predictions'] /
                   max(1, self.performance_metrics['correct_predictions'] +
                       self.performance_metrics['false_positives'] +
                       self.performance_metrics['false_negatives']))

        return {
            'total_signals': total_signals,
            'accuracy': accuracy,
            'average_confidence': self.performance_metrics['average_confidence'],
            'quality_distribution': self.performance_metrics['quality_distribution'],
            'recent_signals': len(self.signal_history),
            'false_positive_rate': self.performance_metrics['false_positives'] / max(1, total_signals),
            'false_negative_rate': self.performance_metrics['false_negatives'] / max(1, total_signals)
        }

    def optimize_thresholds(self, validation_data: List[Dict]) -> Dict[str, float]:
        """Optimize confidence thresholds based on validation data"""
        # This would implement threshold optimization
        # For now, return current thresholds
        return {
            'confidence_threshold': self.confidence_threshold,
            'quality_weights': self.quality_weights
        }
