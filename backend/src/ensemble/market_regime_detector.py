"""
Market Regime Detection System for Enhanced Signal Quality
Implements Subtask 25.3: Market Regime Detection System
Provides comprehensive market regime classification for signal filtering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT_BULLISH = "breakout_bullish"
    BREAKOUT_BEARISH = "breakout_bearish"
    CONSOLIDATION = "consolidation"


@dataclass
class RegimeAnalysis:
    """Container for regime analysis results"""
    current_regime: MarketRegime
    regime_strength: float  # 0-1, how strong the regime signal is
    regime_duration: int    # How long current regime has persisted
    transition_probability: float  # Probability of regime change
    confidence: float       # Confidence in regime classification
    components: Dict[str, float]  # Individual indicator values


class MarketRegimeDetector:
    """
    Comprehensive market regime detection system
    Uses multiple indicators to classify market conditions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize market regime detector

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Indicator parameters
        self.adx_period = self.config.get('adx_period', 14)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        self.atr_period = self.config.get('atr_period', 14)
        self.trend_period = self.config.get('trend_period', 50)
        self.volume_period = self.config.get('volume_period', 20)

        # Regime thresholds
        self.trending_threshold = self.config.get('trending_threshold', 25)
        self.volatile_threshold = self.config.get('volatile_threshold', 0.02)
        self.ranging_threshold = self.config.get('ranging_threshold', 0.5)

        # Regime history for persistence analysis
        self.regime_history: List[Tuple[pd.Timestamp, MarketRegime]] = []
        self.max_history = self.config.get('max_history', 1000)

        logger.info("Market Regime Detector initialized")

    def detect_regime(self, data: np.ndarray, timestamps: Optional[List[pd.Timestamp]] = None) -> RegimeAnalysis:
        """
        Detect current market regime

        Args:
            data: OHLCV data array [n_periods, 5] or [n_periods, 6] with volume
            timestamps: Optional timestamps for each data point

        Returns:
            RegimeAnalysis object with detailed regime information
        """
        if len(data) < max(self.adx_period, self.bb_period, self.trend_period) + 10:
            return RegimeAnalysis(
                current_regime=MarketRegime.RANGING,
                regime_strength=0.5,
                regime_duration=0,
                transition_probability=0.5,
                confidence=0.0,
                components={}
            )

        # Convert to DataFrame for easier manipulation
        if data.shape[1] >= 5:
            df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'][:data.shape[1]])
        else:
            df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
            df['volume'] = 1000  # Default volume if not provided

        if timestamps and len(timestamps) == len(df):
            df.index = timestamps

        # Calculate all indicators
        indicators = self._calculate_indicators(df)

        # Classify regime based on indicators
        regime_scores = self._calculate_regime_scores(indicators)

        # Determine primary regime
        primary_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
        regime_strength = regime_scores[primary_regime]

        # Calculate regime persistence
        current_time = timestamps[-1] if timestamps else pd.Timestamp.now()
        regime_duration = self._calculate_regime_duration(primary_regime, current_time)

        # Calculate transition probability
        transition_prob = self._calculate_transition_probability(indicators, primary_regime)

        # Calculate overall confidence
        confidence = self._calculate_regime_confidence(regime_scores, indicators)

        # Update regime history
        self._update_regime_history(primary_regime, current_time)

        return RegimeAnalysis(
            current_regime=primary_regime,
            regime_strength=regime_strength,
            regime_duration=regime_duration,
            transition_probability=transition_prob,
            confidence=confidence,
            components=indicators
        )

    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators for regime detection"""
        indicators = {}

        # ADX for trend strength
        indicators['adx'] = self._calculate_adx(df)

        # Bollinger Band width for volatility
        indicators['bb_width'] = self._calculate_bb_width(df)

        # ATR for volatility
        indicators['atr'] = self._calculate_atr(df)
        indicators['atr_normalized'] = indicators['atr'] / df['close'].iloc[-1]

        # Trend indicators
        indicators['trend_slope'] = self._calculate_trend_slope(df)
        indicators['price_position'] = self._calculate_price_position(df)

        # Volume indicators
        indicators['volume_trend'] = self._calculate_volume_trend(df)
        indicators['volume_spike'] = self._calculate_volume_spike(df)

        # Momentum indicators
        indicators['momentum'] = self._calculate_momentum(df)
        indicators['rsi'] = self._calculate_rsi(df)

        # Volatility clustering
        indicators['volatility_clustering'] = self._calculate_volatility_clustering(df)

        return indicators

    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """Calculate Average Directional Index"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        if len(close) < self.adx_period + 1:
            return 0.0

        # Calculate True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # Calculate Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                          np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                           np.maximum(low[:-1] - low[1:], 0), 0)

        # Smooth the values
        atr = self._smooth_series(tr, self.adx_period)
        di_plus = 100 * self._smooth_series(dm_plus, self.adx_period) / atr
        di_minus = 100 * self._smooth_series(dm_minus, self.adx_period) / atr

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = self._smooth_series(dx, self.adx_period)

        return adx[-1] if len(adx) > 0 else 0.0

    def _calculate_bb_width(self, df: pd.DataFrame) -> float:
        """Calculate Bollinger Band width as volatility measure"""
        close = df['close']

        if len(close) < self.bb_period:
            return 0.0

        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()

        bb_width = (std * self.bb_std * 2) / sma
        return bb_width.iloc[-1] if not bb_width.empty else 0.0

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        if len(close) < 2:
            return 0.0

        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(self.atr_period).mean()

        return atr.iloc[-1] if not atr.empty else 0.0

    def _calculate_trend_slope(self, df: pd.DataFrame) -> float:
        """Calculate trend slope using linear regression"""
        close = df['close'].values

        if len(close) < self.trend_period:
            return 0.0

        recent_prices = close[-self.trend_period:]
        x = np.arange(len(recent_prices))

        # Linear regression
        slope = np.polyfit(x, recent_prices, 1)[0]

        # Normalize by price level
        normalized_slope = slope / recent_prices[-1]

        return normalized_slope

    def _calculate_price_position(self, df: pd.DataFrame) -> float:
        """Calculate price position relative to recent range"""
        close = df['close']
        high = df['high']
        low = df['low']

        if len(close) < self.trend_period:
            return 0.5

        recent_high = high[-self.trend_period:].max()
        recent_low = low[-self.trend_period:].min()
        current_price = close.iloc[-1]

        if recent_high == recent_low:
            return 0.5

        position = (current_price - recent_low) / (recent_high - recent_low)
        return position

    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate volume trend"""
        volume = df['volume']

        if len(volume) < self.volume_period:
            return 0.0

        recent_volume = volume[-self.volume_period:].values
        x = np.arange(len(recent_volume))

        slope = np.polyfit(x, recent_volume, 1)[0]
        normalized_slope = slope / recent_volume[-1] if recent_volume[-1] > 0 else 0

        return normalized_slope

    def _calculate_volume_spike(self, df: pd.DataFrame) -> float:
        """Detect volume spikes"""
        volume = df['volume']

        if len(volume) < self.volume_period:
            return 0.0

        avg_volume = volume[-self.volume_period:-1].mean()
        current_volume = volume.iloc[-1]

        spike_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        return min(spike_ratio, 5.0)  # Cap at 5x

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate price momentum"""
        close = df['close']

        if len(close) < 10:
            return 0.0

        momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
        return momentum

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        close = df['close']

        if len(close) < period + 1:
            return 50.0

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50.0

    def _calculate_volatility_clustering(self, df: pd.DataFrame) -> float:
        """Detect volatility clustering"""
        close = df['close']

        if len(close) < 20:
            return 0.0

        returns = close.pct_change().dropna()
        volatility = returns.rolling(5).std()

        # Check if recent volatility is higher than average
        recent_vol = volatility[-5:].mean()
        avg_vol = volatility[-20:-5].mean()

        clustering_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
        return min(clustering_ratio, 3.0)  # Cap at 3x

    def _smooth_series(self, series: np.ndarray, period: int) -> np.ndarray:
        """Apply exponential smoothing to a series"""
        if len(series) < period:
            return series

        alpha = 2.0 / (period + 1)
        smoothed = np.zeros_like(series)
        smoothed[0] = series[0]

        for i in range(1, len(series)):
            smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i-1]

        return smoothed

    def _calculate_regime_scores(self, indicators: Dict[str, float]) -> Dict[MarketRegime, float]:
        """Calculate scores for each regime based on indicators"""
        scores = {regime: 0.0 for regime in MarketRegime}

        adx = indicators.get('adx', 0)
        bb_width = indicators.get('bb_width', 0)
        trend_slope = indicators.get('trend_slope', 0)
        atr_normalized = indicators.get('atr_normalized', 0)
        momentum = indicators.get('momentum', 0)
        price_position = indicators.get('price_position', 0.5)
        volume_spike = indicators.get('volume_spike', 1)

        # Trending regimes
        if adx > self.trending_threshold:
            if trend_slope > 0.001 and momentum > 0:
                scores[MarketRegime.TRENDING_BULLISH] = adx / 100 * (1 + abs(trend_slope) * 100)
            elif trend_slope < -0.001 and momentum < 0:
                scores[MarketRegime.TRENDING_BEARISH] = adx / 100 * (1 + abs(trend_slope) * 100)

        # Ranging regime
        if adx < self.trending_threshold and bb_width < self.ranging_threshold:
            scores[MarketRegime.RANGING] = (1 - adx / 100) * (1 - bb_width)

        # Volatile regime
        if atr_normalized > self.volatile_threshold or bb_width > 0.05:
            scores[MarketRegime.VOLATILE] = min(atr_normalized * 50, 1.0) + min(bb_width * 20, 1.0)

        # Breakout regimes
        if volume_spike > 2.0 and abs(momentum) > 0.02:
            if momentum > 0 and price_position > 0.8:
                scores[MarketRegime.BREAKOUT_BULLISH] = volume_spike / 5 * abs(momentum) * 50
            elif momentum < 0 and price_position < 0.2:
                scores[MarketRegime.BREAKOUT_BEARISH] = volume_spike / 5 * abs(momentum) * 50

        # Consolidation regime
        if adx < 20 and bb_width < 0.02 and abs(momentum) < 0.01:
            scores[MarketRegime.CONSOLIDATION] = (1 - adx / 100) * (1 - bb_width * 50) * (1 - abs(momentum) * 100)

        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {regime: score / max_score for regime, score in scores.items()}

        return scores

    def _calculate_regime_duration(self, current_regime: MarketRegime, current_time: pd.Timestamp) -> int:
        """Calculate how long the current regime has persisted"""
        if not self.regime_history:
            return 0

        duration = 0
        for timestamp, regime in reversed(self.regime_history):
            if regime == current_regime:
                duration += 1
            else:
                break

        return duration

    def _calculate_transition_probability(self, indicators: Dict[str, float], current_regime: MarketRegime) -> float:
        """Calculate probability of regime transition"""
        # Simple heuristic based on indicator extremes
        adx = indicators.get('adx', 0)
        bb_width = indicators.get('bb_width', 0)
        momentum = indicators.get('momentum', 0)

        # High transition probability if indicators are at extremes
        transition_signals = 0

        if adx > 50 or adx < 15:  # Very strong trend or very weak trend
            transition_signals += 1

        if bb_width > 0.08 or bb_width < 0.01:  # Very high or very low volatility
            transition_signals += 1

        if abs(momentum) > 0.05:  # Strong momentum
            transition_signals += 1

        # Normalize to probability
        transition_prob = min(transition_signals / 3.0, 1.0)

        return transition_prob

    def _calculate_regime_confidence(self, regime_scores: Dict[MarketRegime, float], indicators: Dict[str, float]) -> float:
        """Calculate confidence in regime classification"""
        scores = list(regime_scores.values())

        if not scores:
            return 0.0

        # Confidence based on separation between top scores
        sorted_scores = sorted(scores, reverse=True)

        if len(sorted_scores) >= 2:
            confidence = sorted_scores[0] - sorted_scores[1]
        else:
            confidence = sorted_scores[0]

        # Adjust confidence based on data quality
        data_quality = min(1.0, len(indicators) / 10.0)  # Assume 10 indicators is full quality

        return confidence * data_quality

    def _update_regime_history(self, regime: MarketRegime, timestamp: pd.Timestamp):
        """Update regime history"""
        self.regime_history.append((timestamp, regime))

        # Limit history size
        if len(self.regime_history) > self.max_history:
            self.regime_history = self.regime_history[-self.max_history:]

    def is_favorable_regime(self, regime: MarketRegime, strategy_type: str = "trend_following") -> bool:
        """
        Check if current regime is favorable for trading

        Args:
            regime: Current market regime
            strategy_type: Type of trading strategy

        Returns:
            True if regime is favorable for the strategy
        """
        if strategy_type == "trend_following":
            return regime in [MarketRegime.TRENDING_BULLISH, MarketRegime.TRENDING_BEARISH]
        elif strategy_type == "mean_reversion":
            return regime in [MarketRegime.RANGING, MarketRegime.CONSOLIDATION]
        elif strategy_type == "breakout":
            return regime in [MarketRegime.BREAKOUT_BULLISH, MarketRegime.BREAKOUT_BEARISH]
        elif strategy_type == "volatility":
            return regime == MarketRegime.VOLATILE
        else:
            return True  # Default: trade in all regimes

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime history"""
        if not self.regime_history:
            return {}

        regimes = [regime for _, regime in self.regime_history]

        # Count occurrences
        regime_counts = {}
        for regime in MarketRegime:
            regime_counts[regime.value] = regimes.count(regime)

        # Calculate average durations
        regime_durations = {}
        current_regime = None
        current_duration = 0
        durations = []

        for _, regime in self.regime_history:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    durations.append((current_regime, current_duration))
                current_regime = regime
                current_duration = 1

        # Add final duration
        if current_regime is not None:
            durations.append((current_regime, current_duration))

        # Calculate average durations by regime
        for regime in MarketRegime:
            regime_durations[regime.value] = np.mean([
                duration for reg, duration in durations if reg == regime
            ]) if any(reg == regime for reg, _ in durations) else 0

        return {
            'regime_counts': regime_counts,
            'regime_durations': regime_durations,
            'total_periods': len(self.regime_history),
            'current_regime': regimes[-1].value if regimes else None
        }
