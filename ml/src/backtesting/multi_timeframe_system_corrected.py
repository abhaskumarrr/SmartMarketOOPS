#!/usr/bin/env python3
"""
Multi-Timeframe Trading System (Corrected)

Proper timeframe hierarchy:
HIGHER TIMEFRAMES (Trend & Bias):
- 1D (Daily) - Primary trend direction
- 4H (4-hour) - Secondary trend confirmation
- 1H (1-hour) - Market structure & bias

LOWER TIMEFRAMES (Entry & Execution):
- 15M (15-minute) - Entry signals
- 5M (5-minute) - Precise execution

Key Principle: Higher timeframes have more influence on price moves.
Lower timeframes provide granular data but contain more noise.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketBias(Enum):
    """Market bias from higher timeframes"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class CorrectedTimeframeHierarchy:
    """Corrected timeframe hierarchy"""
    # HIGHER TIMEFRAMES (Trend & Bias)
    primary_trend: str = "1d"      # Primary trend (Daily)
    secondary_trend: str = "4h"    # Secondary trend (4-hour)
    market_structure: str = "1h"   # Market structure & bias (1-hour)

    # LOWER TIMEFRAMES (Entry & Execution)
    entry_signals: str = "15m"     # Entry signals (15-minute)
    execution: str = "5m"          # Precise execution (5-minute)

@dataclass
class MultiTimeframeConfig:
    """Multi-timeframe trading configuration"""
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-02-01"
    initial_capital: float = 10000.0

    # Timeframe hierarchy
    timeframes: CorrectedTimeframeHierarchy = field(default_factory=CorrectedTimeframeHierarchy)

    # Multi-timeframe rules
    require_all_htf_alignment: bool = True  # All higher TFs must align
    min_htf_agreement: int = 2              # Minimum higher TF agreement

    # Trading parameters
    max_position_size: float = 0.1
    confidence_threshold: float = 0.6
    max_daily_trades: int = 20

    # Risk management
    max_drawdown_limit: float = 0.15
    transaction_cost: float = 0.001


class MultiTimeframeAnalyzer:
    """
    Multi-timeframe analyzer with corrected hierarchy
    """

    def __init__(self, config: MultiTimeframeConfig):
        """Initialize analyzer"""
        self.config = config
        self.timeframe_data = {}
        self.higher_timeframes = [
            config.timeframes.primary_trend,    # 1d
            config.timeframes.secondary_trend,  # 4h
            config.timeframes.market_structure  # 1h
        ]
        self.lower_timeframes = [
            config.timeframes.entry_signals,    # 15m
            config.timeframes.execution         # 5m
        ]

    def load_all_timeframe_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Load data for all timeframes"""
        try:
            from production_real_data_backtester import RealDataFetcher
            data_fetcher = RealDataFetcher()

            all_timeframes = self.higher_timeframes + self.lower_timeframes

            for tf in all_timeframes:
                logger.info(f"Loading {tf} data...")

                # Higher timeframes need more historical data
                if tf in self.higher_timeframes:
                    # Extend history for higher timeframes
                    days_back = 180 if tf == "1d" else 90 if tf == "4h" else 30
                    extended_start = (pd.to_datetime(start_date) - timedelta(days=days_back)).strftime('%Y-%m-%d')
                    data = data_fetcher.fetch_real_data(symbol, extended_start, end_date, tf)
                else:
                    # Lower timeframes use standard range
                    data = data_fetcher.fetch_real_data(symbol, start_date, end_date, tf)

                if data is not None and len(data) > 50:
                    enhanced_data = self._create_timeframe_indicators(data, tf)
                    self.timeframe_data[tf] = enhanced_data
                    logger.info(f"✅ {tf}: {len(enhanced_data)} candles loaded")
                else:
                    logger.warning(f"❌ {tf}: Failed to load data")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to load timeframe data: {e}")
            return False

    def _create_timeframe_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Create indicators specific to timeframe role"""
        df = data.copy()

        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['hl2'] = (df['high'] + df['low']) / 2

        # Timeframe-specific parameters
        if timeframe == "1d":
            # Daily - Primary trend
            fast_ma, slow_ma, long_ma = 20, 50, 200
            rsi_period = 14
        elif timeframe == "4h":
            # 4-hour - Secondary trend
            fast_ma, slow_ma, long_ma = 15, 35, 100
            rsi_period = 14
        elif timeframe == "1h":
            # 1-hour - Market structure
            fast_ma, slow_ma, long_ma = 10, 25, 50
            rsi_period = 14
        elif timeframe == "15m":
            # 15-minute - Entry signals
            fast_ma, slow_ma = 8, 20
            rsi_period = 10
        else:  # 5m
            # 5-minute - Execution
            fast_ma, slow_ma = 5, 15
            rsi_period = 7

        # Moving averages
        df['sma_fast'] = df['close'].rolling(fast_ma).mean()
        df['sma_slow'] = df['close'].rolling(slow_ma).mean()
        df['ema_fast'] = df['close'].ewm(span=fast_ma).mean()

        # Long-term trend for higher timeframes
        if timeframe in ["1d", "4h", "1h"]:
            df['sma_long'] = df['close'].rolling(long_ma).mean()
            df['long_trend'] = (df['close'] > df['sma_long']).astype(int)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Trend strength
        df['ma_distance'] = (df['sma_fast'] - df['sma_slow']) / df['sma_slow']
        df['trend_strength'] = abs(df['ma_distance'])

        # Price position relative to MAs
        df['above_fast_ma'] = (df['close'] > df['sma_fast']).astype(int)
        df['above_slow_ma'] = (df['close'] > df['sma_slow']).astype(int)

        # Lower timeframe specific indicators
        if timeframe in ["15m", "5m"]:
            # Bollinger Bands for entry precision
            bb_period = 20
            df['bb_sma'] = df['close'].rolling(bb_period).mean()
            df['bb_std'] = df['close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_sma'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_sma'] - (df['bb_std'] * 2)

            # Short-term momentum
            df['momentum_3'] = df['close'].pct_change(3)
            df['momentum_5'] = df['close'].pct_change(5)

        return df.dropna()

    def analyze_higher_timeframe_consensus(self, current_time: datetime) -> Tuple[MarketBias, float]:
        """Analyze all higher timeframes for consensus bias"""
        try:
            htf_signals = {}

            # Analyze each higher timeframe
            for tf in self.higher_timeframes:
                data = self.timeframe_data[tf]
                current_candle = self._get_current_candle(data, current_time)

                if current_candle is None:
                    continue

                # Calculate bias for this timeframe
                bias_score = 0

                # 1. Moving average alignment
                if current_candle['close'] > current_candle['sma_fast'] > current_candle['sma_slow']:
                    bias_score += 2  # Strong bullish
                elif current_candle['close'] > current_candle['sma_fast']:
                    bias_score += 1  # Moderate bullish
                elif current_candle['close'] < current_candle['sma_fast'] < current_candle['sma_slow']:
                    bias_score -= 2  # Strong bearish
                elif current_candle['close'] < current_candle['sma_fast']:
                    bias_score -= 1  # Moderate bearish

                # 2. Long-term trend (if available)
                if 'long_trend' in current_candle:
                    if current_candle['long_trend'] == 1:
                        bias_score += 1
                    else:
                        bias_score -= 1

                # 3. RSI momentum
                rsi = current_candle['rsi']
                if rsi > 60:
                    bias_score += 1
                elif rsi < 40:
                    bias_score -= 1

                # 4. Trend strength
                strength = current_candle['trend_strength']
                if strength > 0.02:  # Strong trend
                    bias_score = bias_score * 1.5  # Amplify signal

                htf_signals[tf] = bias_score
                logger.debug(f"{tf} bias score: {bias_score}")

            # Calculate consensus
            if not htf_signals:
                return MarketBias.NEUTRAL, 0.0

            # Weight timeframes by importance (Daily > 4H > 1H)
            weights = {"1d": 3.0, "4h": 2.0, "1h": 1.5}

            weighted_score = 0
            total_weight = 0

            for tf, score in htf_signals.items():
                weight = weights.get(tf, 1.0)
                weighted_score += score * weight
                total_weight += weight

            consensus_score = weighted_score / total_weight if total_weight > 0 else 0

            # Determine bias strength
            if consensus_score >= 2.5:
                bias = MarketBias.STRONG_BULLISH
            elif consensus_score >= 1.0:
                bias = MarketBias.BULLISH
            elif consensus_score <= -2.5:
                bias = MarketBias.STRONG_BEARISH
            elif consensus_score <= -1.0:
                bias = MarketBias.BEARISH
            else:
                bias = MarketBias.NEUTRAL

            # Calculate confidence based on agreement
            agreement_count = sum(1 for score in htf_signals.values() if abs(score) >= 1)
            confidence = min(agreement_count / len(htf_signals), 1.0)

            return bias, confidence

        except Exception as e:
            logger.error(f"Higher timeframe consensus error: {e}")
            return MarketBias.NEUTRAL, 0.0

    def analyze_lower_timeframe_entry(self, current_time: datetime, htf_bias: MarketBias, htf_confidence: float) -> Dict[str, Any]:
        """Analyze lower timeframes for precise entry timing"""
        try:
            # Only trade if higher timeframes show clear bias
            if htf_bias == MarketBias.NEUTRAL or htf_confidence < 0.5:
                return {'signal': 'hold', 'confidence': 0.0, 'method': 'no_htf_consensus'}

            # Get lower timeframe data
            entry_data = self.timeframe_data[self.config.timeframes.entry_signals]  # 15m
            exec_data = self.timeframe_data[self.config.timeframes.execution]       # 5m

            entry_candle = self._get_current_candle(entry_data, current_time)
            exec_candle = self._get_current_candle(exec_data, current_time)

            if entry_candle is None or exec_candle is None:
                return {'signal': 'hold', 'confidence': 0.0, 'method': 'no_ltf_data'}

            entry_signals = []
            confidence = 0.3

            # Determine trade direction from higher timeframe bias
            if htf_bias in [MarketBias.BULLISH, MarketBias.STRONG_BULLISH]:
                target_direction = 'buy'

                # 15m entry signals for bullish bias
                if entry_candle['close'] > entry_candle['ema_fast']:
                    entry_signals.append('buy')
                    confidence += 0.2

                if entry_candle['rsi'] > 45 and entry_candle['rsi'] < 70:
                    entry_signals.append('buy')
                    confidence += 0.1

                # 5m execution timing
                if 'momentum_3' in exec_candle and exec_candle['momentum_3'] > 0.0005:
                    entry_signals.append('buy')
                    confidence += 0.2

                # Bollinger Band entry (buy dips)
                if 'bb_lower' in exec_candle:
                    bb_position = (exec_candle['close'] - exec_candle['bb_lower']) / (exec_candle['bb_upper'] - exec_candle['bb_lower'])
                    if bb_position < 0.3:  # Near lower band
                        entry_signals.append('buy')
                        confidence += 0.2

            elif htf_bias in [MarketBias.BEARISH, MarketBias.STRONG_BEARISH]:
                target_direction = 'sell'

                # 15m entry signals for bearish bias
                if entry_candle['close'] < entry_candle['ema_fast']:
                    entry_signals.append('sell')
                    confidence += 0.2

                if entry_candle['rsi'] < 55 and entry_candle['rsi'] > 30:
                    entry_signals.append('sell')
                    confidence += 0.1

                # 5m execution timing
                if 'momentum_3' in exec_candle and exec_candle['momentum_3'] < -0.0005:
                    entry_signals.append('sell')
                    confidence += 0.2

                # Bollinger Band entry (sell rallies)
                if 'bb_upper' in exec_candle:
                    bb_position = (exec_candle['close'] - exec_candle['bb_lower']) / (exec_candle['bb_upper'] - exec_candle['bb_lower'])
                    if bb_position > 0.7:  # Near upper band
                        entry_signals.append('sell')
                        confidence += 0.2

            else:
                return {'signal': 'hold', 'confidence': 0.0, 'method': 'neutral_htf_bias'}

            # Count signals in target direction
            target_signals = entry_signals.count(target_direction)

            # Boost confidence based on higher timeframe strength
            if htf_bias in [MarketBias.STRONG_BULLISH, MarketBias.STRONG_BEARISH]:
                confidence *= 1.3  # 30% boost for strong HTF bias

            # Require minimum signal alignment
            if target_signals >= 2:
                return {
                    'signal': target_direction,
                    'confidence': min(confidence * htf_confidence, 0.95),
                    'method': 'multi_timeframe_aligned',
                    'htf_bias': htf_bias.value,
                    'htf_confidence': htf_confidence,
                    'ltf_signals': target_signals,
                    'total_signals': len(entry_signals)
                }
            else:
                return {
                    'signal': 'hold',
                    'confidence': confidence,
                    'method': 'insufficient_ltf_signals',
                    'htf_bias': htf_bias.value,
                    'ltf_signals': target_signals
                }

        except Exception as e:
            logger.error(f"Lower timeframe entry error: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'method': 'error'}

    def _get_current_candle(self, data: pd.DataFrame, current_time: datetime) -> Optional[pd.Series]:
        """Get current candle for given time"""
        try:
            mask = data['timestamp'] <= current_time
            if mask.any():
                return data[mask].iloc[-1]
            return None
        except Exception:
            return None


def run_multi_timeframe_optimization():
    """Run multi-timeframe trading optimization"""
    print("🔄 MULTI-TIMEFRAME TRADING SYSTEM")
    print("=" * 50)
    print("Corrected timeframe hierarchy:")
    print("📈 HIGHER TIMEFRAMES (Trend & Bias):")
    print("   • 1D (Daily) - Primary trend direction")
    print("   • 4H (4-hour) - Secondary trend confirmation")
    print("   • 1H (1-hour) - Market structure & bias")
    print("📉 LOWER TIMEFRAMES (Entry & Execution):")
    print("   • 15M (15-minute) - Entry signals")
    print("   • 5M (5-minute) - Precise execution")

    try:
        config = MultiTimeframeConfig()
        analyzer = MultiTimeframeAnalyzer(config)

        # Load all timeframe data
        print(f"\n📡 Loading multi-timeframe data...")
        success = analyzer.load_all_timeframe_data(
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date
        )

        if not success:
            print("❌ Failed to load multi-timeframe data")
            return None

        print("✅ All timeframes loaded successfully")

        # Run multi-timeframe backtest
        print(f"\n💰 Running multi-timeframe backtest...")
        result = run_multi_timeframe_backtest(analyzer, config)

        if result:
            print(f"\n🎯 MULTI-TIMEFRAME RESULTS:")
            print("=" * 30)
            print(f"Total Trades: {result['total_trades']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"HTF Aligned Trades: {result['htf_aligned_trades']}")
            print(f"Average HTF Confidence: {result['avg_htf_confidence']:.2%}")

            # Compare with baseline
            baseline_trades = 4
            improvement = result['total_trades'] / baseline_trades
            print(f"\n📊 vs Baseline:")
            print(f"Trade frequency: {improvement:.1f}x improvement")

            if result['total_trades'] >= 20:
                print("✅ EXCELLENT multi-timeframe performance!")
            elif result['total_trades'] >= 10:
                print("⚠️  Good multi-timeframe improvement")
            else:
                print("❌ Multi-timeframe needs refinement")

            return result

        else:
            print("❌ Multi-timeframe backtest failed")
            return None

    except Exception as e:
        print(f"❌ Multi-timeframe optimization error: {e}")
        return None


def run_multi_timeframe_backtest(analyzer: MultiTimeframeAnalyzer, config: MultiTimeframeConfig) -> Dict[str, Any]:
    """Run multi-timeframe backtest"""
    try:
        # Use 15m timeframe as base for iteration
        base_data = analyzer.timeframe_data[config.timeframes.entry_signals]

        # Trading state
        capital = config.initial_capital
        position = 0.0
        trades = []
        equity_curve = []

        # Multi-timeframe tracking
        htf_aligned_trades = 0
        htf_confidences = []
        daily_trades = 0
        last_trade_date = None

        for i in range(50, len(base_data)):
            current_row = base_data.iloc[i]
            current_time = current_row['timestamp']
            current_price = current_row['close']
            current_date = current_time.date()

            # Reset daily counter
            if last_trade_date != current_date:
                daily_trades = 0
                last_trade_date = current_date

            if daily_trades >= config.max_daily_trades:
                continue

            # 1. Analyze higher timeframe consensus
            htf_bias, htf_confidence = analyzer.analyze_higher_timeframe_consensus(current_time)

            # 2. Analyze lower timeframe entry
            ltf_signal = analyzer.analyze_lower_timeframe_entry(current_time, htf_bias, htf_confidence)

            signal = ltf_signal['signal']
            confidence = ltf_signal['confidence']

            # Execute trades
            if (signal == 'buy' and confidence >= config.confidence_threshold and position <= 0):
                # Buy signal
                if position < 0:  # Close short
                    cost = abs(position) * current_price * (1 + config.transaction_cost)
                    capital -= cost
                    position = 0
                    daily_trades += 1

                # Open long
                position_size = config.max_position_size * confidence
                position_value = capital * position_size
                shares = position_value / current_price
                cost = shares * current_price * (1 + config.transaction_cost)

                if cost <= capital:
                    capital -= cost
                    position = shares
                    daily_trades += 1
                    htf_aligned_trades += 1
                    htf_confidences.append(htf_confidence)

                    trades.append({
                        'timestamp': current_time,
                        'action': 'buy',
                        'price': current_price,
                        'confidence': confidence,
                        'htf_bias': htf_bias.value,
                        'htf_confidence': htf_confidence,
                        'method': ltf_signal['method']
                    })

            elif (signal == 'sell' and confidence >= config.confidence_threshold and position >= 0):
                # Sell signal
                if position > 0:  # Close long
                    proceeds = position * current_price * (1 - config.transaction_cost)
                    capital += proceeds
                    position = 0
                    daily_trades += 1
                    htf_aligned_trades += 1
                    htf_confidences.append(htf_confidence)

                    trades.append({
                        'timestamp': current_time,
                        'action': 'sell',
                        'price': current_price,
                        'confidence': confidence,
                        'htf_bias': htf_bias.value,
                        'htf_confidence': htf_confidence,
                        'method': ltf_signal['method']
                    })

            # Update equity curve
            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)

        # Final calculations
        final_price = base_data['close'].iloc[-1]
        if position > 0:
            final_capital = capital + (position * final_price * (1 - config.transaction_cost))
        else:
            final_capital = capital

        total_return = (final_capital - config.initial_capital) / config.initial_capital

        # Calculate metrics
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365) if returns.std() > 0 else 0

            rolling_max = pd.Series(equity_curve).expanding().max()
            drawdown = (pd.Series(equity_curve) - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        return {
            'total_trades': len(trades),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': final_capital,
            'htf_aligned_trades': htf_aligned_trades,
            'avg_htf_confidence': np.mean(htf_confidences) if htf_confidences else 0,
            'trades': trades
        }

    except Exception as e:
        logger.error(f"Multi-timeframe backtest error: {e}")
        return None


if __name__ == "__main__":
    run_multi_timeframe_optimization()
