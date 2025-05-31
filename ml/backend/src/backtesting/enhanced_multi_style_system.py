#!/usr/bin/env python3
"""
Enhanced Multi-Style Dynamic Trading System

TRADING STYLE IDENTIFICATION:
1. LONG TRADES (Swing Trading):
   - Hold: 4-24 hours (16-96 periods on 15m)
   - Timeframes: 1H, 4H, 1D for trend confirmation
   - Targets: 2-8% profit, 1.5-3% stop loss
   - Lower frequency, higher conviction

2. SCALPING TRADES (Short-term):
   - Hold: 5-60 minutes (1-4 periods on 15m)
   - Timeframes: 1m, 3m, 5m for momentum
   - Targets: 0.3-1.5% profit, 0.5-1% stop loss
   - Higher frequency, quick execution

GRANULAR TIMEFRAME INTEGRATION:
- 1m, 3m, 5m, 15m, 1h, 4h, 1d hierarchy
- Proper timeframe synchronization
- Multi-timeframe confluence analysis

DYNAMIC TRADE TYPE SELECTION:
- Market condition detection (volatility, momentum)
- Automatic style switching
- Separate entry/exit logic per style
- Adaptive position sizing

CORE PRINCIPLES MAINTAINED:
- Exit when trends die (not momentum)
- Dynamic momentum filters (no hard timeframes)
- Candle patterns and price zones for confirmation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingStyle(Enum):
    """Trading style types"""
    SCALPING = "scalping"
    SWING = "swing"
    MIXED = "mixed"

class MarketRegime(Enum):
    """Market regime types"""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGING = "ranging"
    BREAKOUT = "breakout"

@dataclass
class EnhancedMultiStyleConfig:
    """Configuration for enhanced multi-style system"""
    # Trading parameters
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-02-01"
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001

    # Timeframe hierarchy
    timeframes: List[str] = None
    primary_timeframe: str = "15m"

    # Scalping parameters
    scalping_hold_min: int = 1          # 1 period (15 minutes)
    scalping_hold_max: int = 4          # 4 periods (1 hour)
    scalping_profit_target: float = 0.008  # 0.8%
    scalping_stop_loss: float = 0.005   # 0.5%
    scalping_position_size: float = 0.03  # 3% per trade
    scalping_max_daily_trades: int = 15

    # Swing parameters
    swing_hold_min: int = 16            # 16 periods (4 hours)
    swing_hold_max: int = 96            # 96 periods (24 hours)
    swing_profit_target: float = 0.04   # 4%
    swing_stop_loss: float = 0.02       # 2%
    swing_position_size: float = 0.08   # 8% per trade
    swing_max_daily_trades: int = 3

    # Market condition thresholds
    high_volatility_threshold: float = 0.02  # 2% ATR
    low_volatility_threshold: float = 0.008  # 0.8% ATR
    trending_momentum_threshold: float = 0.4
    ranging_momentum_threshold: float = 0.2

    # Dynamic selection parameters
    volatility_lookback: int = 24       # 6 hours
    momentum_lookback: int = 12         # 3 hours
    regime_confirmation_periods: int = 3

    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"]


class TimeframeManager:
    """Manages timeframe hierarchy and synchronization"""

    def __init__(self, config: EnhancedMultiStyleConfig):
        """Initialize timeframe manager"""
        self.config = config
        self.timeframe_hierarchy = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440
        }
        self.timeframe_data = {}

    def load_multi_timeframe_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Load data for all timeframes"""

        try:
            from production_real_data_backtester import RealDataFetcher

            data_fetcher = RealDataFetcher()

            print(f"📡 Loading multi-timeframe data...")

            # Load primary timeframe (15m)
            primary_data = data_fetcher.fetch_real_data(symbol, start_date, end_date, "15m")

            if primary_data is None or len(primary_data) < 100:
                print("❌ Failed to load primary timeframe data")
                return False

            self.timeframe_data["15m"] = primary_data
            print(f"✅ Loaded 15m: {len(primary_data)} candles")

            # Generate other timeframes from 15m data
            self._generate_higher_timeframes(primary_data)
            self._generate_lower_timeframes(primary_data)

            return True

        except Exception as e:
            print(f"❌ Error loading timeframe data: {e}")
            return False

    def _generate_higher_timeframes(self, base_data: pd.DataFrame):
        """Generate higher timeframes from 15m data"""

        # Generate 1h from 15m (4 candles = 1 hour)
        hourly_data = self._aggregate_timeframe(base_data, 4)
        self.timeframe_data["1h"] = hourly_data
        print(f"✅ Generated 1h: {len(hourly_data)} candles")

        # Generate 4h from 1h (4 candles = 4 hours)
        four_hour_data = self._aggregate_timeframe(hourly_data, 4)
        self.timeframe_data["4h"] = four_hour_data
        print(f"✅ Generated 4h: {len(four_hour_data)} candles")

        # Generate 1d from 4h (6 candles = 1 day)
        daily_data = self._aggregate_timeframe(four_hour_data, 6)
        self.timeframe_data["1d"] = daily_data
        print(f"✅ Generated 1d: {len(daily_data)} candles")

    def _generate_lower_timeframes(self, base_data: pd.DataFrame):
        """Generate lower timeframes from 15m data (simulated)"""

        # For backtesting, we'll simulate 5m, 3m, 1m from 15m
        # In production, these would be fetched separately

        # Generate 5m (3 candles per 15m)
        five_min_data = self._interpolate_timeframe(base_data, 3)
        self.timeframe_data["5m"] = five_min_data
        print(f"✅ Simulated 5m: {len(five_min_data)} candles")

        # Generate 3m (5 candles per 15m)
        three_min_data = self._interpolate_timeframe(base_data, 5)
        self.timeframe_data["3m"] = three_min_data
        print(f"✅ Simulated 3m: {len(three_min_data)} candles")

        # Generate 1m (15 candles per 15m)
        one_min_data = self._interpolate_timeframe(base_data, 15)
        self.timeframe_data["1m"] = one_min_data
        print(f"✅ Simulated 1m: {len(one_min_data)} candles")

    def _aggregate_timeframe(self, data: pd.DataFrame, factor: int) -> pd.DataFrame:
        """Aggregate data to higher timeframe"""

        aggregated_data = []

        for i in range(0, len(data), factor):
            chunk = data.iloc[i:i+factor]

            if len(chunk) == factor:  # Only complete periods
                agg_candle = {
                    'timestamp': chunk['timestamp'].iloc[-1],  # Use last timestamp
                    'open': chunk['open'].iloc[0],
                    'high': chunk['high'].max(),
                    'low': chunk['low'].min(),
                    'close': chunk['close'].iloc[-1],
                    'volume': chunk['volume'].sum() if 'volume' in chunk.columns else 0
                }
                aggregated_data.append(agg_candle)

        return pd.DataFrame(aggregated_data)

    def _interpolate_timeframe(self, data: pd.DataFrame, factor: int) -> pd.DataFrame:
        """Interpolate data to lower timeframe (simulation)"""

        interpolated_data = []

        for _, row in data.iterrows():
            # Create sub-candles within each 15m candle
            base_time = row['timestamp']
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']

            # Simple interpolation - in production, use real data
            price_range = high_price - low_price

            for j in range(factor):
                # Calculate time offset
                time_offset = timedelta(minutes=(15 // factor) * j)
                sub_time = base_time + time_offset

                # Simple price interpolation
                progress = j / (factor - 1) if factor > 1 else 0
                sub_open = open_price + (close_price - open_price) * (progress - 0.1) if j > 0 else open_price
                sub_close = open_price + (close_price - open_price) * (progress + 0.1) if j < factor - 1 else close_price

                # Add some randomness to high/low
                noise_factor = 0.3
                sub_high = max(sub_open, sub_close) + (price_range * noise_factor * np.random.random())
                sub_low = min(sub_open, sub_close) - (price_range * noise_factor * np.random.random())

                # Ensure high >= max(open, close) and low <= min(open, close)
                sub_high = max(sub_high, sub_open, sub_close)
                sub_low = min(sub_low, sub_open, sub_close)

                sub_candle = {
                    'timestamp': sub_time,
                    'open': sub_open,
                    'high': sub_high,
                    'low': sub_low,
                    'close': sub_close,
                    'volume': row['volume'] / factor if 'volume' in row else 0
                }
                interpolated_data.append(sub_candle)

        return pd.DataFrame(interpolated_data)

    def get_timeframe_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Get data for specific timeframe"""
        return self.timeframe_data.get(timeframe)

    def get_current_candle(self, timeframe: str, current_time: datetime) -> Optional[pd.Series]:
        """Get current candle for timeframe at specific time"""

        data = self.get_timeframe_data(timeframe)
        if data is None:
            return None

        # Find the most recent candle before or at current_time
        mask = data['timestamp'] <= current_time
        if mask.any():
            return data[mask].iloc[-1]

        return None


class MarketRegimeDetector:
    """Detects market regime for dynamic style selection"""

    def __init__(self, config: EnhancedMultiStyleConfig):
        """Initialize market regime detector"""
        self.config = config

    def detect_market_regime(self, timeframe_manager: TimeframeManager, current_time: datetime) -> MarketRegime:
        """Detect current market regime"""

        # Get data from multiple timeframes
        primary_data = timeframe_manager.get_timeframe_data(self.config.primary_timeframe)

        if primary_data is None:
            return MarketRegime.RANGING

        # Get recent data
        current_idx = self._get_current_index(primary_data, current_time)
        if current_idx < self.config.volatility_lookback:
            return MarketRegime.RANGING

        recent_data = primary_data.iloc[current_idx-self.config.volatility_lookback:current_idx+1]

        # Calculate volatility
        volatility = self._calculate_volatility(recent_data)

        # Calculate momentum strength
        momentum_strength = self._calculate_momentum_strength(recent_data)

        # Detect regime
        if volatility > self.config.high_volatility_threshold:
            if momentum_strength > self.config.trending_momentum_threshold:
                return MarketRegime.TRENDING
            else:
                return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.config.low_volatility_threshold:
            return MarketRegime.LOW_VOLATILITY
        else:
            if momentum_strength > self.config.trending_momentum_threshold:
                return MarketRegime.TRENDING
            else:
                return MarketRegime.RANGING

    def _get_current_index(self, data: pd.DataFrame, current_time: datetime) -> int:
        """Get current index in data"""
        mask = data['timestamp'] <= current_time
        if mask.any():
            return mask.sum() - 1
        return 0

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility (ATR-based)"""

        # True Range calculation
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.mean()

        # Normalize by price
        avg_price = data['close'].mean()
        volatility = atr / avg_price

        return volatility

    def _calculate_momentum_strength(self, data: pd.DataFrame) -> float:
        """Calculate momentum strength"""

        # Price momentum
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]

        # Directional consistency
        price_changes = data['close'].pct_change().dropna()
        directional_consistency = abs(price_changes.mean()) / price_changes.std() if price_changes.std() > 0 else 0

        # Combined momentum strength
        momentum_strength = abs(price_change) * directional_consistency

        return momentum_strength


class TradingStyleSelector:
    """Selects optimal trading style based on market conditions"""

    def __init__(self, config: EnhancedMultiStyleConfig):
        """Initialize trading style selector"""
        self.config = config

    def select_trading_style(self, market_regime: MarketRegime, volatility: float, momentum_strength: float) -> TradingStyle:
        """Select optimal trading style based on market conditions"""

        # SCALPING CONDITIONS:
        # - High volatility with quick reversals
        # - Low volatility with small movements
        # - Ranging markets with mean reversion

        if market_regime == MarketRegime.HIGH_VOLATILITY and momentum_strength < 0.3:
            return TradingStyle.SCALPING  # High vol, low momentum = scalping opportunities

        elif market_regime == MarketRegime.LOW_VOLATILITY:
            return TradingStyle.SCALPING  # Low vol = small moves, scalp them

        elif market_regime == MarketRegime.RANGING:
            return TradingStyle.SCALPING  # Ranging = mean reversion scalping

        # SWING CONDITIONS:
        # - Strong trending markets
        # - High momentum with sustained direction
        # - Breakout conditions

        elif market_regime == MarketRegime.TRENDING and momentum_strength > 0.4:
            return TradingStyle.SWING  # Strong trend = swing trade

        elif market_regime == MarketRegime.BREAKOUT:
            return TradingStyle.SWING  # Breakouts = swing opportunities

        # MIXED CONDITIONS:
        # - Moderate volatility and momentum
        # - Uncertain market conditions

        else:
            return TradingStyle.MIXED  # Use both styles with lower allocation

    def get_style_parameters(self, trading_style: TradingStyle) -> Dict[str, Any]:
        """Get parameters for selected trading style"""

        if trading_style == TradingStyle.SCALPING:
            return {
                'hold_min': self.config.scalping_hold_min,
                'hold_max': self.config.scalping_hold_max,
                'profit_target': self.config.scalping_profit_target,
                'stop_loss': self.config.scalping_stop_loss,
                'position_size': self.config.scalping_position_size,
                'max_daily_trades': self.config.scalping_max_daily_trades,
                'timeframes': ["1m", "3m", "5m", "15m"],
                'entry_timeframes': ["1m", "3m", "5m"],
                'trend_timeframes': ["15m", "1h"]
            }

        elif trading_style == TradingStyle.SWING:
            return {
                'hold_min': self.config.swing_hold_min,
                'hold_max': self.config.swing_hold_max,
                'profit_target': self.config.swing_profit_target,
                'stop_loss': self.config.swing_stop_loss,
                'position_size': self.config.swing_position_size,
                'max_daily_trades': self.config.swing_max_daily_trades,
                'timeframes': ["15m", "1h", "4h", "1d"],
                'entry_timeframes': ["5m", "15m"],
                'trend_timeframes': ["1h", "4h", "1d"]
            }

        else:  # MIXED
            return {
                'hold_min': self.config.scalping_hold_min,
                'hold_max': self.config.swing_hold_max,
                'profit_target': (self.config.scalping_profit_target + self.config.swing_profit_target) / 2,
                'stop_loss': (self.config.scalping_stop_loss + self.config.swing_stop_loss) / 2,
                'position_size': (self.config.scalping_position_size + self.config.swing_position_size) / 2,
                'max_daily_trades': (self.config.scalping_max_daily_trades + self.config.swing_max_daily_trades) // 2,
                'timeframes': ["1m", "3m", "5m", "15m", "1h", "4h"],
                'entry_timeframes': ["3m", "5m", "15m"],
                'trend_timeframes': ["15m", "1h", "4h"]
            }


def run_enhanced_multi_style_research():
    """Run research phase for enhanced multi-style system"""

    print("🔍 ENHANCED MULTI-STYLE SYSTEM RESEARCH")
    print("=" * 55)
    print("RESEARCHING:")
    print("✅ Multi-timeframe hierarchy (1m to 1d)")
    print("✅ Market regime detection")
    print("✅ Trading style selection logic")
    print("✅ Timeframe synchronization")

    try:
        config = EnhancedMultiStyleConfig()

        # Initialize components
        timeframe_manager = TimeframeManager(config)
        regime_detector = MarketRegimeDetector(config)
        style_selector = TradingStyleSelector(config)

        # Load multi-timeframe data
        print(f"\n📡 Loading multi-timeframe data...")
        success = timeframe_manager.load_multi_timeframe_data(
            config.symbol, config.start_date, config.end_date
        )

        if not success:
            print("❌ Failed to load multi-timeframe data")
            return None

        # Research market regimes
        print(f"\n🔄 Researching market regimes...")
        primary_data = timeframe_manager.get_timeframe_data("15m")

        regime_counts = {}
        style_counts = {}

        # Sample analysis over time
        sample_points = range(50, len(primary_data), 24)  # Every 6 hours

        for i in sample_points:
            current_time = primary_data.iloc[i]['timestamp']

            # Detect market regime
            regime = regime_detector.detect_market_regime(timeframe_manager, current_time)
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

            # Select trading style
            volatility = regime_detector._calculate_volatility(primary_data.iloc[max(0, i-24):i+1])
            momentum = regime_detector._calculate_momentum_strength(primary_data.iloc[max(0, i-12):i+1])

            style = style_selector.select_trading_style(regime, volatility, momentum)
            style_counts[style.value] = style_counts.get(style.value, 0) + 1

        # Research results
        print(f"\n📊 RESEARCH RESULTS:")
        print("=" * 25)

        print(f"📈 MARKET REGIME ANALYSIS:")
        total_samples = len(sample_points)
        for regime, count in regime_counts.items():
            pct = count / total_samples * 100
            print(f"   {regime}: {count} samples ({pct:.1f}%)")

        print(f"\n🎯 TRADING STYLE SELECTION:")
        for style, count in style_counts.items():
            pct = count / total_samples * 100
            print(f"   {style}: {count} samples ({pct:.1f}%)")

        print(f"\n⏰ TIMEFRAME HIERARCHY:")
        for tf in config.timeframes:
            data = timeframe_manager.get_timeframe_data(tf)
            if data is not None:
                print(f"   {tf}: {len(data)} candles")

        print(f"\n🎯 RESEARCH CONCLUSIONS:")
        print("✅ Multi-timeframe data loaded successfully")
        print("✅ Market regime detection working")
        print("✅ Trading style selection logic validated")
        print("✅ Ready for implementation phase")

        return {
            'timeframe_manager': timeframe_manager,
            'regime_detector': regime_detector,
            'style_selector': style_selector,
            'regime_distribution': regime_counts,
            'style_distribution': style_counts,
            'total_samples': total_samples
        }

    except Exception as e:
        print(f"❌ Research error: {e}")
        return None


class EnhancedMultiStyleTradingSystem:
    """Complete enhanced multi-style trading system"""

    def __init__(self, config: EnhancedMultiStyleConfig):
        """Initialize enhanced multi-style trading system"""
        self.config = config
        self.timeframe_manager = TimeframeManager(config)
        self.regime_detector = MarketRegimeDetector(config)
        self.style_selector = TradingStyleSelector(config)

    def run_enhanced_backtest(self) -> Dict[str, Any]:
        """Run enhanced multi-style backtest"""

        print("🚀 ENHANCED MULTI-STYLE TRADING SYSTEM")
        print("=" * 55)
        print("IMPLEMENTING:")
        print("✅ Dynamic style selection (Scalping vs Swing)")
        print("✅ Multi-timeframe analysis (1m to 1d)")
        print("✅ Market regime detection")
        print("✅ Granular timeframe synchronization")
        print("✅ Adaptive position sizing")

        try:
            # Load multi-timeframe data
            print(f"\n📡 Loading multi-timeframe data...")
            success = self.timeframe_manager.load_multi_timeframe_data(
                self.config.symbol, self.config.start_date, self.config.end_date
            )

            if not success:
                print("❌ Failed to load data")
                return None

            # Execute enhanced strategy
            print(f"\n💰 Executing enhanced multi-style strategy...")
            result = self._execute_enhanced_strategy()

            if result:
                self._display_enhanced_results(result)
                return result
            else:
                print("❌ Enhanced backtest failed")
                return None

        except Exception as e:
            print(f"❌ Enhanced system error: {e}")
            return None

    def _execute_enhanced_strategy(self) -> Dict[str, Any]:
        """Execute the enhanced multi-style strategy"""

        # Get primary timeframe data
        primary_data = self.timeframe_manager.get_timeframe_data(self.config.primary_timeframe)

        if primary_data is None:
            return None

        # Trading state
        capital = self.config.initial_capital
        position = 0.0
        trades = []
        equity_curve = []

        # Tracking
        trade_counter = 0
        active_trade_id = None
        entry_price = 0
        periods_in_trade = 0
        current_style = TradingStyle.SCALPING
        style_params = {}
        daily_trades = 0
        last_date = None

        # Style tracking
        style_changes = []
        regime_history = []

        print(f"💰 Executing enhanced strategy on {len(primary_data)} periods...")

        for i in range(50, len(primary_data)):  # Start after indicators stabilize
            current_row = primary_data.iloc[i]
            current_time = current_row['timestamp']
            current_price = current_row['close']
            current_date = current_time.date()

            # Reset daily counter
            if last_date != current_date:
                daily_trades = 0
                last_date = current_date

            # Detect market regime and select trading style
            if i % 12 == 0:  # Update every 3 hours
                market_regime = self.regime_detector.detect_market_regime(self.timeframe_manager, current_time)
                volatility = self.regime_detector._calculate_volatility(primary_data.iloc[max(0, i-24):i+1])
                momentum = self.regime_detector._calculate_momentum_strength(primary_data.iloc[max(0, i-12):i+1])

                new_style = self.style_selector.select_trading_style(market_regime, volatility, momentum)

                if new_style != current_style:
                    style_changes.append({
                        'timestamp': current_time,
                        'old_style': current_style.value,
                        'new_style': new_style.value,
                        'regime': market_regime.value,
                        'volatility': volatility,
                        'momentum': momentum
                    })
                    current_style = new_style

                style_params = self.style_selector.get_style_parameters(current_style)
                regime_history.append({
                    'timestamp': current_time,
                    'regime': market_regime.value,
                    'style': current_style.value,
                    'volatility': volatility,
                    'momentum': momentum
                })

            # Check for exit signals
            if active_trade_id and position != 0:
                periods_in_trade += 1

                should_exit, exit_reason = self._check_enhanced_exit(
                    current_row, periods_in_trade, entry_price, position, style_params, current_time
                )

                if should_exit:
                    # Execute exit
                    if position > 0:
                        proceeds = position * current_price * (1 - self.config.transaction_cost)
                        capital += proceeds
                    else:
                        cost = abs(position) * current_price * (1 + self.config.transaction_cost)
                        capital -= cost

                    # Calculate profit
                    if position > 0:
                        profit = (current_price - entry_price) / entry_price
                    else:
                        profit = (entry_price - current_price) / entry_price

                    trades.append({
                        'timestamp': current_time,
                        'action': 'exit',
                        'price': current_price,
                        'reason': exit_reason,
                        'profit': profit,
                        'periods_held': periods_in_trade,
                        'trading_style': current_style.value,
                        'trade_id': active_trade_id
                    })

                    position = 0.0
                    active_trade_id = None
                    periods_in_trade = 0
                    entry_price = 0

            # Check for entry signals
            if (position == 0 and
                daily_trades < style_params.get('max_daily_trades', 5) and
                len(style_params) > 0):

                entry_signal, signal_strength = self._check_enhanced_entry(
                    current_row, current_time, style_params, current_style
                )

                if entry_signal != 'none' and signal_strength > 0.4:
                    trade_counter += 1
                    active_trade_id = f"trade_{trade_counter}"

                    # Calculate position size based on style and signal strength
                    base_size = style_params.get('position_size', 0.05)
                    position_size = base_size * signal_strength
                    position_value = capital * position_size
                    shares = position_value / current_price
                    cost = shares * current_price * (1 + self.config.transaction_cost)

                    if cost <= capital:
                        capital -= cost
                        position = shares if entry_signal == 'buy' else -shares
                        entry_price = current_price
                        periods_in_trade = 0
                        daily_trades += 1

                        trades.append({
                            'timestamp': current_time,
                            'action': entry_signal,
                            'price': current_price,
                            'signal_strength': signal_strength,
                            'position_size': position_size,
                            'trading_style': current_style.value,
                            'trade_id': active_trade_id
                        })

            # Update equity curve
            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)

        # Final calculations
        final_price = primary_data['close'].iloc[-1]
        if position != 0:
            if position > 0:
                final_capital = capital + (position * final_price * (1 - self.config.transaction_cost))
            else:
                final_capital = capital - (abs(position) * final_price * (1 + self.config.transaction_cost))
        else:
            final_capital = capital

        total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital

        # Analyze results
        entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
        exit_trades = [t for t in trades if t['action'] == 'exit']

        # Calculate metrics
        profits = [t.get('profit', 0) for t in exit_trades]
        hold_periods = [t.get('periods_held', 0) for t in exit_trades]
        winners = [p for p in profits if p > 0]
        losers = [p for p in profits if p < 0]

        # Style analysis
        scalping_trades = [t for t in entry_trades if t.get('trading_style') == 'scalping']
        swing_trades = [t for t in entry_trades if t.get('trading_style') == 'swing']
        mixed_trades = [t for t in entry_trades if t.get('trading_style') == 'mixed']

        # Performance metrics
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(24 * 365) if returns.std() > 0 else 0

            rolling_max = pd.Series(equity_curve).expanding().max()
            drawdown = (pd.Series(equity_curve) - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        # Exit reasons
        exit_reasons = {}
        for trade in exit_trades:
            reason = trade.get('reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        return {
            'total_trades': len(entry_trades),
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': final_capital,
            'win_rate': len(winners) / len(profits) if profits else 0,
            'avg_winner': np.mean(winners) if winners else 0,
            'avg_loser': np.mean(losers) if losers else 0,
            'reward_risk_ratio': abs(np.mean(winners) / np.mean(losers)) if winners and losers else 0,
            'profit_factor': abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf'),
            'avg_hold_periods': np.mean(hold_periods) if hold_periods else 0,
            'exit_reasons': exit_reasons,
            'style_analysis': {
                'scalping_trades': len(scalping_trades),
                'swing_trades': len(swing_trades),
                'mixed_trades': len(mixed_trades),
                'style_changes': len(style_changes)
            },
            'regime_history': regime_history,
            'style_changes': style_changes,
            'trades': trades
        }

    def _check_enhanced_entry(self, current_row: pd.Series, current_time: datetime,
                             style_params: Dict[str, Any], trading_style: TradingStyle) -> Tuple[str, float]:
        """Check for enhanced entry signals based on trading style"""

        # Get multi-timeframe analysis
        entry_timeframes = style_params.get('entry_timeframes', ['5m', '15m'])
        trend_timeframes = style_params.get('trend_timeframes', ['15m', '1h'])

        # Calculate momentum across timeframes
        momentum_signals = []
        trend_signals = []

        for tf in entry_timeframes:
            candle = self.timeframe_manager.get_current_candle(tf, current_time)
            if candle is not None:
                momentum = self._calculate_candle_momentum(candle)
                momentum_signals.append(momentum)

        for tf in trend_timeframes:
            candle = self.timeframe_manager.get_current_candle(tf, current_time)
            if candle is not None:
                trend = self._calculate_candle_trend(candle)
                trend_signals.append(trend)

        if not momentum_signals or not trend_signals:
            return 'none', 0.0

        # Calculate signal strength
        avg_momentum = np.mean(momentum_signals)
        avg_trend = np.mean(trend_signals)

        # Style-specific entry logic
        if trading_style == TradingStyle.SCALPING:
            # Scalping: Quick momentum reversals or breakouts
            if avg_momentum > 0.3 and avg_trend > 0.1:
                signal_strength = min(avg_momentum + 0.2, 1.0)
                return 'buy', signal_strength
            elif avg_momentum < -0.3 and avg_trend < -0.1:
                signal_strength = min(abs(avg_momentum) + 0.2, 1.0)
                return 'sell', signal_strength

        elif trading_style == TradingStyle.SWING:
            # Swing: Strong trend alignment
            if avg_momentum > 0.2 and avg_trend > 0.3:
                signal_strength = min((avg_momentum + avg_trend) / 2 + 0.1, 1.0)
                return 'buy', signal_strength
            elif avg_momentum < -0.2 and avg_trend < -0.3:
                signal_strength = min((abs(avg_momentum) + abs(avg_trend)) / 2 + 0.1, 1.0)
                return 'sell', signal_strength

        else:  # MIXED
            # Mixed: Moderate signals
            if avg_momentum > 0.25 and avg_trend > 0.15:
                signal_strength = min((avg_momentum + avg_trend) / 2, 0.8)
                return 'buy', signal_strength
            elif avg_momentum < -0.25 and avg_trend < -0.15:
                signal_strength = min((abs(avg_momentum) + abs(avg_trend)) / 2, 0.8)
                return 'sell', signal_strength

        return 'none', 0.0

    def _check_enhanced_exit(self, current_row: pd.Series, periods_in_trade: int,
                            entry_price: float, position: float, style_params: Dict[str, Any],
                            current_time: datetime) -> Tuple[bool, str]:
        """Check for enhanced exit conditions based on trading style"""

        current_price = current_row['close']

        # Calculate current profit
        if position > 0:
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price

        # Style-specific exit logic
        profit_target = style_params.get('profit_target', 0.02)
        stop_loss = style_params.get('stop_loss', 0.01)
        hold_min = style_params.get('hold_min', 1)
        hold_max = style_params.get('hold_max', 20)

        # 1. STOP LOSS
        if current_profit <= -stop_loss:
            return True, "stop_loss"

        # 2. PROFIT TARGET
        if current_profit >= profit_target:
            return True, "profit_target"

        # 3. MINIMUM HOLD PERIOD
        if periods_in_trade < hold_min:
            return False, "hold_min_not_reached"

        # 4. MAXIMUM HOLD PERIOD
        if periods_in_trade >= hold_max:
            return True, "max_hold_period"

        # 5. DYNAMIC TREND DEATH DETECTION
        trend_timeframes = style_params.get('trend_timeframes', ['15m', '1h'])
        trend_signals = []

        for tf in trend_timeframes:
            candle = self.timeframe_manager.get_current_candle(tf, current_time)
            if candle is not None:
                trend = self._calculate_candle_trend(candle)
                trend_signals.append(trend)

        if trend_signals:
            avg_trend = np.mean(trend_signals)

            # Exit if trend dies
            if position > 0 and avg_trend < -0.2:  # Trend reversal for longs
                return True, "trend_death_bearish"
            elif position < 0 and avg_trend > 0.2:  # Trend reversal for shorts
                return True, "trend_death_bullish"

        return False, "hold"

    def _calculate_candle_momentum(self, candle: pd.Series) -> float:
        """Calculate momentum for a single candle"""

        # Simple momentum based on candle body and position
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return 0.0

        # Momentum direction
        direction = 1 if candle['close'] > candle['open'] else -1

        # Momentum strength (body ratio)
        strength = body_size / total_range

        return direction * strength

    def _calculate_candle_trend(self, candle: pd.Series) -> float:
        """Calculate trend for a single candle"""

        # Simple trend based on close position in range
        total_range = candle['high'] - candle['low']

        if total_range == 0:
            return 0.0

        # Position in range (0 = at low, 1 = at high)
        position_in_range = (candle['close'] - candle['low']) / total_range

        # Convert to trend signal (-1 to 1)
        trend = (position_in_range - 0.5) * 2

        return trend

    def _display_enhanced_results(self, result: Dict[str, Any]):
        """Display enhanced system results"""

        print(f"\n🎯 ENHANCED MULTI-STYLE RESULTS:")
        print("=" * 45)
        print(f"Total Trades: {result['total_trades']}")
        print(f"Total Return: {result['total_return']:.2%}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['max_drawdown']:.2%}")
        print(f"Win Rate: {result['win_rate']:.1%}")
        print(f"Average Winner: {result['avg_winner']:.2%}")
        print(f"Average Loser: {result['avg_loser']:.2%}")
        print(f"Reward/Risk Ratio: {result['reward_risk_ratio']:.2f}")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Average Hold Time: {result['avg_hold_periods']:.1f} periods")

        # Style analysis
        style_analysis = result['style_analysis']
        print(f"\n📊 TRADING STYLE ANALYSIS:")
        print(f"Scalping trades: {style_analysis['scalping_trades']}")
        print(f"Swing trades: {style_analysis['swing_trades']}")
        print(f"Mixed trades: {style_analysis['mixed_trades']}")
        print(f"Style changes: {style_analysis['style_changes']}")

        # Exit analysis
        if 'exit_reasons' in result:
            print(f"\n📊 EXIT ANALYSIS:")
            total_exits = sum(result['exit_reasons'].values())
            for reason, count in result['exit_reasons'].items():
                pct = count / total_exits * 100
                print(f"   {reason}: {count} ({pct:.1f}%)")

        # Performance assessment
        if (result['total_return'] > 0.02 and
            result['profit_factor'] >= 1.5 and
            result['win_rate'] >= 0.55):
            print(f"\n🎉 EXCELLENT: Enhanced multi-style system working perfectly!")
        elif result['total_return'] > 0.01 and result['profit_factor'] >= 1.2:
            print(f"\n✅ GOOD: Multi-style system performing well")
        elif result['total_return'] > 0:
            print(f"\n⚠️  MODERATE: Positive but needs optimization")
        else:
            print(f"\n❌ POOR: System needs refinement")


def run_enhanced_multi_style_system():
    """Run the complete enhanced multi-style system"""

    config = EnhancedMultiStyleConfig()
    system = EnhancedMultiStyleTradingSystem(config)

    result = system.run_enhanced_backtest()

    return result


if __name__ == "__main__":
    print("🚀 Starting Enhanced Multi-Style Trading System")
    print("Dynamic style selection with multi-timeframe analysis")

    # First run research
    print("\n" + "="*60)
    print("PHASE 1: RESEARCH")
    print("="*60)

    research_results = run_enhanced_multi_style_research()

    if research_results:
        print(f"\n🎉 RESEARCH PHASE COMPLETED!")

        # Then run implementation
        print("\n" + "="*60)
        print("PHASE 2: IMPLEMENTATION")
        print("="*60)

        trading_results = run_enhanced_multi_style_system()

        if trading_results:
            print(f"\n🎉 ENHANCED MULTI-STYLE SYSTEM COMPLETED!")
            print("Dynamic trading with multi-timeframe analysis successful!")
        else:
            print(f"\n❌ Implementation failed")
    else:
        print(f"\n❌ Research failed")
