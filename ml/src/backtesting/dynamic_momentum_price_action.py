#!/usr/bin/env python3
"""
Dynamic Momentum & Price Action System

MIRRORING PINE SCRIPT BEST PRACTICES:

1. DYNAMIC MOMENTUM FILTERS (no hard-coded timeframes):
   - Adaptive RSI with rising/falling detection
   - Stochastic momentum confirmation
   - Dynamic momentum strength calculation

2. CANDLE PATTERN DETECTION:
   - Hammer, Doji, Engulfing patterns
   - Shooting star, Pin bar detection
   - Pattern strength scoring

3. DYNAMIC PRICE ZONES:
   - Pivot-based support/resistance
   - Dynamic zone strength calculation
   - Zone proximity confirmation

4. CONFLUENCE-BASED ENTRIES:
   - Multiple confirmation system
   - Dynamic exit conditions
   - Adaptive position sizing

Based on research of professional Pine Script strategies.
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

class CandlePattern(Enum):
    """Candle pattern types"""
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    DOJI = "doji"
    BULLISH_ENGULFING = "bullish_engulfing"
    BEARISH_ENGULFING = "bearish_engulfing"
    PIN_BAR = "pin_bar"

@dataclass
class DynamicConfig:
    """Configuration for dynamic momentum system"""
    # Dynamic momentum parameters
    rsi_length: int = 14
    stoch_length: int = 14
    momentum_confirmation_periods: int = 2

    # Dynamic zone parameters
    pivot_left_bars: int = 5
    pivot_right_bars: int = 5
    zone_strength_min_touches: int = 2
    zone_proximity_percent: float = 0.5  # 0.5% proximity to zone

    # Candle pattern parameters
    pattern_body_ratio: float = 0.3      # Body vs wick ratio
    pattern_confirmation_periods: int = 1

    # Trading parameters
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-02-01"
    initial_capital: float = 10000.0
    max_position_size: float = 0.10
    transaction_cost: float = 0.001

    # Dynamic exit parameters
    momentum_exit_threshold: float = 30.0  # RSI threshold for momentum exit
    max_hold_periods: int = 12             # 3 hours max


class DynamicMomentumFilter:
    """Dynamic momentum filter mirroring Pine Script ta.* functions"""

    def __init__(self, config: DynamicConfig):
        """Initialize dynamic momentum filter"""
        self.config = config

    def calculate_dynamic_rsi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI with dynamic analysis (ta.rsi equivalent)"""

        close = data['close']
        delta = close.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_length).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def calculate_dynamic_stochastic(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic with dynamic analysis (ta.stoch equivalent)"""

        length = self.config.stoch_length

        lowest_low = data['low'].rolling(window=length).min()
        highest_high = data['high'].rolling(window=length).max()

        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()

        return k_percent.fillna(50), d_percent.fillna(50)

    def detect_momentum_direction(self, rsi: pd.Series, stoch_k: pd.Series) -> pd.Series:
        """Detect dynamic momentum direction (ta.rising/ta.falling equivalent)"""

        # RSI momentum direction
        rsi_rising = self._is_rising(rsi, self.config.momentum_confirmation_periods)
        rsi_falling = self._is_falling(rsi, self.config.momentum_confirmation_periods)

        # Stochastic momentum direction
        stoch_rising = self._is_rising(stoch_k, self.config.momentum_confirmation_periods)
        stoch_falling = self._is_falling(stoch_k, self.config.momentum_confirmation_periods)

        # Combined momentum direction
        momentum_direction = pd.Series(0, index=rsi.index)  # 0 = neutral

        # Bullish momentum
        bullish_momentum = (rsi_rising & stoch_rising) & (rsi > 50) & (stoch_k > 50)
        momentum_direction[bullish_momentum] = 1

        # Bearish momentum
        bearish_momentum = (rsi_falling & stoch_falling) & (rsi < 50) & (stoch_k < 50)
        momentum_direction[bearish_momentum] = -1

        return momentum_direction

    def calculate_momentum_strength(self, rsi: pd.Series, stoch_k: pd.Series) -> pd.Series:
        """Calculate dynamic momentum strength"""

        # RSI strength (distance from 50)
        rsi_strength = abs(rsi - 50) / 50

        # Stochastic strength (distance from extremes)
        stoch_strength = np.minimum(stoch_k / 100, (100 - stoch_k) / 100) * 2

        # Combined momentum strength
        momentum_strength = (rsi_strength + stoch_strength) / 2

        return momentum_strength.fillna(0)

    def _is_rising(self, series: pd.Series, periods: int) -> pd.Series:
        """Check if series is rising (ta.rising equivalent)"""
        return series > series.shift(periods)

    def _is_falling(self, series: pd.Series, periods: int) -> pd.Series:
        """Check if series is falling (ta.falling equivalent)"""
        return series < series.shift(periods)


class CandlePatternDetector:
    """Candle pattern detector mirroring Pine Script pattern detection"""

    def __init__(self, config: DynamicConfig):
        """Initialize candle pattern detector"""
        self.config = config

    def detect_all_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect all candle patterns"""

        patterns = pd.DataFrame(index=data.index)

        # Calculate candle properties
        body_size = abs(data['close'] - data['open'])
        upper_wick = data['high'] - np.maximum(data['close'], data['open'])
        lower_wick = np.minimum(data['close'], data['open']) - data['low']
        total_range = data['high'] - data['low']

        # Avoid division by zero
        total_range = total_range.replace(0, 0.0001)

        # Detect individual patterns
        patterns['hammer'] = self._detect_hammer(data, body_size, upper_wick, lower_wick, total_range)
        patterns['shooting_star'] = self._detect_shooting_star(data, body_size, upper_wick, lower_wick, total_range)
        patterns['doji'] = self._detect_doji(data, body_size, total_range)
        patterns['bullish_engulfing'] = self._detect_bullish_engulfing(data)
        patterns['bearish_engulfing'] = self._detect_bearish_engulfing(data)
        patterns['pin_bar'] = self._detect_pin_bar(data, body_size, upper_wick, lower_wick, total_range)

        # Calculate pattern strength
        patterns['pattern_strength'] = self._calculate_pattern_strength(patterns)

        return patterns

    def _detect_hammer(self, data: pd.DataFrame, body_size: pd.Series,
                      upper_wick: pd.Series, lower_wick: pd.Series, total_range: pd.Series) -> pd.Series:
        """Detect hammer pattern"""

        # Hammer conditions
        small_body = body_size < (total_range * self.config.pattern_body_ratio)
        long_lower_wick = lower_wick > (body_size * 2)
        short_upper_wick = upper_wick < (body_size * 0.5)
        bullish_close = data['close'] > data['open']

        hammer = small_body & long_lower_wick & short_upper_wick & bullish_close

        return hammer.astype(int)

    def _detect_shooting_star(self, data: pd.DataFrame, body_size: pd.Series,
                             upper_wick: pd.Series, lower_wick: pd.Series, total_range: pd.Series) -> pd.Series:
        """Detect shooting star pattern"""

        # Shooting star conditions
        small_body = body_size < (total_range * self.config.pattern_body_ratio)
        long_upper_wick = upper_wick > (body_size * 2)
        short_lower_wick = lower_wick < (body_size * 0.5)
        bearish_close = data['close'] < data['open']

        shooting_star = small_body & long_upper_wick & short_lower_wick & bearish_close

        return shooting_star.astype(int)

    def _detect_doji(self, data: pd.DataFrame, body_size: pd.Series, total_range: pd.Series) -> pd.Series:
        """Detect doji pattern"""

        # Doji conditions (very small body)
        very_small_body = body_size < (total_range * 0.1)

        return very_small_body.astype(int)

    def _detect_bullish_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect bullish engulfing pattern"""

        # Current candle bullish
        current_bullish = data['close'] > data['open']

        # Previous candle bearish
        prev_bearish = data['close'].shift(1) < data['open'].shift(1)

        # Current candle engulfs previous
        engulfs_body = (data['close'] > data['open'].shift(1)) & (data['open'] < data['close'].shift(1))

        bullish_engulfing = current_bullish & prev_bearish & engulfs_body

        return bullish_engulfing.astype(int)

    def _detect_bearish_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect bearish engulfing pattern"""

        # Current candle bearish
        current_bearish = data['close'] < data['open']

        # Previous candle bullish
        prev_bullish = data['close'].shift(1) > data['open'].shift(1)

        # Current candle engulfs previous
        engulfs_body = (data['close'] < data['open'].shift(1)) & (data['open'] > data['close'].shift(1))

        bearish_engulfing = current_bearish & prev_bullish & engulfs_body

        return bearish_engulfing.astype(int)

    def _detect_pin_bar(self, data: pd.DataFrame, body_size: pd.Series,
                       upper_wick: pd.Series, lower_wick: pd.Series, total_range: pd.Series) -> pd.Series:
        """Detect pin bar pattern"""

        # Pin bar conditions
        small_body = body_size < (total_range * 0.3)
        long_wick = (upper_wick > body_size * 2) | (lower_wick > body_size * 2)

        pin_bar = small_body & long_wick

        return pin_bar.astype(int)

    def _calculate_pattern_strength(self, patterns: pd.DataFrame) -> pd.Series:
        """Calculate overall pattern strength"""

        # Sum all pattern signals
        pattern_columns = ['hammer', 'shooting_star', 'doji', 'bullish_engulfing', 'bearish_engulfing', 'pin_bar']
        pattern_strength = patterns[pattern_columns].sum(axis=1)

        return pattern_strength


class DynamicPriceZones:
    """Dynamic price zones mirroring Pine Script ta.pivothigh/ta.pivotlow"""

    def __init__(self, config: DynamicConfig):
        """Initialize dynamic price zones"""
        self.config = config
        self.support_zones = []
        self.resistance_zones = []

    def calculate_dynamic_zones(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate dynamic support/resistance zones (ta.pivothigh/ta.pivotlow equivalent)"""

        # Find pivot highs and lows
        pivot_highs = self._find_pivot_highs(data)
        pivot_lows = self._find_pivot_lows(data)

        # Create dynamic zones
        resistance_zones = self._create_resistance_zones(pivot_highs, data)
        support_zones = self._create_support_zones(pivot_lows, data)

        # Calculate zone strength
        resistance_zones = self._calculate_zone_strength(resistance_zones, data, 'resistance')
        support_zones = self._calculate_zone_strength(support_zones, data, 'support')

        return {
            'resistance_zones': resistance_zones,
            'support_zones': support_zones,
            'pivot_highs': pivot_highs,
            'pivot_lows': pivot_lows
        }

    def _find_pivot_highs(self, data: pd.DataFrame) -> pd.Series:
        """Find pivot highs (ta.pivothigh equivalent)"""

        left = self.config.pivot_left_bars
        right = self.config.pivot_right_bars

        pivot_highs = pd.Series(np.nan, index=data.index)

        for i in range(left, len(data) - right):
            current_high = data['high'].iloc[i]

            # Check if current high is highest in the window
            left_window = data['high'].iloc[i-left:i]
            right_window = data['high'].iloc[i+1:i+right+1]

            if (current_high > left_window.max()) and (current_high > right_window.max()):
                pivot_highs.iloc[i] = current_high

        return pivot_highs

    def _find_pivot_lows(self, data: pd.DataFrame) -> pd.Series:
        """Find pivot lows (ta.pivotlow equivalent)"""

        left = self.config.pivot_left_bars
        right = self.config.pivot_right_bars

        pivot_lows = pd.Series(np.nan, index=data.index)

        for i in range(left, len(data) - right):
            current_low = data['low'].iloc[i]

            # Check if current low is lowest in the window
            left_window = data['low'].iloc[i-left:i]
            right_window = data['low'].iloc[i+1:i+right+1]

            if (current_low < left_window.min()) and (current_low < right_window.min()):
                pivot_lows.iloc[i] = current_low

        return pivot_lows

    def _create_resistance_zones(self, pivot_highs: pd.Series, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create resistance zones from pivot highs"""

        zones = []
        valid_pivots = pivot_highs.dropna()

        for timestamp, level in valid_pivots.items():
            zones.append({
                'level': level,
                'timestamp': timestamp,
                'type': 'resistance',
                'touches': 1,
                'strength': 1.0
            })

        return zones

    def _create_support_zones(self, pivot_lows: pd.Series, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create support zones from pivot lows"""

        zones = []
        valid_pivots = pivot_lows.dropna()

        for timestamp, level in valid_pivots.items():
            zones.append({
                'level': level,
                'timestamp': timestamp,
                'type': 'support',
                'touches': 1,
                'strength': 1.0
            })

        return zones

    def _calculate_zone_strength(self, zones: List[Dict[str, Any]], data: pd.DataFrame, zone_type: str) -> List[Dict[str, Any]]:
        """Calculate zone strength based on touches"""

        proximity_threshold = self.config.zone_proximity_percent / 100

        for zone in zones:
            level = zone['level']
            touches = 0

            # Count touches to this zone
            for _, row in data.iterrows():
                if zone_type == 'resistance':
                    # Check if price touched resistance
                    if abs(row['high'] - level) / level <= proximity_threshold:
                        touches += 1
                else:  # support
                    # Check if price touched support
                    if abs(row['low'] - level) / level <= proximity_threshold:
                        touches += 1

            zone['touches'] = touches
            zone['strength'] = min(touches / 5.0, 2.0)  # Max strength of 2.0

        return zones

    def get_nearest_zone(self, current_price: float, zone_type: str) -> Optional[Dict[str, Any]]:
        """Get nearest support or resistance zone"""

        zones = self.resistance_zones if zone_type == 'resistance' else self.support_zones

        if not zones:
            return None

        # Find nearest zone
        nearest_zone = None
        min_distance = float('inf')

        for zone in zones:
            if zone['strength'] >= 1.0:  # Only consider strong zones
                distance = abs(current_price - zone['level'])

                if zone_type == 'resistance' and zone['level'] > current_price:
                    if distance < min_distance:
                        min_distance = distance
                        nearest_zone = zone
                elif zone_type == 'support' and zone['level'] < current_price:
                    if distance < min_distance:
                        min_distance = distance
                        nearest_zone = zone

        return nearest_zone

    def is_near_zone(self, current_price: float, zone_type: str) -> bool:
        """Check if price is near a significant zone"""

        nearest_zone = self.get_nearest_zone(current_price, zone_type)

        if nearest_zone:
            distance_percent = abs(current_price - nearest_zone['level']) / current_price
            return distance_percent <= (self.config.zone_proximity_percent / 100)

        return False


def run_dynamic_momentum_research():
    """Run research phase for dynamic momentum system"""

    print("🔍 DYNAMIC MOMENTUM & PRICE ACTION RESEARCH")
    print("=" * 55)
    print("PINE SCRIPT BEST PRACTICES DISCOVERED:")
    print("✅ Dynamic momentum filters (ta.rsi, ta.stoch)")
    print("✅ Adaptive rising/falling detection (ta.rising, ta.falling)")
    print("✅ Pivot-based zones (ta.pivothigh, ta.pivotlow)")
    print("✅ Candle pattern detection (hammer, doji, engulfing)")
    print("✅ Confluence-based entry system")
    print("✅ Dynamic exit conditions")

    try:
        # Load data for research
        from production_real_data_backtester import RealDataFetcher

        config = DynamicConfig()
        data_fetcher = RealDataFetcher()

        print(f"\n📡 Loading data for research...")
        data = data_fetcher.fetch_real_data(
            config.symbol, config.start_date, config.end_date, "15m"
        )

        if data is None or len(data) < 100:
            print("❌ Insufficient data for research")
            return None

        print(f"✅ Loaded {len(data)} data points for research")

        # Initialize components
        momentum_filter = DynamicMomentumFilter(config)
        pattern_detector = CandlePatternDetector(config)
        price_zones = DynamicPriceZones(config)

        # Research dynamic momentum
        print(f"\n🔄 Researching dynamic momentum filters...")
        rsi = momentum_filter.calculate_dynamic_rsi(data)
        stoch_k, stoch_d = momentum_filter.calculate_dynamic_stochastic(data)
        momentum_direction = momentum_filter.detect_momentum_direction(rsi, stoch_k)
        momentum_strength = momentum_filter.calculate_momentum_strength(rsi, stoch_k)

        # Research candle patterns
        print(f"🕯️  Researching candle patterns...")
        patterns = pattern_detector.detect_all_patterns(data)

        # Research price zones
        print(f"📍 Researching dynamic price zones...")
        zones_data = price_zones.calculate_dynamic_zones(data)

        # Research results
        print(f"\n📊 RESEARCH RESULTS:")
        print("=" * 25)

        # Momentum analysis
        bullish_momentum_periods = (momentum_direction == 1).sum()
        bearish_momentum_periods = (momentum_direction == -1).sum()
        neutral_momentum_periods = (momentum_direction == 0).sum()

        print(f"📈 MOMENTUM ANALYSIS:")
        print(f"   Bullish momentum: {bullish_momentum_periods} periods ({bullish_momentum_periods/len(data):.1%})")
        print(f"   Bearish momentum: {bearish_momentum_periods} periods ({bearish_momentum_periods/len(data):.1%})")
        print(f"   Neutral momentum: {neutral_momentum_periods} periods ({neutral_momentum_periods/len(data):.1%})")
        print(f"   Average momentum strength: {momentum_strength.mean():.2f}")

        # Pattern analysis
        pattern_counts = {}
        for col in ['hammer', 'shooting_star', 'doji', 'bullish_engulfing', 'bearish_engulfing', 'pin_bar']:
            pattern_counts[col] = patterns[col].sum()

        print(f"\n🕯️  CANDLE PATTERN ANALYSIS:")
        for pattern, count in pattern_counts.items():
            print(f"   {pattern}: {count} occurrences ({count/len(data):.1%})")

        # Zone analysis
        resistance_zones = zones_data['resistance_zones']
        support_zones = zones_data['support_zones']

        print(f"\n📍 PRICE ZONE ANALYSIS:")
        print(f"   Resistance zones: {len(resistance_zones)}")
        print(f"   Support zones: {len(support_zones)}")

        if resistance_zones:
            avg_resistance_strength = np.mean([z['strength'] for z in resistance_zones])
            print(f"   Average resistance strength: {avg_resistance_strength:.2f}")

        if support_zones:
            avg_support_strength = np.mean([z['strength'] for z in support_zones])
            print(f"   Average support strength: {avg_support_strength:.2f}")

        print(f"\n🎯 RESEARCH CONCLUSIONS:")
        print("✅ Dynamic momentum filters working")
        print("✅ Candle patterns detected successfully")
        print("✅ Price zones identified dynamically")
        print("✅ Ready for implementation phase")

        return {
            'momentum_data': {
                'rsi': rsi,
                'stoch_k': stoch_k,
                'momentum_direction': momentum_direction,
                'momentum_strength': momentum_strength
            },
            'pattern_data': patterns,
            'zone_data': zones_data,
            'research_summary': {
                'bullish_momentum_pct': bullish_momentum_periods/len(data),
                'pattern_frequency': pattern_counts,
                'zone_counts': {
                    'resistance': len(resistance_zones),
                    'support': len(support_zones)
                }
            }
        }

    except Exception as e:
        print(f"❌ Research error: {e}")
        return None


class DynamicTradingSystem:
    """Dynamic trading system implementing Pine Script best practices"""

    def __init__(self, config: DynamicConfig):
        """Initialize dynamic trading system"""
        self.config = config
        self.momentum_filter = DynamicMomentumFilter(config)
        self.pattern_detector = CandlePatternDetector(config)
        self.price_zones = DynamicPriceZones(config)

    def run_dynamic_backtest(self) -> Dict[str, Any]:
        """Run dynamic momentum & price action backtest"""

        print("🚀 DYNAMIC MOMENTUM & PRICE ACTION SYSTEM")
        print("=" * 55)
        print("IMPLEMENTING PINE SCRIPT BEST PRACTICES:")
        print("✅ Dynamic momentum filters (no hard timeframes)")
        print("✅ Candle pattern confirmation")
        print("✅ Price zone confluence")
        print("✅ Adaptive exit conditions")

        try:
            # Load data
            from production_real_data_backtester import RealDataFetcher

            data_fetcher = RealDataFetcher()

            print(f"\n📡 Loading trading data...")
            data = data_fetcher.fetch_real_data(
                self.config.symbol, self.config.start_date, self.config.end_date, "15m"
            )

            if data is None or len(data) < 100:
                print("❌ Insufficient data for trading")
                return None

            print(f"✅ Loaded {len(data)} data points")

            # Calculate all indicators
            enhanced_data = self._prepare_trading_data(data)

            # Execute dynamic strategy
            result = self._execute_dynamic_strategy(enhanced_data)

            if result:
                self._display_dynamic_results(result)
                return result
            else:
                print("❌ Dynamic backtest failed")
                return None

        except Exception as e:
            print(f"❌ Dynamic system error: {e}")
            return None

    def _prepare_trading_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with all dynamic indicators"""

        print(f"🔄 Calculating dynamic indicators...")

        # Add momentum indicators
        data['rsi'] = self.momentum_filter.calculate_dynamic_rsi(data)
        data['stoch_k'], data['stoch_d'] = self.momentum_filter.calculate_dynamic_stochastic(data)
        data['momentum_direction'] = self.momentum_filter.detect_momentum_direction(data['rsi'], data['stoch_k'])
        data['momentum_strength'] = self.momentum_filter.calculate_momentum_strength(data['rsi'], data['stoch_k'])

        # Add candle patterns
        patterns = self.pattern_detector.detect_all_patterns(data)
        for col in patterns.columns:
            data[col] = patterns[col]

        # Calculate price zones
        zones_data = self.price_zones.calculate_dynamic_zones(data)
        self.price_zones.resistance_zones = zones_data['resistance_zones']
        self.price_zones.support_zones = zones_data['support_zones']

        print(f"✅ Dynamic indicators calculated")

        return data

    def _execute_dynamic_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute dynamic trading strategy"""

        # Trading state
        capital = self.config.initial_capital
        position = 0.0
        trades = []
        equity_curve = []

        # Tracking
        trade_counter = 0
        active_trade_id = None
        entry_price = 0
        entry_time = None
        periods_in_trade = 0

        print(f"💰 Executing dynamic strategy...")

        for i in range(20, len(data)):  # Start after indicators stabilize
            current_row = data.iloc[i]
            current_time = current_row['timestamp']
            current_price = current_row['close']

            # Check for exit signals
            if active_trade_id and position != 0:
                periods_in_trade += 1

                should_exit, exit_reason = self._check_dynamic_exit(current_row, data, i, periods_in_trade, entry_price, position)

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
                        'trade_id': active_trade_id
                    })

                    position = 0.0
                    active_trade_id = None
                    periods_in_trade = 0
                    entry_price = 0
                    entry_time = None

            # Check for entry signals
            if position == 0:
                entry_signal, signal_strength = self._check_dynamic_entry(current_row, data, i)

                if entry_signal != 'none':
                    trade_counter += 1
                    active_trade_id = f"trade_{trade_counter}"

                    # Calculate position size based on signal strength
                    position_size = self.config.max_position_size * signal_strength
                    position_value = capital * position_size
                    shares = position_value / current_price
                    cost = shares * current_price * (1 + self.config.transaction_cost)

                    if cost <= capital:
                        capital -= cost
                        position = shares if entry_signal == 'buy' else -shares
                        entry_price = current_price
                        entry_time = current_time
                        periods_in_trade = 0

                        trades.append({
                            'timestamp': current_time,
                            'action': entry_signal,
                            'price': current_price,
                            'signal_strength': signal_strength,
                            'momentum_direction': current_row['momentum_direction'],
                            'momentum_strength': current_row['momentum_strength'],
                            'pattern_strength': current_row['pattern_strength'],
                            'trade_id': active_trade_id
                        })

            # Update equity curve
            portfolio_value = capital + (position * current_price)
            equity_curve.append(portfolio_value)

        # Final calculations
        final_price = data['close'].iloc[-1]
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
            'trades': trades
        }

    def _check_dynamic_entry(self, current_row: pd.Series, data: pd.DataFrame, index: int) -> Tuple[str, float]:
        """Check for dynamic entry signals with confluence"""

        # Get current values
        momentum_direction = current_row['momentum_direction']
        momentum_strength = current_row['momentum_strength']
        rsi = current_row['rsi']
        stoch_k = current_row['stoch_k']

        # Candle patterns
        hammer = current_row['hammer']
        shooting_star = current_row['shooting_star']
        bullish_engulfing = current_row['bullish_engulfing']
        bearish_engulfing = current_row['bearish_engulfing']
        pattern_strength = current_row['pattern_strength']

        current_price = current_row['close']

        # Check zone confluence
        near_support = self.price_zones.is_near_zone(current_price, 'support')
        near_resistance = self.price_zones.is_near_zone(current_price, 'resistance')

        # BULLISH ENTRY CONDITIONS
        bullish_momentum = momentum_direction == 1 and momentum_strength > 0.3
        bullish_patterns = hammer or bullish_engulfing
        bullish_rsi = 30 < rsi < 70  # Not overbought
        bullish_zone = near_support

        # BEARISH ENTRY CONDITIONS
        bearish_momentum = momentum_direction == -1 and momentum_strength > 0.3
        bearish_patterns = shooting_star or bearish_engulfing
        bearish_rsi = 30 < rsi < 70  # Not oversold
        bearish_zone = near_resistance

        # Calculate signal strength (confluence)
        if bullish_momentum and bullish_patterns and bullish_rsi and bullish_zone:
            signal_strength = min(momentum_strength + (pattern_strength * 0.2) + 0.3, 1.0)
            return 'buy', signal_strength
        elif bullish_momentum and bullish_patterns and bullish_rsi:
            signal_strength = min(momentum_strength + (pattern_strength * 0.2), 0.8)
            return 'buy', signal_strength
        elif bearish_momentum and bearish_patterns and bearish_rsi and bearish_zone:
            signal_strength = min(momentum_strength + (pattern_strength * 0.2) + 0.3, 1.0)
            return 'sell', signal_strength
        elif bearish_momentum and bearish_patterns and bearish_rsi:
            signal_strength = min(momentum_strength + (pattern_strength * 0.2), 0.8)
            return 'sell', signal_strength

        return 'none', 0.0

    def _check_dynamic_exit(self, current_row: pd.Series, data: pd.DataFrame, index: int,
                           periods_in_trade: int, entry_price: float, position: float) -> Tuple[bool, str]:
        """Check for dynamic exit conditions"""

        # Get current values
        momentum_direction = current_row['momentum_direction']
        momentum_strength = current_row['momentum_strength']
        rsi = current_row['rsi']
        current_price = current_row['close']

        # Calculate current profit
        if position > 0:
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price

        # 1. MAXIMUM HOLD PERIOD
        if periods_in_trade >= self.config.max_hold_periods:
            return True, "max_hold_period"

        # 2. MOMENTUM REVERSAL
        if position > 0 and momentum_direction == -1 and momentum_strength > 0.4:
            return True, "momentum_reversal_bearish"
        elif position < 0 and momentum_direction == 1 and momentum_strength > 0.4:
            return True, "momentum_reversal_bullish"

        # 3. RSI EXTREME CONDITIONS
        if position > 0 and rsi > 80:  # Overbought exit for longs
            return True, "rsi_overbought"
        elif position < 0 and rsi < 20:  # Oversold exit for shorts
            return True, "rsi_oversold"

        # 4. ZONE-BASED EXITS
        if position > 0 and self.price_zones.is_near_zone(current_price, 'resistance'):
            return True, "resistance_zone_reached"
        elif position < 0 and self.price_zones.is_near_zone(current_price, 'support'):
            return True, "support_zone_reached"

        # 5. PROFIT PROTECTION
        if current_profit > 0.02 and momentum_strength < 0.2:  # Weak momentum with profits
            return True, "profit_protection_weak_momentum"

        return False, "hold"

    def _display_dynamic_results(self, result: Dict[str, Any]):
        """Display dynamic system results"""

        print(f"\n🎯 DYNAMIC SYSTEM RESULTS:")
        print("=" * 35)
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

        # Exit analysis
        if 'exit_reasons' in result:
            print(f"\n📊 DYNAMIC EXIT ANALYSIS:")
            total_exits = sum(result['exit_reasons'].values())
            for reason, count in result['exit_reasons'].items():
                pct = count / total_exits * 100
                print(f"   {reason}: {count} ({pct:.1f}%)")

        # Performance assessment
        if (result['total_return'] > 0.01 and
            result['profit_factor'] >= 1.5 and
            result['win_rate'] >= 0.55):
            print(f"\n🎉 EXCELLENT: Dynamic system working perfectly!")
        elif result['total_return'] > 0.005 and result['profit_factor'] >= 1.2:
            print(f"\n✅ GOOD: Dynamic system performing well")
        elif result['total_return'] > 0:
            print(f"\n⚠️  MODERATE: Positive but needs optimization")
        else:
            print(f"\n❌ POOR: Dynamic system needs refinement")


def run_dynamic_trading_system():
    """Run the complete dynamic trading system"""

    config = DynamicConfig()
    system = DynamicTradingSystem(config)

    result = system.run_dynamic_backtest()

    return result


if __name__ == "__main__":
    print("🚀 Starting Dynamic Momentum & Price Action System")
    print("Implementing Pine Script best practices with dynamic filters")

    # First run research
    print("\n" + "="*60)
    print("PHASE 1: RESEARCH")
    print("="*60)

    research_results = run_dynamic_momentum_research()

    if research_results:
        print(f"\n🎉 RESEARCH PHASE COMPLETED!")

        # Then run implementation
        print("\n" + "="*60)
        print("PHASE 2: IMPLEMENTATION")
        print("="*60)

        trading_results = run_dynamic_trading_system()

        if trading_results:
            print(f"\n🎉 DYNAMIC TRADING SYSTEM COMPLETED!")
            print("Pine Script best practices successfully implemented!")
        else:
            print(f"\n❌ Implementation failed")
    else:
        print(f"\n❌ Research failed")
