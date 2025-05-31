#!/usr/bin/env python3
"""
Trend-Zone Trading System

CORRECT IMPLEMENTATION based on user clarification:

TREND (Higher Timeframes):
- 1H, 4H, 1D trend direction
- Exit when TREND dies (not momentum)
- Trend death = higher timeframe structure breaks

MOMENTUM (Lower Timeframes):
- 5m, 15m momentum for entry/exit timing
- Use momentum to time entries within trend
- Use momentum to time exits at zones

ZONES (Support/Resistance):
- Key levels that determine trade duration
- Trade from zone to zone
- Exit at next significant zone
- Zones determine how long to hold trades

Strategy:
1. Higher timeframe TREND for direction
2. Lower timeframe MOMENTUM for timing
3. ZONES for targets and duration
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

class TrendState(Enum):
    """Higher timeframe trend states"""
    STRONG_UP = "strong_up"
    UP = "up"
    NEUTRAL = "neutral"
    DOWN = "down"
    STRONG_DOWN = "strong_down"

@dataclass
class TrendZoneConfig:
    """Configuration for trend-zone system"""
    # Entry parameters
    symbol: str = "BTCUSDT"
    start_date: str = "2024-01-01"
    end_date: str = "2024-02-01"
    initial_capital: float = 10000.0
    confidence_threshold: float = 0.6
    max_position_size: float = 0.1
    max_daily_trades: int = 8
    transaction_cost: float = 0.001

    # Trend analysis (Higher timeframes)
    trend_timeframes: List[str] = None
    trend_death_threshold: float = 0.02  # 2% trend structure break

    # Momentum analysis (Lower timeframes)
    momentum_timeframes: List[str] = None
    momentum_entry_threshold: float = 0.001  # 0.1% momentum for entry
    momentum_exit_threshold: float = -0.0005  # -0.05% momentum for exit

    # Zone analysis
    zone_lookback_periods: int = 50  # Look back 50 periods for zones
    zone_strength_threshold: float = 3  # Minimum 3 touches for strong zone
    zone_proximity_threshold: float = 0.005  # 0.5% proximity to zone

    def __post_init__(self):
        if self.trend_timeframes is None:
            self.trend_timeframes = ["1d", "4h", "1h"]
        if self.momentum_timeframes is None:
            self.momentum_timeframes = ["15m", "5m"]


class TrendAnalyzer:
    """Analyze higher timeframe trends"""

    def __init__(self, timeframe_data: Dict[str, pd.DataFrame]):
        """Initialize trend analyzer"""
        self.timeframe_data = timeframe_data

    def analyze_trend_state(self, current_time: datetime, timeframes: List[str]) -> Tuple[TrendState, float]:
        """Analyze current trend state across timeframes"""

        trend_scores = []

        for tf in timeframes:
            if tf not in self.timeframe_data:
                continue

            data = self.timeframe_data[tf]
            current_candle = self._get_current_candle(data, current_time)

            if current_candle is None:
                continue

            # Analyze trend for this timeframe
            trend_score = self._calculate_trend_score(data, current_candle)

            # Weight by timeframe importance
            weight = {"1d": 3.0, "4h": 2.0, "1h": 1.0}.get(tf, 1.0)
            trend_scores.append(trend_score * weight)

        if not trend_scores:
            return TrendState.NEUTRAL, 0.0

        # Calculate weighted average trend
        avg_trend = sum(trend_scores) / len(trend_scores)
        confidence = min(abs(avg_trend), 1.0)

        # Determine trend state
        if avg_trend >= 0.6:
            state = TrendState.STRONG_UP
        elif avg_trend >= 0.3:
            state = TrendState.UP
        elif avg_trend <= -0.6:
            state = TrendState.STRONG_DOWN
        elif avg_trend <= -0.3:
            state = TrendState.DOWN
        else:
            state = TrendState.NEUTRAL

        return state, confidence

    def is_trend_dead(self, current_time: datetime, timeframes: List[str],
                     previous_trend: TrendState, threshold: float) -> bool:
        """Check if higher timeframe trend has died"""

        current_trend, confidence = self.analyze_trend_state(current_time, timeframes)

        # Trend death conditions
        if previous_trend in [TrendState.STRONG_UP, TrendState.UP]:
            # Uptrend death
            return current_trend in [TrendState.DOWN, TrendState.STRONG_DOWN]
        elif previous_trend in [TrendState.STRONG_DOWN, TrendState.DOWN]:
            # Downtrend death
            return current_trend in [TrendState.UP, TrendState.STRONG_UP]

        return False

    def _calculate_trend_score(self, data: pd.DataFrame, current_candle: pd.Series) -> float:
        """Calculate trend score for timeframe"""

        # Get recent data
        recent_data = data[data.index <= current_candle.name].tail(20)

        if len(recent_data) < 10:
            return 0.0

        score = 0.0

        # 1. Moving average alignment
        if len(recent_data) >= 20:
            ma_fast = recent_data['close'].rolling(10).mean().iloc[-1]
            ma_slow = recent_data['close'].rolling(20).mean().iloc[-1]

            if current_candle['close'] > ma_fast > ma_slow:
                score += 0.4
            elif current_candle['close'] < ma_fast < ma_slow:
                score -= 0.4

        # 2. Price momentum
        price_change = recent_data['close'].pct_change(5).iloc[-1]
        if not pd.isna(price_change):
            score += np.clip(price_change * 10, -0.3, 0.3)  # Scale momentum

        # 3. Higher highs/lower lows
        recent_highs = recent_data['high'].tail(10)
        recent_lows = recent_data['low'].tail(10)

        if len(recent_highs) >= 5:
            if recent_highs.iloc[-1] > recent_highs.iloc[-3]:
                score += 0.2
            elif recent_lows.iloc[-1] < recent_lows.iloc[-3]:
                score -= 0.2

        return np.clip(score, -1.0, 1.0)

    def _get_current_candle(self, data: pd.DataFrame, current_time: datetime) -> Optional[pd.Series]:
        """Get current candle for timeframe"""
        try:
            mask = data['timestamp'] <= current_time
            if mask.any():
                return data[mask].iloc[-1]
            return None
        except Exception:
            return None


class MomentumAnalyzer:
    """Analyze lower timeframe momentum for entry/exit timing"""

    def __init__(self, timeframe_data: Dict[str, pd.DataFrame]):
        """Initialize momentum analyzer"""
        self.timeframe_data = timeframe_data

    def get_entry_momentum(self, current_time: datetime, timeframes: List[str],
                          trend_direction: TrendState) -> Tuple[bool, float]:
        """Get momentum signal for entry timing"""

        momentum_signals = []

        for tf in timeframes:
            if tf not in self.timeframe_data:
                continue

            data = self.timeframe_data[tf]
            current_candle = self._get_current_candle(data, current_time)

            if current_candle is None:
                continue

            # Calculate momentum
            momentum = self._calculate_momentum(data, current_candle)
            momentum_signals.append(momentum)

        if not momentum_signals:
            return False, 0.0

        avg_momentum = sum(momentum_signals) / len(momentum_signals)

        # Check if momentum aligns with trend
        if trend_direction in [TrendState.UP, TrendState.STRONG_UP]:
            entry_signal = avg_momentum > 0.001  # Positive momentum for uptrend
        elif trend_direction in [TrendState.DOWN, TrendState.STRONG_DOWN]:
            entry_signal = avg_momentum < -0.001  # Negative momentum for downtrend
        else:
            entry_signal = False

        return entry_signal, abs(avg_momentum)

    def get_exit_momentum(self, current_time: datetime, timeframes: List[str],
                         position_direction: str) -> Tuple[bool, float]:
        """Get momentum signal for exit timing"""

        momentum_signals = []

        for tf in timeframes:
            if tf not in self.timeframe_data:
                continue

            data = self.timeframe_data[tf]
            current_candle = self._get_current_candle(data, current_time)

            if current_candle is None:
                continue

            momentum = self._calculate_momentum(data, current_candle)
            momentum_signals.append(momentum)

        if not momentum_signals:
            return False, 0.0

        avg_momentum = sum(momentum_signals) / len(momentum_signals)

        # Check for momentum exit signals
        if position_direction == 'buy':
            exit_signal = avg_momentum < -0.0005  # Negative momentum for long exit
        else:  # sell
            exit_signal = avg_momentum > 0.0005   # Positive momentum for short exit

        return exit_signal, abs(avg_momentum)

    def _calculate_momentum(self, data: pd.DataFrame, current_candle: pd.Series) -> float:
        """Calculate momentum for timeframe"""

        recent_data = data[data.index <= current_candle.name].tail(10)

        if len(recent_data) < 5:
            return 0.0

        # Price velocity (rate of change)
        momentum_3 = recent_data['close'].pct_change(3).iloc[-1]
        momentum_5 = recent_data['close'].pct_change(5).iloc[-1]

        # Combine momentums
        if pd.isna(momentum_3):
            momentum_3 = 0
        if pd.isna(momentum_5):
            momentum_5 = 0

        # Weight recent momentum more
        combined_momentum = (momentum_3 * 0.7) + (momentum_5 * 0.3)

        return combined_momentum

    def _get_current_candle(self, data: pd.DataFrame, current_time: datetime) -> Optional[pd.Series]:
        """Get current candle for timeframe"""
        try:
            mask = data['timestamp'] <= current_time
            if mask.any():
                return data[mask].iloc[-1]
            return None
        except Exception:
            return None


class ZoneAnalyzer:
    """Analyze support/resistance zones for trade targets"""

    def __init__(self, price_data: pd.DataFrame):
        """Initialize zone analyzer"""
        self.price_data = price_data
        self.zones = []

    def identify_zones(self, lookback_periods: int = 50) -> List[Dict[str, Any]]:
        """Identify key support/resistance zones"""

        if len(self.price_data) < lookback_periods:
            return []

        recent_data = self.price_data.tail(lookback_periods)
        zones = []

        # Find swing highs and lows
        highs = self._find_swing_points(recent_data['high'], 'high')
        lows = self._find_swing_points(recent_data['low'], 'low')

        # Convert to zones
        for level, touch_list in highs.items():
            if len(touch_list) >= 2:  # Minimum 2 touches
                zones.append({
                    'level': level,
                    'type': 'resistance',
                    'strength': len(touch_list),
                    'last_touch': max([t['time'] for t in touch_list])
                })

        for level, touch_list in lows.items():
            if len(touch_list) >= 2:
                zones.append({
                    'level': level,
                    'type': 'support',
                    'strength': len(touch_list),
                    'last_touch': max([t['time'] for t in touch_list])
                })

        # Sort by strength
        zones.sort(key=lambda x: x['strength'], reverse=True)
        self.zones = zones[:10]  # Keep top 10 zones

        return self.zones

    def get_next_zone(self, current_price: float, direction: str) -> Optional[Dict[str, Any]]:
        """Get next significant zone in trade direction"""

        if not self.zones:
            return None

        relevant_zones = []

        for zone in self.zones:
            if direction == 'buy':
                # For long trades, look for resistance above
                if zone['level'] > current_price and zone['type'] == 'resistance':
                    distance = (zone['level'] - current_price) / current_price
                    relevant_zones.append((zone, distance))
            else:  # sell
                # For short trades, look for support below
                if zone['level'] < current_price and zone['type'] == 'support':
                    distance = (current_price - zone['level']) / current_price
                    relevant_zones.append((zone, distance))

        if not relevant_zones:
            return None

        # Return closest significant zone
        relevant_zones.sort(key=lambda x: x[1])  # Sort by distance
        return relevant_zones[0][0]

    def is_at_zone(self, current_price: float, proximity_threshold: float = 0.005) -> Optional[Dict[str, Any]]:
        """Check if price is at a significant zone"""

        for zone in self.zones:
            distance = abs(current_price - zone['level']) / current_price
            if distance <= proximity_threshold:
                return zone

        return None

    def _find_swing_points(self, series: pd.Series, point_type: str) -> Dict[float, List[Dict]]:
        """Find swing highs or lows"""

        swing_points = {}
        tolerance = 0.002  # 0.2% tolerance for grouping levels

        for i in range(2, len(series) - 2):
            if point_type == 'high':
                # Swing high
                if (series.iloc[i] > series.iloc[i-1] and
                    series.iloc[i] > series.iloc[i-2] and
                    series.iloc[i] > series.iloc[i+1] and
                    series.iloc[i] > series.iloc[i+2]):

                    level = series.iloc[i]
                    time = series.index[i]

                    # Group similar levels
                    grouped = False
                    for existing_level in swing_points.keys():
                        if abs(level - existing_level) / existing_level <= tolerance:
                            swing_points[existing_level].append({'level': level, 'time': time})
                            grouped = True
                            break

                    if not grouped:
                        swing_points[level] = [{'level': level, 'time': time}]

            else:  # low
                # Swing low
                if (series.iloc[i] < series.iloc[i-1] and
                    series.iloc[i] < series.iloc[i-2] and
                    series.iloc[i] < series.iloc[i+1] and
                    series.iloc[i] < series.iloc[i+2]):

                    level = series.iloc[i]
                    time = series.index[i]

                    # Group similar levels
                    grouped = False
                    for existing_level in swing_points.keys():
                        if abs(level - existing_level) / existing_level <= tolerance:
                            swing_points[existing_level].append({'level': level, 'time': time})
                            grouped = True
                            break

                    if not grouped:
                        swing_points[level] = [{'level': level, 'time': time}]

        # Convert to count format
        zone_counts = {}
        for level, touches in swing_points.items():
            zone_counts[level] = touches  # touches is already a list

        return zone_counts


def run_trend_zone_backtest():
    """Run trend-zone trading system backtest"""

    print("🎯 TREND-ZONE TRADING SYSTEM")
    print("=" * 40)
    print("CORRECT IMPLEMENTATION:")
    print("✅ TREND (1H/4H/1D) - Direction & exit when trend dies")
    print("✅ MOMENTUM (5M/15M) - Entry/exit timing")
    print("✅ ZONES (S/R levels) - Trade targets & duration")
    print("✅ Exit when higher timeframe TREND dies (not momentum)")

    try:
        # Import multi-timeframe system
        from multi_timeframe_system_corrected import MultiTimeframeAnalyzer, MultiTimeframeConfig

        config = TrendZoneConfig()
        base_config = MultiTimeframeConfig()
        analyzer = MultiTimeframeAnalyzer(base_config)

        # Load data
        print(f"\n📡 Loading multi-timeframe data...")
        success = analyzer.load_all_timeframe_data(
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date
        )

        if not success:
            print("❌ Failed to load data")
            return None

        # Run trend-zone backtest
        print(f"\n💰 Running trend-zone backtest...")
        result = run_trend_zone_strategy(analyzer, config)

        if result:
            print(f"\n🎯 TREND-ZONE RESULTS:")
            print("=" * 25)
            print(f"Total Trades: {result['total_trades']}")
            print(f"Total Return: {result['total_return']:.2%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"Win Rate: {result['win_rate']:.1%}")
            print(f"Average Winner: {result['avg_winner']:.2%}")
            print(f"Average Loser: {result['avg_loser']:.2%}")
            print(f"Average Hold Time: {result['avg_hold_periods']:.1f} periods")
            print(f"Reward/Risk Ratio: {result['reward_risk_ratio']:.2f}")

            # Exit analysis
            if 'exit_reasons' in result:
                print(f"\n📊 EXIT REASONS:")
                total_exits = sum(result['exit_reasons'].values())
                for reason, count in result['exit_reasons'].items():
                    pct = count / total_exits * 100
                    print(f"   {reason}: {count} ({pct:.1f}%)")

            # Zone analysis
            if 'zone_analysis' in result:
                zone_data = result['zone_analysis']
                print(f"\n🎯 ZONE ANALYSIS:")
                print(f"Zones identified: {zone_data.get('zones_found', 0)}")
                print(f"Zone-based exits: {zone_data.get('zone_exits', 0)}")
                print(f"Zone hit rate: {zone_data.get('zone_hit_rate', 0):.1%}")

            return result

        else:
            print("❌ Trend-zone backtest failed")
            return None

    except Exception as e:
        print(f"❌ Trend-zone system error: {e}")
        return None


def run_trend_zone_strategy(analyzer, config):
    """Run the trend-zone strategy"""

    # Initialize analyzers
    trend_analyzer = TrendAnalyzer(analyzer.timeframe_data)
    momentum_analyzer = MomentumAnalyzer(analyzer.timeframe_data)

    # Use 15m as base timeframe
    base_data = analyzer.timeframe_data['15m'].copy()
    base_data = base_data.set_index('timestamp')

    # Initialize zone analyzer
    zone_analyzer = ZoneAnalyzer(base_data)
    zones = zone_analyzer.identify_zones(config.zone_lookback_periods)

    print(f"📍 Identified {len(zones)} key zones")

    # Trading state
    capital = config.initial_capital
    position = 0.0
    trades = []
    equity_curve = []

    # Tracking
    trade_counter = 0
    active_trade_id = None
    current_trend = TrendState.NEUTRAL
    target_zone = None
    exit_reasons = {}
    zone_analysis = {'zones_found': len(zones), 'zone_exits': 0, 'zone_hits': 0}

    for i in range(50, len(base_data)):
        current_time = base_data.index[i]
        current_price = base_data.iloc[i]['close']

        # Update trend analysis
        trend_state, trend_confidence = trend_analyzer.analyze_trend_state(current_time, config.trend_timeframes)

        # Check for exit signals on active position
        if active_trade_id and position != 0:
            should_exit = False
            exit_reason = ""

            # 1. Check if TREND died (main exit condition)
            if trend_analyzer.is_trend_dead(current_time, config.trend_timeframes, current_trend, config.trend_death_threshold):
                should_exit = True
                exit_reason = "trend_died"

            # 2. Check if reached target zone
            elif target_zone:
                if zone_analyzer.is_at_zone(current_price, config.zone_proximity_threshold):
                    should_exit = True
                    exit_reason = "target_zone_reached"
                    zone_analysis['zone_hits'] += 1

            # 3. Check momentum exit signal
            else:
                position_direction = 'buy' if position > 0 else 'sell'
                momentum_exit, momentum_strength = momentum_analyzer.get_exit_momentum(
                    current_time, config.momentum_timeframes, position_direction
                )
                if momentum_exit and momentum_strength > 0.002:  # Strong momentum exit
                    should_exit = True
                    exit_reason = "momentum_exit"

            # Execute exit
            if should_exit:
                if position > 0:
                    proceeds = position * current_price * (1 - config.transaction_cost)
                    capital += proceeds
                else:
                    cost = abs(position) * current_price * (1 + config.transaction_cost)
                    capital -= cost

                # Calculate profit
                entry_trade = next((t for t in trades if t.get('trade_id') == active_trade_id and t['action'] in ['buy', 'sell']), None)
                if entry_trade:
                    entry_price = entry_trade['price']
                    if position > 0:
                        profit = (current_price - entry_price) / entry_price
                    else:
                        profit = (entry_price - current_price) / entry_price
                else:
                    profit = 0

                exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
                if exit_reason == "target_zone_reached":
                    zone_analysis['zone_exits'] += 1

                trades.append({
                    'timestamp': current_time,
                    'action': 'exit',
                    'price': current_price,
                    'reason': exit_reason,
                    'profit': profit,
                    'trade_id': active_trade_id
                })

                position = 0.0
                active_trade_id = None
                target_zone = None

        # Check for new entry signals (only if no position)
        if position == 0:
            # Check momentum for entry timing
            momentum_entry, momentum_strength = momentum_analyzer.get_entry_momentum(
                current_time, config.momentum_timeframes, trend_state
            )

            # Enter if trend is strong and momentum aligns
            if (trend_state in [TrendState.UP, TrendState.STRONG_UP, TrendState.DOWN, TrendState.STRONG_DOWN] and
                momentum_entry and momentum_strength > config.momentum_entry_threshold and
                trend_confidence > 0.5):

                trade_counter += 1
                active_trade_id = f"trade_{trade_counter}"
                current_trend = trend_state

                # Determine position direction
                if trend_state in [TrendState.UP, TrendState.STRONG_UP]:
                    signal = 'buy'
                    position_size = config.max_position_size * trend_confidence
                    position_value = capital * position_size
                    shares = position_value / current_price
                    cost = shares * current_price * (1 + config.transaction_cost)

                    if cost <= capital:
                        capital -= cost
                        position = shares

                        # Find target zone
                        target_zone = zone_analyzer.get_next_zone(current_price, 'buy')

                else:  # DOWN trend
                    signal = 'sell'
                    position_size = config.max_position_size * trend_confidence
                    position_value = capital * position_size
                    shares = position_value / current_price
                    cost = shares * current_price * (1 + config.transaction_cost)

                    if cost <= capital:
                        capital -= cost
                        position = -shares

                        # Find target zone
                        target_zone = zone_analyzer.get_next_zone(current_price, 'sell')

                if position != 0:
                    trades.append({
                        'timestamp': current_time,
                        'action': signal,
                        'price': current_price,
                        'trend_state': trend_state.value,
                        'momentum_strength': momentum_strength,
                        'target_zone': target_zone['level'] if target_zone else None,
                        'trade_id': active_trade_id
                    })

        # Update equity curve
        portfolio_value = capital + (position * current_price)
        equity_curve.append(portfolio_value)

    # Final calculations
    final_price = base_data.iloc[-1]['close']
    if position != 0:
        if position > 0:
            final_capital = capital + (position * final_price * (1 - config.transaction_cost))
        else:
            final_capital = capital - (abs(position) * final_price * (1 + config.transaction_cost))
    else:
        final_capital = capital

    total_return = (final_capital - config.initial_capital) / config.initial_capital

    # Analyze results
    entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
    exit_trades = [t for t in trades if t['action'] == 'exit']

    # Calculate metrics
    profits = [t.get('profit', 0) for t in exit_trades]
    winners = [p for p in profits if p > 0]
    losers = [p for p in profits if p < 0]

    # Calculate zone hit rate
    if zone_analysis['zone_exits'] > 0:
        zone_analysis['zone_hit_rate'] = zone_analysis['zone_hits'] / zone_analysis['zone_exits']

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
        'avg_hold_periods': len(equity_curve) / len(entry_trades) if entry_trades else 0,
        'exit_reasons': exit_reasons,
        'zone_analysis': zone_analysis,
        'trades': trades
    }


if __name__ == "__main__":
    run_trend_zone_backtest()
