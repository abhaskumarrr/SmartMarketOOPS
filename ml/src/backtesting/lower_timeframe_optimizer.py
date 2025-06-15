#!/usr/bin/env python3
"""
Lower Timeframe Trading Optimizer

This implements support for lower timeframes (1m, 5m, 15m) to dramatically increase
trade frequency and capture more market opportunities.

Key features:
1. Multi-timeframe data fetching (1m, 5m, 15m, 30m, 1h)
2. Timeframe-specific indicators and parameters
3. Scalping strategies for 1-5 minute timeframes
4. Intraday momentum and mean reversion
5. Higher frequency signal generation
6. Optimized for lower timeframe characteristics

Expected improvements:
- 1m timeframe: 60x more data points than 1h
- 5m timeframe: 12x more data points than 1h
- Much higher trade frequency potential
- Better capture of intraday movements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LowerTimeframeConfig:
    """Configuration for lower timeframe trading"""
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"  # 1m, 5m, 15m, 30m, 1h
    start_date: str = "2024-01-01"
    end_date: str = "2024-01-07"  # Shorter period for lower timeframes
    initial_capital: float = 10000.0

    # Timeframe-specific parameters
    confidence_threshold: float = 0.3
    signal_threshold: float = 0.001
    max_position_size: float = 0.1
    max_daily_trades: int = 100  # Much higher for lower timeframes

    # Scalping parameters
    quick_profit_target: float = 0.002  # 0.2% quick profit
    tight_stop_loss: float = 0.001      # 0.1% tight stop
    use_scalping: bool = True

    # Timeframe-specific indicators
    fast_ema: int = 3
    slow_ema: int = 8
    rsi_period: int = 7
    bb_period: int = 10

    # Risk management
    max_drawdown_limit: float = 0.15
    transaction_cost: float = 0.0005  # Lower costs for frequent trading


class LowerTimeframeDataFetcher:
    """
    Enhanced data fetcher with lower timeframe support
    """

    def __init__(self):
        """Initialize lower timeframe data fetcher"""
        self.binance_client = None
        self.delta_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize exchange clients"""
        try:
            import ccxt
            self.binance_client = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
            })
            logger.info("✅ Binance client initialized for lower timeframes")
        except Exception as e:
            logger.warning(f"Binance client failed: {e}")

        try:
            from standalone_delta_client import StandaloneDeltaExchangeClient
            self.delta_client = StandaloneDeltaExchangeClient(testnet=True)
            logger.info("✅ Delta Exchange client initialized for lower timeframes")
        except Exception as e:
            logger.warning(f"Delta client failed: {e}")

    def fetch_lower_timeframe_data(self, symbol: str, timeframe: str,
                                  start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch lower timeframe data with proper handling"""
        logger.info(f"Fetching {timeframe} data for {symbol}")

        # Calculate expected data points
        expected_points = self._calculate_expected_points(timeframe, start_date, end_date)
        logger.info(f"Expected data points: {expected_points:,}")

        # Try Binance first (better for lower timeframes)
        if self.binance_client:
            data = self._fetch_from_binance_lower_tf(symbol, timeframe, start_date, end_date)
            if data is not None and len(data) > 100:
                logger.info(f"✅ Fetched {len(data):,} {timeframe} candles from Binance")
                return data

        # Try Delta Exchange
        if self.delta_client:
            data = self._fetch_from_delta_lower_tf(symbol, timeframe, start_date, end_date)
            if data is not None and len(data) > 100:
                logger.info(f"✅ Fetched {len(data):,} {timeframe} candles from Delta")
                return data

        # Generate realistic lower timeframe data as fallback
        logger.warning("Using realistic lower timeframe fallback data")
        return self._generate_lower_timeframe_fallback(symbol, timeframe, start_date, end_date)

    def _calculate_expected_points(self, timeframe: str, start_date: str, end_date: str) -> int:
        """Calculate expected number of data points"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        total_hours = (end_dt - start_dt).total_seconds() / 3600

        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
            '2h': 120, '4h': 240, '1d': 1440
        }

        minutes = timeframe_minutes.get(timeframe, 60)
        return int(total_hours * 60 / minutes)

    def _fetch_from_binance_lower_tf(self, symbol: str, timeframe: str,
                                    start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch lower timeframe data from Binance"""
        try:
            # Convert dates to timestamps
            start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
            end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

            # Binance has limits, so we might need to fetch in chunks
            all_data = []
            current_start = start_ts

            # Fetch in chunks to avoid rate limits
            while current_start < end_ts:
                try:
                    ohlcv = self.binance_client.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_start,
                        limit=1000  # Binance limit
                    )

                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)

                    # Update start time for next chunk
                    if len(ohlcv) < 1000:
                        break

                    last_timestamp = ohlcv[-1][0]
                    current_start = last_timestamp + 1

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Binance chunk fetch failed: {e}")
                    break

            if not all_data:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Binance lower timeframe fetch error: {e}")
            return None

    def _fetch_from_delta_lower_tf(self, symbol: str, timeframe: str,
                                  start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch lower timeframe data from Delta Exchange"""
        try:
            # Convert symbol format for Delta
            delta_symbol = symbol.replace('USDT', 'USD') if 'USDT' in symbol else symbol

            # Calculate days back
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            days_back = (end_dt - start_dt).days

            # Fetch data
            historical_data = self.delta_client.get_historical_ohlcv(
                symbol=delta_symbol,
                interval=timeframe,
                days_back=min(days_back, 30),  # Delta limits
                end_time=end_dt
            )

            if not historical_data:
                return None

            # Convert to DataFrame
            df_data = []
            for candle in historical_data:
                if isinstance(candle, dict):
                    df_data.append({
                        'timestamp': pd.to_datetime(candle.get('timestamp', 0), unit='ms'),
                        'open': float(candle.get('open', 0)),
                        'high': float(candle.get('high', 0)),
                        'low': float(candle.get('low', 0)),
                        'close': float(candle.get('close', 0)),
                        'volume': float(candle.get('volume', 0))
                    })

            if not df_data:
                return None

            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Delta lower timeframe fetch error: {e}")
            return None

    def _generate_lower_timeframe_fallback(self, symbol: str, timeframe: str,
                                         start_date: str, end_date: str) -> pd.DataFrame:
        """Generate realistic lower timeframe fallback data"""
        logger.info(f"Generating realistic {timeframe} fallback data")

        # Calculate periods
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        timeframe_freq = {
            '1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T', '1h': 'H'
        }

        freq = timeframe_freq.get(timeframe, '5T')
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)

        # Base price
        base_price = 45000.0 if 'BTC' in symbol.upper() else 2500.0

        # Generate realistic lower timeframe price movements
        np.random.seed(42)

        # Lower timeframe characteristics
        if timeframe == '1m':
            base_volatility = 0.0008  # 0.08% per minute
            noise_factor = 0.5
        elif timeframe == '5m':
            base_volatility = 0.002   # 0.2% per 5 minutes
            noise_factor = 0.3
        elif timeframe == '15m':
            base_volatility = 0.004   # 0.4% per 15 minutes
            noise_factor = 0.2
        else:
            base_volatility = 0.008   # 0.8% per 30 minutes
            noise_factor = 0.1

        prices = [base_price]

        for i in range(len(timestamps)):
            # Base random walk
            base_return = np.random.normal(0, base_volatility)

            # Add microstructure noise (more for lower timeframes)
            noise = np.random.normal(0, base_volatility * noise_factor)

            # Add some autocorrelation (momentum/mean reversion)
            if len(prices) > 1:
                recent_return = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0

                # Lower timeframes have more mean reversion
                if timeframe in ['1m', '5m']:
                    base_return -= recent_return * 0.1  # Mean reversion
                else:
                    base_return += recent_return * 0.05  # Slight momentum

            # Add occasional spikes (more frequent in lower timeframes)
            spike_probability = 0.01 if timeframe == '1m' else 0.005
            if np.random.random() < spike_probability:
                spike = np.random.normal(0, base_volatility * 3)
                base_return += spike

            total_return = base_return + noise
            new_price = prices[-1] * (1 + total_return)
            prices.append(max(new_price, 0.01))

        # Generate OHLCV data
        data = []
        for i, (timestamp, close_price) in enumerate(zip(timestamps, prices[1:])):
            open_price = prices[i]

            # Realistic intraday range (smaller for lower timeframes)
            range_factor = base_volatility * 0.5
            intraday_range = abs(np.random.normal(0, close_price * range_factor))

            high_price = max(open_price, close_price) + intraday_range
            low_price = min(open_price, close_price) - intraday_range

            # Ensure OHLC consistency
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # Volume (higher frequency for lower timeframes)
            base_volume = 500000 if timeframe == '1m' else 1000000
            price_change = abs(close_price - open_price) / open_price
            volume_multiplier = 1 + (price_change * 10)
            volume = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0)

            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df):,} realistic {timeframe} candles")
        return df


class LowerTimeframeSignalGenerator:
    """
    Signal generator optimized for lower timeframes
    """

    def __init__(self, config: LowerTimeframeConfig):
        """Initialize lower timeframe signal generator"""
        self.config = config
        self.timeframe_multipliers = {
            '1m': 5.0,   # Very aggressive for 1-minute
            '5m': 3.0,   # Aggressive for 5-minute
            '15m': 2.0,  # Moderate for 15-minute
            '30m': 1.5,  # Slightly aggressive for 30-minute
            '1h': 1.0    # Standard for 1-hour
        }
        self.sensitivity = self.timeframe_multipliers.get(config.timeframe, 1.0)

    def generate_signals(self, data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Generate signals optimized for lower timeframes"""
        if index < 20:
            return {'signal': 'hold', 'confidence': 0.0, 'method': 'insufficient_data'}

        current_data = data.iloc[max(0, index-20):index+1]
        latest = current_data.iloc[-1]

        signals = []

        # 1. Scalping signals (very short-term)
        if self.config.use_scalping:
            scalping_signal = self._generate_scalping_signal(current_data)
            signals.append(scalping_signal)

        # 2. Micro momentum signals
        momentum_signal = self._generate_micro_momentum_signal(current_data)
        signals.append(momentum_signal)

        # 3. Quick mean reversion signals
        reversion_signal = self._generate_quick_reversion_signal(current_data)
        signals.append(reversion_signal)

        # 4. Breakout signals (for lower timeframes)
        breakout_signal = self._generate_micro_breakout_signal(current_data)
        signals.append(breakout_signal)

        # 5. Volume spike signals
        if 'volume' in current_data.columns:
            volume_signal = self._generate_volume_spike_signal(current_data)
            signals.append(volume_signal)

        # Combine signals with timeframe sensitivity
        return self._combine_lower_timeframe_signals(signals)

    def _generate_scalping_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate scalping signals for very short-term trades"""
        latest = data.iloc[-1]

        # Very short EMAs for scalping
        ema_fast = data['close'].ewm(span=self.config.fast_ema).mean().iloc[-1]
        ema_slow = data['close'].ewm(span=self.config.slow_ema).mean().iloc[-1]

        # Micro price movements
        micro_change = data['close'].pct_change(1).iloc[-1]

        score = 0
        confidence = 0.2 * self.sensitivity

        # EMA crossover (very sensitive)
        if latest['close'] > ema_fast > ema_slow:
            score += 1.0
            confidence += 0.3
        elif latest['close'] < ema_fast < ema_slow:
            score -= 1.0
            confidence += 0.3

        # Micro momentum
        if micro_change > 0.0005 * self.sensitivity:  # 0.05% for 1m, scaled for others
            score += 0.8
            confidence += 0.2
        elif micro_change < -0.0005 * self.sensitivity:
            score -= 0.8
            confidence += 0.2

        # Quick profit opportunity
        if abs(micro_change) > self.config.quick_profit_target / 2:
            confidence += 0.3

        if score > 0.6:
            return {'signal': 'buy', 'confidence': min(confidence, 0.9), 'method': 'scalping'}
        elif score < -0.6:
            return {'signal': 'sell', 'confidence': min(confidence, 0.9), 'method': 'scalping'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'scalping'}

    def _generate_micro_momentum_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate micro momentum signals"""
        # Very short-term momentum
        returns_1 = data['close'].pct_change(1).iloc[-1]
        returns_3 = data['close'].pct_change(3).iloc[-1]
        returns_5 = data['close'].pct_change(5).iloc[-1]

        score = 0
        confidence = 0.2 * self.sensitivity

        # Immediate momentum
        threshold_1 = 0.0003 * self.sensitivity  # Very small for lower timeframes
        if returns_1 > threshold_1:
            score += 1.2
            confidence += 0.4
        elif returns_1 < -threshold_1:
            score -= 1.2
            confidence += 0.4

        # Short momentum
        threshold_3 = 0.001 * self.sensitivity
        if returns_3 > threshold_3:
            score += 0.8
            confidence += 0.2
        elif returns_3 < -threshold_3:
            score -= 0.8
            confidence += 0.2

        # Momentum acceleration
        if len(data) >= 6:
            momentum_change = returns_3 - data['close'].pct_change(3).iloc[-2]
            if abs(momentum_change) > 0.0002 * self.sensitivity:
                confidence += 0.2
                if momentum_change > 0:
                    score += 0.5
                else:
                    score -= 0.5

        if score > 0.7:
            return {'signal': 'buy', 'confidence': min(confidence, 0.9), 'method': 'micro_momentum'}
        elif score < -0.7:
            return {'signal': 'sell', 'confidence': min(confidence, 0.9), 'method': 'micro_momentum'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'micro_momentum'}

    def _generate_quick_reversion_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate quick mean reversion signals"""
        latest = data.iloc[-1]

        # Very short Bollinger Bands
        sma = data['close'].rolling(self.config.bb_period).mean().iloc[-1]
        std = data['close'].rolling(self.config.bb_period).std().iloc[-1]

        bb_upper = sma + (std * 1.5)  # Tighter bands for lower timeframes
        bb_lower = sma - (std * 1.5)

        # Distance from mean
        deviation = (latest['close'] - sma) / sma

        score = 0
        confidence = 0.2 * self.sensitivity

        # Quick reversion opportunities
        reversion_threshold = 0.002 * self.sensitivity  # Smaller for lower timeframes

        if latest['close'] < bb_lower or deviation < -reversion_threshold:
            score += 1.0  # Buy on dips
            confidence += 0.4
        elif latest['close'] > bb_upper or deviation > reversion_threshold:
            score -= 1.0  # Sell on peaks
            confidence += 0.4

        # Extreme deviations (quick scalp opportunities)
        if abs(deviation) > 0.005 * self.sensitivity:
            confidence += 0.3

        if score > 0.6:
            return {'signal': 'buy', 'confidence': min(confidence, 0.8), 'method': 'quick_reversion'}
        elif score < -0.6:
            return {'signal': 'sell', 'confidence': min(confidence, 0.8), 'method': 'quick_reversion'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'quick_reversion'}

    def _generate_micro_breakout_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate micro breakout signals"""
        latest = data.iloc[-1]

        # Very short-term high/low
        lookback = max(3, int(10 / self.sensitivity))  # Shorter for lower timeframes
        high_recent = data['high'].rolling(lookback).max().iloc[-1]
        low_recent = data['low'].rolling(lookback).min().iloc[-1]

        score = 0
        confidence = 0.2 * self.sensitivity

        # Micro breakouts (very small movements)
        breakout_threshold = 0.0002 * self.sensitivity

        if latest['close'] > high_recent * (1 + breakout_threshold):
            score += 1.0
            confidence += 0.4
        elif latest['close'] < low_recent * (1 - breakout_threshold):
            score -= 1.0
            confidence += 0.4

        # Volume confirmation
        if 'volume' in data.columns:
            vol_avg = data['volume'].rolling(lookback).mean().iloc[-1]
            if latest['volume'] > vol_avg * 1.3:
                confidence += 0.2

        if score > 0.5:
            return {'signal': 'buy', 'confidence': min(confidence, 0.8), 'method': 'micro_breakout'}
        elif score < -0.5:
            return {'signal': 'sell', 'confidence': min(confidence, 0.8), 'method': 'micro_breakout'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'micro_breakout'}

    def _generate_volume_spike_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate volume spike signals"""
        latest = data.iloc[-1]

        # Volume analysis
        vol_avg = data['volume'].rolling(10).mean().iloc[-1]
        vol_ratio = latest['volume'] / vol_avg

        # Price-volume relationship
        price_change = data['close'].pct_change(1).iloc[-1]

        score = 0
        confidence = 0.2

        # Volume spikes (more sensitive for lower timeframes)
        spike_threshold = 1.5 / self.sensitivity  # Lower threshold for lower timeframes

        if vol_ratio > spike_threshold:
            confidence += 0.3
            if price_change > 0.0001 * self.sensitivity:
                score += 0.8
            elif price_change < -0.0001 * self.sensitivity:
                score -= 0.8

        # Extreme volume
        if vol_ratio > 2.0:
            confidence += 0.2

        if score > 0.4:
            return {'signal': 'buy', 'confidence': min(confidence, 0.7), 'method': 'volume_spike'}
        elif score < -0.4:
            return {'signal': 'sell', 'confidence': min(confidence, 0.7), 'method': 'volume_spike'}
        else:
            return {'signal': 'hold', 'confidence': confidence, 'method': 'volume_spike'}

    def _combine_lower_timeframe_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine signals with lower timeframe sensitivity"""
        buy_signals = [s for s in signals if s['signal'] == 'buy']
        sell_signals = [s for s in signals if s['signal'] == 'sell']

        # Aggressive combination for lower timeframes
        buy_strength = sum(s['confidence'] for s in buy_signals) * self.sensitivity
        sell_strength = sum(s['confidence'] for s in sell_signals) * self.sensitivity

        # Lower threshold for action (more aggressive)
        action_threshold = 0.2 / self.sensitivity

        if buy_strength > action_threshold and buy_strength > sell_strength:
            return {
                'signal': 'buy',
                'confidence': min(buy_strength / len(signals), 0.95),
                'method': f'lower_tf_ensemble_{self.config.timeframe}',
                'component_signals': len(buy_signals),
                'timeframe_sensitivity': self.sensitivity
            }
        elif sell_strength > action_threshold and sell_strength > buy_strength:
            return {
                'signal': 'sell',
                'confidence': min(sell_strength / len(signals), 0.95),
                'method': f'lower_tf_ensemble_{self.config.timeframe}',
                'component_signals': len(sell_signals),
                'timeframe_sensitivity': self.sensitivity
            }
        else:
            return {
                'signal': 'hold',
                'confidence': 0.1,
                'method': f'lower_tf_ensemble_{self.config.timeframe}',
                'component_signals': 0,
                'timeframe_sensitivity': self.sensitivity
            }


def run_lower_timeframe_optimization():
    """Run comprehensive lower timeframe optimization"""
    print("⚡ LOWER TIMEFRAME TRADING OPTIMIZATION")
    print("=" * 60)
    print("Testing multiple lower timeframes for maximum trade frequency:")
    print("✅ 5-minute timeframe (12x more data than 1h)")
    print("✅ 15-minute timeframe (4x more data than 1h)")
    print("✅ 1-hour timeframe (baseline)")
    print("✅ Scalping and micro-movement strategies")
    print("✅ Timeframe-specific indicators and parameters")

    timeframes_to_test = ['5m', '15m', '1h']  # Start with manageable timeframes

    results = {}

    for timeframe in timeframes_to_test:
        print(f"\n🔄 Testing {timeframe} timeframe...")

        # Adjust date range based on timeframe
        if timeframe == '1m':
            start_date, end_date = "2024-01-01", "2024-01-03"  # 2 days for 1m
        elif timeframe == '5m':
            start_date, end_date = "2024-01-01", "2024-01-07"  # 1 week for 5m
        elif timeframe == '15m':
            start_date, end_date = "2024-01-01", "2024-01-14"  # 2 weeks for 15m
        else:
            start_date, end_date = "2024-01-01", "2024-01-31"  # 1 month for higher TF

        config = LowerTimeframeConfig(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            confidence_threshold=0.3,
            signal_threshold=0.001,
            max_daily_trades=200 if timeframe == '1m' else 100 if timeframe == '5m' else 50
        )

        result = test_lower_timeframe_strategy(config)
        if result:
            results[timeframe] = result

            print(f"   ✅ {timeframe}: {result['total_trades']} trades, {result['total_return']:.2%} return")
        else:
            print(f"   ❌ {timeframe}: Failed")

    # Analyze and compare results
    if results:
        print(f"\n📊 LOWER TIMEFRAME COMPARISON")
        print("=" * 50)

        # Sort by trade frequency
        sorted_results = sorted(results.items(), key=lambda x: x[1]['total_trades'], reverse=True)

        baseline_trades = 4  # Original system

        for timeframe, result in sorted_results:
            improvement = result['total_trades'] / baseline_trades

            print(f"\n🏆 {timeframe.upper()} TIMEFRAME:")
            print(f"   🔄 Total Trades: {result['total_trades']}")
            print(f"   📈 Total Return: {result['total_return']:.2%}")
            print(f"   📊 Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"   📉 Max Drawdown: {result['max_drawdown']:.2%}")
            print(f"   🎯 Win Rate: {result['win_rate']:.2%}")
            print(f"   🚀 Improvement: {improvement:.1f}x more trades")
            print(f"   📅 Data Points: {result['data_points']:,}")

            # Performance assessment
            if result['total_trades'] >= 100:
                print(f"   ✅ EXCELLENT trade frequency!")
            elif result['total_trades'] >= 50:
                print(f"   ⚠️  Good trade frequency")
            elif result['total_trades'] >= 20:
                print(f"   ⚠️  Moderate improvement")
            else:
                print(f"   ❌ Still low frequency")

        # Find best overall
        best_timeframe = max(results.items(),
                           key=lambda x: x[1]['total_trades'] * (1 + x[1]['total_return']))

        print(f"\n🏆 BEST OVERALL TIMEFRAME: {best_timeframe[0].upper()}")
        print(f"   Optimal balance of frequency and performance")
        print(f"   {best_timeframe[1]['total_trades']} trades with {best_timeframe[1]['total_return']:.2%} return")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("=" * 20)

        highest_freq = max(results.items(), key=lambda x: x[1]['total_trades'])
        most_profitable = max(results.items(), key=lambda x: x[1]['total_return'])

        print(f"🔄 For maximum trade frequency: {highest_freq[0]} ({highest_freq[1]['total_trades']} trades)")
        print(f"💰 For maximum profitability: {most_profitable[0]} ({most_profitable[1]['total_return']:.2%} return)")
        print(f"⚖️  For balanced approach: {best_timeframe[0]} (best overall score)")

        return results

    else:
        print("❌ No successful lower timeframe tests")
        return None


def test_lower_timeframe_strategy(config: LowerTimeframeConfig) -> Optional[Dict[str, Any]]:
    """Test lower timeframe strategy"""
    try:
        # Fetch lower timeframe data
        data_fetcher = LowerTimeframeDataFetcher()
        data = data_fetcher.fetch_lower_timeframe_data(
            symbol=config.symbol,
            timeframe=config.timeframe,
            start_date=config.start_date,
            end_date=config.end_date
        )

        if data is None or len(data) < 50:
            logger.warning(f"Insufficient data for {config.timeframe}")
            return None

        # Create enhanced data with lower timeframe indicators
        enhanced_data = create_lower_timeframe_indicators(data, config)

        # Run backtest
        result = run_lower_timeframe_backtest(enhanced_data, config)

        if result:
            result['data_points'] = len(data)
            result['timeframe'] = config.timeframe

        return result

    except Exception as e:
        logger.error(f"Lower timeframe test failed: {e}")
        return None
