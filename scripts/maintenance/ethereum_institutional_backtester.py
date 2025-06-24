#!/usr/bin/env python3
"""
ETHEREUM INSTITUTIONAL BACKTESTER
Extended multi-timeframe analysis with 3-6 month historical data
Professional-grade selective trading system optimized for ETH
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthereumInstitutionalBacktester:
    """Professional Ethereum trading system with extended historical analysis"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
        
        # Extended timeframe structure for deeper analysis
        self.timeframes = {
            'weekly': '1w',     # Long-term trend (26 weeks = 6 months)
            'daily': '1d',      # Primary structure (180 days = 6 months)
            'h12': '12h',       # Intermediate trend
            'h4': '4h',         # Setup identification
            'h1': '1h',         # Entry refinement
            'm30': '30m',       # Precise timing
            'm15': '15m'        # Execution timeframe
        }
        
        # Institutional-grade parameters optimized for ETH
        self.min_confluence = 72  # 72% threshold for ETH (slightly higher than BTC)
        self.max_risk_per_trade = 1.2  # 1.2% risk per trade
        self.min_risk_reward = 2.5     # Minimum 2.5:1 RR
        self.max_daily_signals = 3     # Maximum 3 signals per day
        self.max_positions = 2         # Maximum 2 concurrent positions
        
        # Enhanced confluence weights for ETH characteristics
        self.confluence_weights = {
            'trend_alignment': 28,      # Multi-timeframe trend (critical for ETH)
            'momentum': 22,             # RSI/MACD/Stoch signals  
            'volume': 18,               # Volume confirmation (important for ETH)
            'support_resistance': 20,   # Key S/R levels
            'volatility': 8,            # Market volatility
            'price_action': 4           # Price action patterns
        }
        
        # Portfolio tracking
        self.initial_capital = 25000  # Higher capital for ETH testing
        self.current_capital = self.initial_capital
        self.trades = []
        self.daily_signals = {}
        
        logger.info("🚀 ETHEREUM INSTITUTIONAL BACKTESTER INITIALIZED")
        logger.info(f"   💎 Symbol: ETH/USDT")
        logger.info(f"   📊 Confluence Threshold: {self.min_confluence}%")
        logger.info(f"   🎯 Max Daily Signals: {self.max_daily_signals}")
        logger.info(f"   ⚖️ Risk Management: {self.max_risk_per_trade}% per trade")
        logger.info(f"   💰 Initial Capital: ${self.initial_capital:,}")

    def fetch_extended_data(self, symbol: str = 'ETH/USDT', months: int = 6) -> Dict[str, pd.DataFrame]:
        """Fetch extended historical data across all timeframes"""
        logger.info(f"📊 Fetching {months}-month extended data for {symbol}...")
        
        data = {}
        
        try:
            for name, timeframe in self.timeframes.items():
                # Calculate appropriate limits for extended periods
                if timeframe == '1w':
                    limit = months * 4 + 4  # Weeks + buffer
                elif timeframe == '1d':
                    limit = months * 30 + 10  # Days + buffer
                elif timeframe == '12h':
                    limit = months * 60 + 20  # 12h periods + buffer
                elif timeframe == '4h':
                    limit = months * 180 + 50  # 4h periods + buffer
                elif timeframe == '1h':
                    limit = months * 720 + 100  # Hourly + buffer (limited to last portion)
                elif timeframe == '30m':
                    limit = min(1000, months * 30 + 50)  # 30m periods (API limited)
                else:  # 15m
                    limit = min(1000, months * 15 + 50)  # 15m periods (API limited)
                    
                logger.info(f"   Fetching {name} ({timeframe}) - {limit} candles...")
                
                # Fetch data with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.error(f"   ❌ Failed to fetch {name} after {max_retries} attempts: {e}")
                            continue
                        logger.warning(f"   ⚠️ Retry {attempt + 1} for {name}...")
                        continue
                
                if not ohlcv:
                    logger.error(f"   ❌ No data for {name}")
                    continue
                
                # Process data
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add comprehensive indicators
                self.add_ethereum_indicators(df)
                
                data[name] = df
                start_date = df.index[0].strftime('%Y-%m-%d')
                end_date = df.index[-1].strftime('%Y-%m-%d')
                logger.info(f"   ✅ {name.upper()}: {len(df)} candles ({start_date} to {end_date})")
                
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching extended data: {e}")
            return {}

    def add_ethereum_indicators(self, df: pd.DataFrame):
        """Add comprehensive technical indicators optimized for Ethereum"""
        
        # Trend indicators (multiple timeframes)
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # EMA calculation
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Momentum indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_fast'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        df['rsi_slow'] = ta.momentum.RSIIndicator(df['close'], window=21).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        
        # ATR for volatility
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Volume indicators (critical for ETH)
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_rsi'] = ta.momentum.RSIIndicator(df['volume'], window=14).rsi()
        
        # Support/Resistance levels
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot'] - df['low']
        df['support_1'] = 2 * df['pivot'] - df['high']
        df['resistance_2'] = df['pivot'] + (df['high'] - df['low'])
        df['support_2'] = df['pivot'] - (df['high'] - df['low'])
        
        # Price action patterns
        df['price_range'] = ((df['high'] - df['low']) / df['close']) * 100
        df['body_size'] = abs(df['close'] - df['open']) / df['close'] * 100
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close'] * 100
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close'] * 100
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        
        # Ichimoku components
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        # Fibonacci retracement levels (dynamic)
        df['fib_236'] = df['low'].rolling(20).min() + 0.236 * (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        df['fib_382'] = df['low'].rolling(20).min() + 0.382 * (df['high'].rolling(20).max() - df['low'].rolling(20).min())
        df['fib_618'] = df['low'].rolling(20).min() + 0.618 * (df['high'].rolling(20).max() - df['low'].rolling(20).min())

    def calculate_ethereum_confluence(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict:
        """Calculate enhanced confluence specifically for Ethereum with multi-timeframe analysis"""
        
        confluence_scores = {}
        
        # Get current values from each available timeframe
        timeframe_values = {}
        for tf_name, tf_data in data.items():
            if len(tf_data) > 50:  # Ensure sufficient data
                # Find the closest timestamp
                closest_idx = tf_data.index.get_indexer([current_time], method='nearest')[0]
                if closest_idx >= 0 and closest_idx < len(tf_data):
                    timeframe_values[tf_name] = tf_data.iloc[closest_idx]
        
        if len(timeframe_values) < 3:  # Need at least 3 timeframes
            return {'percentage': 0, 'meets_threshold': False, 'scores': {}}
        
        # 1. ENHANCED TREND ALIGNMENT (28 points)
        trend_score = 0
        trend_signals = []
        
        for tf_name, current_data in timeframe_values.items():
            tf_trend = 0
            
            # Multi-level trend analysis
            if not pd.isna(current_data.get('sma_20', np.nan)) and not pd.isna(current_data.get('sma_50', np.nan)):
                price = current_data['close']
                sma_20 = current_data['sma_20']
                sma_50 = current_data['sma_50']
                ema_50 = current_data.get('ema_50', sma_50)
                
                # Strong bullish
                if price > sma_20 > sma_50 and price > ema_50:
                    tf_trend = 2
                # Weak bullish
                elif price > sma_20 or price > sma_50:
                    tf_trend = 1
                # Strong bearish
                elif price < sma_20 < sma_50 and price < ema_50:
                    tf_trend = -2
                # Weak bearish
                elif price < sma_20 or price < sma_50:
                    tf_trend = -1
                
                trend_signals.append((tf_name, tf_trend))
        
        # Calculate trend alignment score
        if len(trend_signals) >= 3:
            # Weight higher timeframes more heavily
            weights = {'weekly': 4, 'daily': 3, 'h12': 2, 'h4': 2, 'h1': 1, 'm30': 1, 'm15': 1}
            
            bullish_strength = sum(weights.get(tf, 1) * max(0, trend) for tf, trend in trend_signals)
            bearish_strength = sum(weights.get(tf, 1) * abs(min(0, trend)) for tf, trend in trend_signals)
            
            total_weight = sum(weights.get(tf, 1) for tf, _ in trend_signals)
            
            if bullish_strength > bearish_strength * 1.5:  # Strong bullish alignment
                trend_score = min(28, int(28 * bullish_strength / (total_weight * 2)))
            elif bearish_strength > bullish_strength * 1.5:  # Strong bearish alignment
                trend_score = min(28, int(28 * bearish_strength / (total_weight * 2)))
            else:  # Mixed or weak trends
                trend_score = 8
        
        confluence_scores['trend_alignment'] = trend_score
        
        # 2. ENHANCED MOMENTUM (22 points)
        momentum_score = 0
        
        # Use daily timeframe for momentum if available
        if 'daily' in timeframe_values:
            daily_data = timeframe_values['daily']
            
            # RSI analysis (multiple timeframes)
            rsi_score = 0
            if not pd.isna(daily_data.get('rsi', np.nan)):
                rsi = daily_data['rsi']
                rsi_fast = daily_data.get('rsi_fast', rsi)
                
                # Optimal RSI zones for entry
                if 45 <= rsi <= 65:  # Neutral momentum zone
                    rsi_score = 8
                elif 35 <= rsi <= 75:  # Acceptable zone
                    rsi_score = 6
                elif rsi < 30 or rsi > 70:  # Extreme zones
                    rsi_score = 4
                    
                # Fast RSI confirmation
                if abs(rsi_fast - rsi) < 10:  # RSI convergence
                    rsi_score += 2
            
            # MACD analysis
            macd_score = 0
            if (not pd.isna(daily_data.get('macd', np.nan)) and 
                not pd.isna(daily_data.get('macd_signal', np.nan)) and
                not pd.isna(daily_data.get('macd_histogram', np.nan))):
                
                macd = daily_data['macd']
                macd_signal = daily_data['macd_signal']
                macd_hist = daily_data['macd_histogram']
                
                # MACD crossover and histogram analysis
                if macd > macd_signal and macd_hist > 0:  # Strong bullish
                    macd_score = 8
                elif macd < macd_signal and macd_hist < 0:  # Strong bearish
                    macd_score = 8
                elif macd > macd_signal or macd < macd_signal:  # Crossover
                    macd_score = 5
                else:
                    macd_score = 2
            
            # Stochastic analysis
            stoch_score = 0
            if (not pd.isna(daily_data.get('stoch_k', np.nan)) and 
                not pd.isna(daily_data.get('stoch_d', np.nan))):
                
                stoch_k = daily_data['stoch_k']
                stoch_d = daily_data['stoch_d']
                
                # Stochastic crossover in optimal zones
                if stoch_k > stoch_d and 20 < stoch_k < 80:  # Bullish crossover
                    stoch_score = 6
                elif stoch_k < stoch_d and 20 < stoch_k < 80:  # Bearish crossover
                    stoch_score = 6
                elif 30 < stoch_k < 70:  # Neutral zone
                    stoch_score = 3
            
            momentum_score = min(22, rsi_score + macd_score + stoch_score)
        
        confluence_scores['momentum'] = momentum_score
        
        # 3. ENHANCED VOLUME ANALYSIS (18 points) - Critical for ETH
        volume_score = 0
        
        if 'daily' in timeframe_values:
            daily_data = timeframe_values['daily']
            
            # Volume ratio analysis
            if not pd.isna(daily_data.get('volume_ratio', np.nan)):
                vol_ratio = daily_data['volume_ratio']
                
                if vol_ratio > 1.5:  # High volume
                    volume_score += 8
                elif vol_ratio > 1.2:  # Above average volume
                    volume_score += 6
                elif vol_ratio > 0.8:  # Normal volume
                    volume_score += 4
                else:  # Low volume
                    volume_score += 2
            
            # Volume RSI
            if not pd.isna(daily_data.get('volume_rsi', np.nan)):
                vol_rsi = daily_data['volume_rsi']
                
                if 40 <= vol_rsi <= 60:  # Balanced volume momentum
                    volume_score += 5
                elif 30 <= vol_rsi <= 70:  # Acceptable volume momentum
                    volume_score += 3
                else:
                    volume_score += 1
            
            # Price-volume relationship
            if (not pd.isna(daily_data.get('close', np.nan)) and 
                not pd.isna(daily_data.get('volume', np.nan))):
                
                # Check for volume confirmation with price movement
                price_change = (daily_data['close'] - daily_data['open']) / daily_data['open'] * 100
                
                if abs(price_change) > 2 and daily_data.get('volume_ratio', 1) > 1.2:
                    volume_score += 5  # Strong price-volume confirmation
                elif abs(price_change) > 1 and daily_data.get('volume_ratio', 1) > 1.0:
                    volume_score += 3  # Moderate confirmation
        
        confluence_scores['volume'] = min(18, volume_score)
        
        # 4. ENHANCED SUPPORT/RESISTANCE (20 points)
        sr_score = 0
        
        if 'daily' in timeframe_values:
            daily_data = timeframe_values['daily']
            price = daily_data['close']
            
            # Multiple S/R level analysis
            sr_levels = []
            for level_name in ['support_1', 'resistance_1', 'support_2', 'resistance_2', 
                              'pivot', 'bb_upper', 'bb_lower']:
                if not pd.isna(daily_data.get(level_name, np.nan)):
                    sr_levels.append(daily_data[level_name])
            
            if sr_levels:
                # Find closest S/R level
                sr_distances = [abs(price - level) / price * 100 for level in sr_levels]
                min_distance = min(sr_distances)
                
                if min_distance < 0.5:  # Very close to S/R (within 0.5%)
                    sr_score = 20
                elif min_distance < 1.0:  # Close to S/R (within 1%)
                    sr_score = 15
                elif min_distance < 2.0:  # Near S/R (within 2%)
                    sr_score = 10
                elif min_distance < 3.0:  # Moderate distance
                    sr_score = 5
                else:
                    sr_score = 2
        
        confluence_scores['support_resistance'] = sr_score
        
        # 5. VOLATILITY ANALYSIS (8 points)
        volatility_score = 0
        
        if 'daily' in timeframe_values:
            daily_data = timeframe_values['daily']
            
            # ATR-based volatility
            if not pd.isna(daily_data.get('atr', np.nan)):
                atr = daily_data['atr']
                price = daily_data['close']
                atr_pct = atr / price * 100
                
                if 1.5 <= atr_pct <= 4.0:  # Optimal volatility for ETH
                    volatility_score = 8
                elif 1.0 <= atr_pct <= 6.0:  # Acceptable volatility
                    volatility_score = 6
                else:  # Too low or too high volatility
                    volatility_score = 3
            
            # Bollinger Band width
            if not pd.isna(daily_data.get('bb_width', np.nan)):
                bb_width = daily_data['bb_width']
                
                if 4 <= bb_width <= 12:  # Good volatility range
                    volatility_score = max(volatility_score, 6)
                elif 2 <= bb_width <= 15:  # Acceptable range
                    volatility_score = max(volatility_score, 4)
        
        confluence_scores['volatility'] = volatility_score
        
        # 6. PRICE ACTION PATTERNS (4 points)
        price_action_score = 0
        
        if 'daily' in timeframe_values:
            daily_data = timeframe_values['daily']
            
            # Candlestick pattern analysis
            body_size = daily_data.get('body_size', 0)
            upper_shadow = daily_data.get('upper_shadow', 0)
            lower_shadow = daily_data.get('lower_shadow', 0)
            
            # Strong body with minimal shadows (trending)
            if body_size > 1.5 and upper_shadow < 0.5 and lower_shadow < 0.5:
                price_action_score = 4
            # Doji or spinning top (indecision)
            elif body_size < 0.3:
                price_action_score = 2
            # Hammer or shooting star
            elif (lower_shadow > body_size * 2) or (upper_shadow > body_size * 2):
                price_action_score = 3
            else:
                price_action_score = 1
        
        confluence_scores['price_action'] = price_action_score
        
        # Calculate total confluence percentage
        total_score = sum(confluence_scores.values())
        max_possible = sum(self.confluence_weights.values())
        confluence_percentage = (total_score / max_possible) * 100
        
        meets_threshold = confluence_percentage >= self.min_confluence
        
        return {
            'percentage': confluence_percentage,
            'meets_threshold': meets_threshold,
            'scores': confluence_scores,
            'trend_signals': trend_signals if 'trend_signals' in locals() else []
        }

    def run_ethereum_backtest(self, months: int = 6) -> Dict:
        """Run comprehensive Ethereum backtest over extended period"""
        
        symbol = 'ETH/USDT'
        logger.info(f"🚀 STARTING ETHEREUM INSTITUTIONAL BACKTEST ({months} MONTHS)")
        logger.info("=" * 80)
        
        # Fetch extended multi-timeframe data
        data = self.fetch_extended_data(symbol, months)
        if not data or len(data) < 3:
            logger.error("❌ Insufficient data for backtesting")
            return {}
        
        # Use daily timeframe as base for iteration
        daily_data = data['daily'].copy()
        
        # Limit to actual backtest period
        end_date = daily_data.index[-1]
        start_date = end_date - timedelta(days=months * 30)
        backtest_period = daily_data[daily_data.index >= start_date].copy()
        
        logger.info(f"📊 Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"📈 Total Days: {len(backtest_period)}")
        
        signals = []
        trades = []
        current_date = None
        signals_today = 0
        
        # Analyze each day in the backtest period
        for i in range(50, len(daily_data)):  # Start after sufficient lookback
            current_day = daily_data.iloc[i]
            current_time = current_day.name
            current_price = current_day['close']
            
            # Skip if outside backtest period
            if current_time < start_date:
                continue
            
            # Reset daily signal counter
            if current_date is None or current_time.date() != current_date:
                current_date = current_time.date()
                signals_today = 0
            
            if signals_today >= self.max_daily_signals:
                continue
            
            # Calculate confluence for this timestamp
            confluence = self.calculate_ethereum_confluence(data, current_time)
            
            analysis = {
                'date': current_time.strftime('%Y-%m-%d'),
                'timestamp': current_time,
                'price': current_price,
                'confluence': confluence['percentage'],
                'signal_generated': confluence['meets_threshold'],
                'confluence_breakdown': confluence['scores']
            }
            
            signals.append(analysis)
            
            # Generate trade if signal triggered
            if confluence['meets_threshold']:
                signals_today += 1
                
                # Determine direction based on trend signals
                trend_signals = confluence.get('trend_signals', [])
                bullish_weight = sum(max(0, trend) for _, trend in trend_signals)
                bearish_weight = sum(abs(min(0, trend)) for _, trend in trend_signals)
                
                if bullish_weight > bearish_weight:
                    direction = 'LONG'
                    stop_loss = current_price * (1 - self.max_risk_per_trade / 100)
                    take_profit = current_price * (1 + (self.max_risk_per_trade * self.min_risk_reward) / 100)
                else:
                    direction = 'SHORT'
                    stop_loss = current_price * (1 + self.max_risk_per_trade / 100)
                    take_profit = current_price * (1 - (self.max_risk_per_trade * self.min_risk_reward) / 100)
                
                trade = {
                    'date': current_time.strftime('%Y-%m-%d'),
                    'timestamp': current_time,
                    'direction': direction,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confluence': confluence['percentage'],
                    'risk_percent': self.max_risk_per_trade,
                    'rr_ratio': self.min_risk_reward,
                    'confluence_breakdown': confluence['scores']
                }
                
                trades.append(trade)
                logger.info(f"{'🟢' if direction == 'LONG' else '🔴'} {direction} Signal: {current_time.strftime('%Y-%m-%d')} @ ${current_price:,.2f} ({confluence['percentage']:.1f}% confluence)")
        
        # Calculate results
        total_signals = len([s for s in signals if s['signal_generated']])
        total_days = len(backtest_period)
        signal_rate = (total_signals / total_days * 100) if total_days > 0 else 0
        avg_confluence = np.mean([s['confluence'] for s in signals]) if signals else 0
        max_confluence = max([s['confluence'] for s in signals]) if signals else 0
        
        results = {
            'symbol': symbol,
            'backtest_period': f"{months} months",
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'total_days_analyzed': len(signals),
            'total_signals': total_signals,
            'signal_rate': f"{signal_rate:.2f}%",
            'average_confluence': f"{avg_confluence:.1f}%",
            'maximum_confluence': f"{max_confluence:.1f}%",
            'threshold_used': f"{self.min_confluence}%",
            'trades_generated': len(trades),
            'max_daily_signals': self.max_daily_signals,
            'risk_per_trade': f"{self.max_risk_per_trade}%",
            'min_rr_ratio': f"{self.min_risk_reward}:1",
            'signals': signals,
            'trades': trades,
            'timeframes_used': list(self.timeframes.keys())
        }
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ethereum_institutional_backtest_{months}m_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print comprehensive summary
        self.print_ethereum_results(results)
        
        return results

    def print_ethereum_results(self, results: Dict):
        """Print comprehensive Ethereum backtest results"""
        
        print(f"\n💎 ETHEREUM INSTITUTIONAL BACKTEST RESULTS")
        print("=" * 60)
        print(f"📊 Symbol: {results['symbol']}")
        print(f"📅 Period: {results['backtest_period']} ({results['start_date']} to {results['end_date']})")
        print(f"🗓️ Total Days Analyzed: {results['total_days_analyzed']}")
        print(f"🎯 Signals Generated: {results['total_signals']}")
        print(f"📈 Signal Rate: {results['signal_rate']}")
        print(f"⚡ Average Confluence: {results['average_confluence']}")
        print(f"🔥 Maximum Confluence: {results['maximum_confluence']}")
        print(f"🎲 Threshold Used: {results['threshold_used']}")
        print(f"💰 Trades Generated: {results['trades_generated']}")
        print(f"⚖️ Risk per Trade: {results['risk_per_trade']}")
        print(f"📊 Min R:R Ratio: {results['min_rr_ratio']}")
        print(f"📈 Timeframes: {', '.join(results['timeframes_used'])}")
        
        if results['trades_generated'] > 0:
            trades = results['trades']
            
            print(f"\n📋 TRADE BREAKDOWN:")
            print("-" * 40)
            
            long_trades = [t for t in trades if t['direction'] == 'LONG']
            short_trades = [t for t in trades if t['direction'] == 'SHORT']
            
            print(f"🟢 LONG Trades: {len(long_trades)}")
            print(f"🔴 SHORT Trades: {len(short_trades)}")
            
            if long_trades:
                print(f"\n🟢 LONG ENTRIES:")
                for trade in long_trades:
                    print(f"   {trade['date']}: ${trade['entry_price']:,.2f} (Confluence: {trade['confluence']:.1f}%)")
            
            if short_trades:
                print(f"\n🔴 SHORT ENTRIES:")
                for trade in short_trades:
                    print(f"   {trade['date']}: ${trade['entry_price']:,.2f} (Confluence: {trade['confluence']:.1f}%)")
            
            # Confluence analysis
            confluences = [t['confluence'] for t in trades]
            print(f"\n📊 CONFLUENCE STATISTICS:")
            print(f"   Average: {np.mean(confluences):.1f}%")
            print(f"   Median: {np.median(confluences):.1f}%")
            print(f"   Min: {np.min(confluences):.1f}%")
            print(f"   Max: {np.max(confluences):.1f}%")
        
        print(f"\n💾 Detailed results saved to: ethereum_institutional_backtest_{results['backtest_period'].replace(' ', '')}_*.json")

def main():
    """Run the Ethereum institutional backtester"""
    backtester = EthereumInstitutionalBacktester()
    
    # Run 6-month backtest (can be adjusted to 3 months)
    print("Choose backtest period:")
    print("1. 3 months")
    print("2. 6 months")
    
    choice = input("Enter choice (1 or 2, default 6): ").strip()
    
    if choice == "1":
        months = 3
    else:
        months = 6
    
    print(f"\n🚀 Starting {months}-month Ethereum institutional backtest...")
    
    results = backtester.run_ethereum_backtest(months)
    
    if results and results['trades_generated'] > 0:
        print(f"\n✅ Ethereum institutional backtest complete!")
        print(f"📈 {results['trades_generated']} high-quality signals generated over {months} months")
        print(f"📊 Average confluence: {results['average_confluence']}")
        print(f"🎯 Signal rate: {results['signal_rate']}")
    else:
        print("\n⚠️ No signals generated - market conditions did not meet institutional criteria")
        print("💡 Consider adjusting confluence threshold or extending time period")

if __name__ == "__main__":
    main() 