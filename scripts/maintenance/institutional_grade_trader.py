#!/usr/bin/env python3
"""
INSTITUTIONAL GRADE SMART MONEY CONCEPTS TRADER
Professional-grade selective trading system based on real institutional logic
Quality over quantity - 2-5 high-probability signals per day
"""

import ccxt
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix technical analysis imports
try:
    import ta
    from ta.trend import ema_indicator, macd_diff
    from ta.momentum import rsi
    from ta.volatility import average_true_range, bollinger_hband, bollinger_lband
except ImportError as e:
    logger.error(f"Technical analysis library import error: {e}")
    logger.info("Installing ta library: pip install ta")

class InstitutionalGradeTrader:
    """Professional Smart Money Concepts Trading Engine"""
    
    def __init__(self):
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        
        # **INSTITUTIONAL PARAMETERS**
        self.confluence_threshold = 85  # 85%+ confluence required
        self.base_risk_percent = 0.5    # 0.5% base risk per trade
        self.max_risk_percent = 1.0     # 1% maximum risk per trade
        self.min_risk_reward = 3.0      # Minimum 3:1 risk-reward
        self.max_daily_signals = 1      # Maximum 1 signal per day
        self.max_positions = 1          # Maximum 1 concurrent position
        
        # **TIMEFRAME HIERARCHY**
        self.timeframes = {
            'weekly': '1w',    # Primary trend
            'daily': '1d',     # Key levels and structure
            'h4': '4h',        # Setup identification
            'h1': '1h',        # Entry timing
            'entry': '15m'     # Precise execution
        }
        
        # **MARKET REGIME FILTERS**
        self.allowed_regimes = ['trending', 'consolidation']
        self.volatility_threshold = 0.05  # 5% max volatility
        
        # Initialize exchange
        try:
            self.exchange = ccxt.binance({
                'apiKey': '',
                'secret': '',
                'sandbox': False,
                'enableRateLimit': True,
            })
        except Exception as e:
            logger.warning(f"Exchange connection failed: {e}")
            self.exchange = None
        
        logger.info("🏛️ INSTITUTIONAL GRADE TRADER INITIALIZED")
        logger.info(f"   📊 Confluence Threshold: {self.confluence_threshold}%")
        logger.info(f"   🎯 Max Daily Signals: {self.max_daily_signals}")
        logger.info(f"   ⚖️ Risk Management: {self.base_risk_percent}%-{self.max_risk_percent}%")
        logger.info(f"   🎪 Min Risk-Reward: {self.min_risk_reward}:1")

    def fetch_multi_timeframe_data(self, symbol: str = 'BTC/USDT') -> Dict[str, pd.DataFrame]:
        """Fetch REAL data across all required timeframes using CCXT"""
        logger.info(f"📊 Fetching REAL multi-timeframe data for {symbol}...")
        
        data = {}
        
        try:
            for name, timeframe in self.timeframes.items():
                # Adjust limit based on timeframe for sufficient data
                if timeframe == '1w':
                    limit = 52  # 1 year of weekly data
                elif timeframe == '1d':
                    limit = 100  # 100 days of daily data
                elif timeframe == '4h':
                    limit = 168  # ~1 month of 4h data
                elif timeframe == '1h':
                    limit = 168  # 1 week of hourly data
                else:  # 15m
                    limit = 288  # 3 days of 15m data
                
                # Fetch real OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    logger.error(f"   ❌ No data received for {name} ({timeframe})")
                    continue
                
                # Create DataFrame with proper column specification
                df = pd.DataFrame(ohlcv)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add essential indicators
                df = self.add_institutional_indicators(df)
                data[name] = df
                
                logger.info(f"   ✅ {name.upper()} ({timeframe}): {len(df)} candles from {df.index[0]} to {df.index[-1]}")
                
        except Exception as e:
            logger.error(f"   ❌ Failed to fetch real data: {e}")
            logger.info("   🔄 Falling back to demo mode with realistic data...")
            return self.fetch_demo_data(symbol)
        
        return data
    
    def fetch_demo_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fallback demo data method"""
        logger.info("   📊 Generating demo data for institutional analysis...")
        
        # Use the existing mock data generation
        mock_data_sets = {
            'weekly': self.generate_mock_data('1w', 52),
            'daily': self.generate_mock_data('1d', 100), 
            'h4': self.generate_mock_data('4h', 168),
            'h1': self.generate_mock_data('1h', 168),
            'entry': self.generate_mock_data('15m', 288)
        }
        
        data = {}
        for name, mock_df in mock_data_sets.items():
            try:
                df = self.add_institutional_indicators(mock_df)
                data[name] = df
                logger.info(f"   ✅ {name.upper()}: {len(df)} demo candles loaded")
            except Exception as e:
                logger.error(f"   ❌ Failed to process {name} demo data: {e}")
                
        return data

    def generate_mock_data(self, timeframe: str, periods: int) -> pd.DataFrame:
        """Generate realistic mock market data for demonstration"""
        
        # Base price around current BTC price
        base_price = 97000
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
        if timeframe == '1w':
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='W')
        elif timeframe == '4h':
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='4H')
        elif timeframe == '1h':
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        elif timeframe == '15m':
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='15T')
        
        # Generate realistic price action
        np.random.seed(42)  # For consistent demo results
        
        prices = []
        current_price = base_price
        
        for i in range(periods):
            # Simulate realistic Bitcoin volatility
            change_pct = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price *= (1 + change_pct)
            prices.append(current_price)
        
        # Create OHLCV data
        df_data = []
        for i, (date, close_price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close
            volatility = close_price * 0.01  # 1% intraday volatility
            
            open_price = close_price + np.random.normal(0, volatility * 0.3)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility * 0.5))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility * 0.5))
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            volume = np.random.normal(1000, 200)  # Mock volume
            
            df_data.append({
                'timestamp': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': max(100, volume)  # Ensure positive volume
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        return df

    def add_institutional_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add institutional-grade technical indicators"""
        
        # **TREND INDICATORS**
        df['ema_21'] = ema_indicator(df['close'], window=21)
        df['ema_50'] = ema_indicator(df['close'], window=50)
        df['ema_200'] = ema_indicator(df['close'], window=200)
        
        # **MOMENTUM INDICATORS**
        df['rsi'] = rsi(df['close'], window=14)
        df['macd'] = macd_diff(df['close'])
        
        # **VOLATILITY INDICATORS**
        df['atr'] = average_true_range(df['high'], df['low'], df['close'], window=14)
        df['bb_upper'] = bollinger_hband(df['close'], window=20)
        df['bb_lower'] = bollinger_lband(df['close'], window=20)
        
        # **VOLUME INDICATORS**
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # **PRICE ACTION**
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=20).std()
        
        return df

    def analyze_primary_trend(self, weekly_data: pd.DataFrame, daily_data: pd.DataFrame) -> Dict:
        """Step 1: Analyze primary trend direction (Weekly/Daily)"""
        
        if len(weekly_data) < 10 or len(daily_data) < 50:
            return {'trend': 'unknown', 'confidence': 0, 'strength': 0}
        
        weekly_latest = weekly_data.iloc[-1]
        daily_latest = daily_data.iloc[-1]
        
        trend_signals = []
        
        # **WEEKLY TREND ANALYSIS**
        if weekly_latest['close'] > weekly_latest['ema_21']:
            trend_signals.append(('weekly_ema21', 'bullish', 25))
        else:
            trend_signals.append(('weekly_ema21', 'bearish', 25))
        
        # **DAILY TREND ANALYSIS**
        if daily_latest['close'] > daily_latest['ema_50']:
            trend_signals.append(('daily_ema50', 'bullish', 20))
        else:
            trend_signals.append(('daily_ema50', 'bearish', 20))
        
        # **MOVING AVERAGE ALIGNMENT**
        daily_ma_aligned = (daily_latest['ema_21'] > daily_latest['ema_50'] > daily_latest['ema_200'])
        if daily_ma_aligned:
            trend_signals.append(('ma_alignment', 'bullish', 15))
        elif daily_latest['ema_21'] < daily_latest['ema_50'] < daily_latest['ema_200']:
            trend_signals.append(('ma_alignment', 'bearish', 15))
        
        # **HIGHER HIGHS/LOWER LOWS**
        daily_recent = daily_data.tail(10)
        recent_highs = daily_recent['high'].rolling(3).max()
        recent_lows = daily_recent['low'].rolling(3).min()
        
        if recent_highs.iloc[-1] > recent_highs.iloc[-3]:
            trend_signals.append(('structure', 'bullish', 20))
        elif recent_lows.iloc[-1] < recent_lows.iloc[-3]:
            trend_signals.append(('structure', 'bearish', 20))
        
        # **MOMENTUM CONFIRMATION**
        if daily_latest['macd'] > 0:
            trend_signals.append(('momentum', 'bullish', 20))
        else:
            trend_signals.append(('momentum', 'bearish', 20))
        
        # **CALCULATE OVERALL TREND**
        bullish_score = sum(weight for signal, direction, weight in trend_signals if direction == 'bullish')
        bearish_score = sum(weight for signal, direction, weight in trend_signals if direction == 'bearish')
        total_possible = sum(weight for signal, direction, weight in trend_signals)
        
        if bullish_score > bearish_score:
            trend = 'bullish'
            confidence = (bullish_score / total_possible) * 100
            strength = min(1.0, bullish_score / 50)
        else:
            trend = 'bearish'  
            confidence = (bearish_score / total_possible) * 100
            strength = min(1.0, bearish_score / 50)
        
        return {
            'trend': trend,
            'confidence': confidence,
            'strength': strength,
            'signals': trend_signals,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score
        }

    def identify_institutional_levels(self, daily_data: pd.DataFrame) -> Dict:
        """Step 2: Identify key institutional levels (Order Blocks, FVGs)"""
        
        levels = {
            'order_blocks': [],
            'fair_value_gaps': [],
            'support_resistance': []
        }
        
        if len(daily_data) < 20:
            return levels
        
        # **ORDER BLOCK DETECTION**
        for i in range(10, len(daily_data)-1):
            current = daily_data.iloc[i]
            prev = daily_data.iloc[i-1]
            next_candle = daily_data.iloc[i+1]
            
            # Bullish Order Block: Strong green candle followed by continuation
            if (current['close'] > current['open'] and 
                (current['close'] - current['open']) > current['atr'] * 0.5 and
                next_candle['close'] > current['close']):
                
                levels['order_blocks'].append({
                    'type': 'bullish',
                    'price': current['low'],
                    'strength': (current['close'] - current['open']) / current['atr'],
                    'timestamp': current.name,
                    'volume_ratio': current['volume_ratio']
                })
            
            # Bearish Order Block: Strong red candle followed by continuation
            elif (current['close'] < current['open'] and 
                  (current['open'] - current['close']) > current['atr'] * 0.5 and
                  next_candle['close'] < current['close']):
                
                levels['order_blocks'].append({
                    'type': 'bearish',
                    'price': current['high'],
                    'strength': (current['open'] - current['close']) / current['atr'],
                    'timestamp': current.name,
                    'volume_ratio': current['volume_ratio']
                })
        
        # **FAIR VALUE GAP DETECTION**
        for i in range(2, len(daily_data)):
            candle1 = daily_data.iloc[i-2]
            candle2 = daily_data.iloc[i-1]  
            candle3 = daily_data.iloc[i]
            
            # Bullish FVG: Gap between candle1 high and candle3 low
            if candle1['high'] < candle3['low']:
                gap_size = candle3['low'] - candle1['high']
                if gap_size > candle2['atr'] * 0.3:  # Significant gap
                    levels['fair_value_gaps'].append({
                        'type': 'bullish',
                        'upper': candle3['low'],
                        'lower': candle1['high'],
                        'size': gap_size,
                        'timestamp': candle3.name
                    })
            
            # Bearish FVG: Gap between candle1 low and candle3 high
            elif candle1['low'] > candle3['high']:
                gap_size = candle1['low'] - candle3['high']
                if gap_size > candle2['atr'] * 0.3:
                    levels['fair_value_gaps'].append({
                        'type': 'bearish',
                        'upper': candle1['low'],
                        'lower': candle3['high'],
                        'size': gap_size,
                        'timestamp': candle3.name
                    })
        
        # **SUPPORT/RESISTANCE LEVELS**
        pivot_highs = []
        pivot_lows = []
        
        for i in range(5, len(daily_data)-5):
            window = daily_data.iloc[i-5:i+6]
            center = daily_data.iloc[i]
            
            # Pivot High
            if center['high'] == window['high'].max():
                pivot_highs.append({'price': center['high'], 'timestamp': center.name})
            
            # Pivot Low  
            if center['low'] == window['low'].min():
                pivot_lows.append({'price': center['low'], 'timestamp': center.name})
        
        levels['support_resistance'] = {'highs': pivot_highs, 'lows': pivot_lows}
        
        return levels

    def calculate_confluence_score(self, 
                                 primary_trend: Dict,
                                 institutional_levels: Dict,
                                 current_price: float,
                                 h4_data: pd.DataFrame,
                                 h1_data: pd.DataFrame) -> Dict:
        """Step 3: Calculate multi-timeframe confluence score"""
        
        confluence_factors = []
        
        # **1. PRIMARY TREND ALIGNMENT (25 points)**
        if primary_trend['confidence'] >= 70:
            confluence_factors.append(('primary_trend', 25 * (primary_trend['confidence'] / 100)))
        
        # **2. INSTITUTIONAL LEVEL PROXIMITY (20 points)**
        near_level = False
        level_strength = 0
        
        # Check Order Blocks
        for ob in institutional_levels['order_blocks']:
            distance_pct = abs(current_price - ob['price']) / current_price * 100
            if distance_pct <= 2.0:  # Within 2% of level
                near_level = True
                level_strength = max(level_strength, min(20, ob['strength'] * 10))
        
        # Check Fair Value Gaps
        for fvg in institutional_levels['fair_value_gaps']:
            if fvg['lower'] <= current_price <= fvg['upper']:
                near_level = True
                level_strength = max(level_strength, 15)
        
        if near_level:
            confluence_factors.append(('institutional_level', level_strength))
        
        # **3. MARKET STRUCTURE CONFIRMATION (15 points)**
        if len(h4_data) >= 5:
            h4_latest = h4_data.iloc[-1]
            h4_prev = h4_data.iloc[-2]
            
            # Break of Structure (BOS)
            if primary_trend['trend'] == 'bullish' and h4_latest['high'] > h4_data['high'].rolling(5).max().iloc[-2]:
                confluence_factors.append(('market_structure', 15))
            elif primary_trend['trend'] == 'bearish' and h4_latest['low'] < h4_data['low'].rolling(5).min().iloc[-2]:
                confluence_factors.append(('market_structure', 15))
        
        # **4. VOLUME CONFIRMATION (10 points)**
        if len(h1_data) >= 2:
            h1_latest = h1_data.iloc[-1]
            if h1_latest['volume_ratio'] >= 1.5:  # 50% above average volume
                confluence_factors.append(('volume_confirmation', 10))
        
        # **5. RSI NOT EXTREME (10 points)**
        if len(h4_data) >= 1:
            h4_rsi = h4_data.iloc[-1]['rsi']
            if 25 <= h4_rsi <= 75:  # Not in extreme zones
                confluence_factors.append(('rsi_not_extreme', 10))
        
        # **6. MOMENTUM ALIGNMENT (10 points)**
        if len(h4_data) >= 1:
            h4_macd = h4_data.iloc[-1]['macd']
            if (primary_trend['trend'] == 'bullish' and h4_macd > 0) or \
               (primary_trend['trend'] == 'bearish' and h4_macd < 0):
                confluence_factors.append(('momentum_alignment', 10))
        
        # **7. VOLATILITY ACCEPTABLE (10 points)**
        if len(h1_data) >= 1:
            current_volatility = h1_data.iloc[-1]['volatility']
            if current_volatility <= self.volatility_threshold:
                confluence_factors.append(('volatility_ok', 10))
        
        # **CALCULATE TOTAL CONFLUENCE**
        total_score = sum(score for factor, score in confluence_factors)
        max_possible = 100
        confluence_percentage = (total_score / max_possible) * 100
        
        return {
            'confluence_score': confluence_percentage,
            'factors': confluence_factors,
            'meets_threshold': confluence_percentage >= self.confluence_threshold,
            'primary_trend': primary_trend['trend'],
            'near_institutional_level': near_level
        }

    def generate_institutional_signal(self, data: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """Generate high-quality institutional signal"""
        
        logger.info("🔍 Analyzing market for institutional-grade opportunities...")
        
        # Get current price
        current_price = data['entry'].iloc[-1]['close']
        
        # **STEP 1: Primary Trend Analysis**
        primary_trend = self.analyze_primary_trend(data['weekly'], data['daily'])
        logger.info(f"   📈 Primary Trend: {primary_trend['trend'].upper()} ({primary_trend['confidence']:.1f}% confidence)")
        
        if primary_trend['confidence'] < 70:
            logger.info("   ❌ Primary trend confidence too low - NO TRADE")
            return None
        
        # **STEP 2: Institutional Level Analysis**
        institutional_levels = self.identify_institutional_levels(data['daily'])
        logger.info(f"   🏛️ Found {len(institutional_levels['order_blocks'])} Order Blocks, {len(institutional_levels['fair_value_gaps'])} FVGs")
        
        # **STEP 3: Confluence Analysis**
        confluence = self.calculate_confluence_score(
            primary_trend, institutional_levels, current_price,
            data['h4'], data['h1']
        )
        
        logger.info(f"   ⚖️ Confluence Score: {confluence['confluence_score']:.1f}%")
        
        if not confluence['meets_threshold']:
            logger.info(f"   ❌ Confluence below {self.confluence_threshold}% threshold - NO TRADE")
            return None
        
        # **STEP 4: Risk-Reward Validation**
        signal_direction = primary_trend['trend']
        entry_price = current_price
        
        # Calculate stop loss based on market structure
        if signal_direction == 'bullish':
            # Place stop below recent swing low or support level
            recent_lows = data['h4']['low'].tail(10)
            stop_loss = recent_lows.min() * 0.995  # 0.5% buffer
        else:
            # Place stop above recent swing high or resistance level
            recent_highs = data['h4']['high'].tail(10)
            stop_loss = recent_highs.max() * 1.005  # 0.5% buffer
        
        risk_amount = abs(entry_price - stop_loss)
        risk_pct = (risk_amount / entry_price) * 100
        
        # Calculate take profit (minimum 3:1 RR)
        if signal_direction == 'bullish':
            take_profit = entry_price + (risk_amount * self.min_risk_reward)
        else:
            take_profit = entry_price - (risk_amount * self.min_risk_reward)
        
        rr_ratio = abs(take_profit - entry_price) / risk_amount
        
        if rr_ratio < self.min_risk_reward:
            logger.info(f"   ❌ Risk-Reward {rr_ratio:.1f}:1 below minimum {self.min_risk_reward}:1 - NO TRADE")
            return None
        
        # **STEP 5: Position Sizing**
        risk_multiplier = 1.0
        if confluence['confluence_score'] >= 95:
            risk_multiplier = 2.0
        elif confluence['confluence_score'] >= 90:
            risk_multiplier = 1.5
        
        position_risk = min(self.max_risk_percent, self.base_risk_percent * risk_multiplier) / 100
        position_size = (self.current_capital * position_risk) / risk_amount
        
        # **GENERATE SIGNAL**
        signal = {
            'symbol': 'BTC/USDT',
            'direction': signal_direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_pct': risk_pct,
            'rr_ratio': rr_ratio,
            'confluence_score': confluence['confluence_score'],
            'primary_trend': primary_trend,
            'confluence_factors': confluence['factors'],
            'timestamp': datetime.now(),
            'timeframe': 'Multi-timeframe analysis'
        }
        
        logger.info("🎯 HIGH-QUALITY INSTITUTIONAL SIGNAL GENERATED:")
        logger.info(f"   📊 {signal_direction.upper()} {signal['symbol']} at ${entry_price:.2f}")
        logger.info(f"   ⚖️ Confluence: {confluence['confluence_score']:.1f}%")
        logger.info(f"   🛡️ Stop Loss: ${stop_loss:.2f} (-{risk_pct:.2f}%)")
        logger.info(f"   🎯 Take Profit: ${take_profit:.2f} (+{rr_ratio:.1f}:1 RR)")
        logger.info(f"   💰 Position Size: {position_size:.4f} units (${position_size * entry_price:.2f})")
        logger.info(f"   🎲 Risk: {risk_pct:.2f}% of capital")
        
        return signal

    def run_institutional_analysis(self, symbol: str = 'BTC/USDT') -> Optional[Dict]:
        """Run complete institutional-grade analysis"""
        
        logger.info("🏛️ STARTING INSTITUTIONAL-GRADE MARKET ANALYSIS")
        logger.info("=" * 80)
        
        # Fetch multi-timeframe data
        data = self.fetch_multi_timeframe_data(symbol)
        if not data:
            logger.error("❌ Failed to fetch market data")
            return None
        
        # Generate institutional signal
        signal = self.generate_institutional_signal(data)
        
        if signal:
            logger.info("✅ INSTITUTIONAL ANALYSIS COMPLETE - SIGNAL GENERATED")
        else:
            logger.info("⏳ INSTITUTIONAL ANALYSIS COMPLETE - WAITING FOR QUALITY SETUP")
        
        return signal


def main():
    """Main execution function"""
    print("🏛️ INSTITUTIONAL GRADE SMART MONEY CONCEPTS TRADER")
    print("=" * 80)
    print("🎯 QUALITY OVER QUANTITY - Professional Trading System")
    print("📊 Multi-timeframe confluence analysis")
    print("⚖️ 85%+ confluence requirement")
    print("🛡️ Professional risk management")
    print("🎪 Minimum 3:1 risk-reward ratios")
    print("📈 2-5 high-quality signals per day maximum")
    print()
    
    # Initialize institutional trader
    trader = InstitutionalGradeTrader()
    
    # Run analysis
    signal = trader.run_institutional_analysis('BTC/USDT')
    
    if signal:
        print("\n🎊 INSTITUTIONAL-GRADE SIGNAL ANALYSIS COMPLETE!")
        print("This is how professional traders analyze markets.")
    else:
        print("\n⏳ No quality setup found - PATIENCE IS KEY in institutional trading.")
        print("Professional traders wait for high-probability opportunities.")

if __name__ == "__main__":
    main()