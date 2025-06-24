#!/usr/bin/env python3
"""
FIXED INSTITUTIONAL TRADER
Corrected confluence calculations and realistic thresholds
Based on diagnostic findings - fixes the broken trend/S&R detection
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedInstitutionalTrader:
    """Fixed institutional trader with proper confluence calculations"""
    
    def __init__(self):
        self.exchange = ccxt.binance()
        
        # Realistic confluence threshold based on diagnostic
        self.min_confluence = 70  # Reduced from 85% to realistic 70%
        
        # Enhanced confluence factors
        self.confluence_weights = {
            'trend_alignment': 25,      # Multi-timeframe trend (most important)
            'momentum': 20,             # RSI/MACD signals  
            'volume': 15,               # Volume confirmation
            'support_resistance': 20,   # Key S/R levels
            'volatility': 10,           # Market volatility
            'price_action': 10          # Price action patterns
        }
        
        self.timeframes = {
            'daily': '1d',
            'h4': '4h', 
            'h1': '1h',
            'm15': '15m'
        }
        
    def fetch_multi_timeframe_data(self, symbol: str = 'BTC/USDT') -> Dict[str, pd.DataFrame]:
        """Fetch real data across timeframes with proper indicators"""
        logger.info(f"📊 Fetching FIXED multi-timeframe data for {symbol}...")
        
        data = {}
        
        try:
            for name, timeframe in self.timeframes.items():
                if timeframe == '1d':
                    limit = 100
                elif timeframe == '4h':
                    limit = 168  # 1 month
                elif timeframe == '1h': 
                    limit = 168  # 1 week
                elif timeframe == '15m':
                    limit = 96   # 1 day
                    
                logger.info(f"Fetching {name} ({timeframe}) data...")
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add comprehensive technical indicators
                self.add_technical_indicators(df)
                
                data[name] = df
                logger.info(f"✅ {name}: {len(df)} candles loaded")
                
            return data
            
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            return {}
    
    def add_technical_indicators(self, df: pd.DataFrame):
        """Add proper technical indicators to dataframe"""
        # Trend indicators
        df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        
        # Momentum indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        
        # Volume indicators  
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Support/Resistance levels
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot'] - df['low']
        df['support_1'] = 2 * df['pivot'] - df['high']
        
        # Price action patterns
        df['price_range'] = ((df['high'] - df['low']) / df['close']) * 100
        df['body_size'] = abs(df['close'] - df['open']) / df['close'] * 100
        
        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
        df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
        
    def calculate_fixed_confluence(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate confluence with FIXED logic"""
        
        # Get current values from each timeframe
        current_daily = data['daily'].iloc[-1]
        current_h4 = data['h4'].iloc[-1] 
        current_h1 = data['h1'].iloc[-1]
        current_m15 = data['m15'].iloc[-1]
        
        confluence_scores = {}
        
        # 1. FIXED TREND ALIGNMENT (25 points)
        trend_score = 0
        
        # Daily trend
        daily_trend = 0
        if len(data['daily']) >= 50:
            if current_daily['close'] > current_daily['sma_20'] > current_daily['sma_50']:
                daily_trend = 1  # Bullish
            elif current_daily['close'] < current_daily['sma_20'] < current_daily['sma_50']:
                daily_trend = -1  # Bearish
                
        # 4H trend
        h4_trend = 0
        if len(data['h4']) >= 20:
            if current_h4['close'] > current_h4['sma_20']:
                h4_trend = 1
            elif current_h4['close'] < current_h4['sma_20']:
                h4_trend = -1
                
        # 1H trend  
        h1_trend = 0
        if len(data['h1']) >= 20:
            if current_h1['close'] > current_h1['ema_20']:
                h1_trend = 1
            elif current_h1['close'] < current_h1['ema_20']:
                h1_trend = -1
        
        # Award points for trend alignment
        if daily_trend != 0 and h4_trend != 0 and h1_trend != 0:
            if daily_trend == h4_trend == h1_trend:  # All aligned
                trend_score = 25
            elif daily_trend == h4_trend or daily_trend == h1_trend:  # Partial alignment
                trend_score = 15
            else:  # Mixed signals
                trend_score = 5
        elif daily_trend != 0 or h4_trend != 0:  # Some trend detected
            trend_score = 10
            
        confluence_scores['trend_alignment'] = trend_score
        
        # 2. FIXED MOMENTUM (20 points)
        momentum_score = 0
        
        # RSI analysis
        rsi_score = 0
        if not pd.isna(current_daily['rsi']):
            rsi = current_daily['rsi']
            if 40 <= rsi <= 60:  # Neutral zone - good for entries
                rsi_score = 10
            elif 30 <= rsi <= 70:  # Acceptable zone
                rsi_score = 7
            elif rsi < 30 or rsi > 70:  # Extreme zones - reversal potential
                rsi_score = 5
                
        # MACD analysis
        macd_score = 0
        if not pd.isna(current_daily['macd']) and not pd.isna(current_daily['macd_signal']):
            if current_daily['macd'] > current_daily['macd_signal']:  # Bullish crossover
                macd_score = 10
            elif current_daily['macd'] < current_daily['macd_signal']:  # Bearish crossover
                macd_score = 10
            else:
                macd_score = 5
                
        momentum_score = rsi_score + macd_score
        confluence_scores['momentum'] = min(momentum_score, 20)  # Cap at 20
        
        # 3. FIXED VOLUME (15 points) 
        volume_score = 0
        if len(data['daily']) >= 10:
            current_volume = current_daily['volume']
            avg_volume = data['daily']['volume'].iloc[-10:].mean()
            
            if current_volume > avg_volume * 2:  # Very high volume
                volume_score = 15
            elif current_volume > avg_volume * 1.5:  # High volume
                volume_score = 12
            elif current_volume > avg_volume:  # Above average
                volume_score = 8
            else:  # Below average
                volume_score = 3
                
        confluence_scores['volume'] = volume_score
        
        # 4. FIXED SUPPORT/RESISTANCE (20 points)
        sr_score = 0
        if not pd.isna(current_daily['support_1']) and not pd.isna(current_daily['resistance_1']):
            current_price = current_daily['close']
            support = current_daily['support_1']
            resistance = current_daily['resistance_1']
            
            # Distance from S/R levels
            dist_to_support = abs(current_price - support) / current_price * 100
            dist_to_resistance = abs(current_price - resistance) / current_price * 100
            
            if dist_to_support < 1 or dist_to_resistance < 1:  # Very close to key level
                sr_score = 20
            elif dist_to_support < 2 or dist_to_resistance < 2:  # Close to key level
                sr_score = 15
            elif dist_to_support < 5 or dist_to_resistance < 5:  # Moderate distance
                sr_score = 10
            else:  # Far from levels
                sr_score = 5
                
        confluence_scores['support_resistance'] = sr_score
        
        # 5. VOLATILITY (10 points)
        volatility_score = 10  # Default good score
        if not pd.isna(current_daily['bb_upper']) and not pd.isna(current_daily['bb_lower']):
            current_price = current_daily['close']
            bb_upper = current_daily['bb_upper']
            bb_lower = current_daily['bb_lower']
            
            if bb_lower <= current_price <= bb_upper:  # Normal volatility
                volatility_score = 10
            else:  # High volatility - be cautious
                volatility_score = 5
                
        confluence_scores['volatility'] = volatility_score
        
        # 6. PRICE ACTION (10 points)
        price_action_score = 0
        
        # Look for strong candle patterns
        if not pd.isna(current_daily['body_size']):
            body_size = current_daily['body_size']
            if body_size > 2:  # Strong directional move
                price_action_score = 10
            elif body_size > 1:  # Moderate move
                price_action_score = 7
            else:  # Weak move
                price_action_score = 3
                
        confluence_scores['price_action'] = price_action_score
        
        # Calculate totals
        total_confluence = sum(confluence_scores.values())
        confluence_percentage = total_confluence  # Already out of 100
        
        return {
            'scores': confluence_scores,
            'total': total_confluence,
            'percentage': confluence_percentage,
            'meets_threshold': confluence_percentage >= self.min_confluence,
            'threshold': self.min_confluence
        }
    
    def run_fixed_backtest(self, symbol: str = 'BTC/USDT', days: int = 30):
        """Run backtest with fixed confluence logic"""
        logger.info(f"🚀 Running FIXED INSTITUTIONAL BACKTEST ({days} days)")
        logger.info("="*60)
        
        # Get daily data for the period
        daily_data = self.exchange.fetch_ohlcv(symbol, '1d', limit=days + 20)
        df_daily = pd.DataFrame(daily_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], unit='ms')
        df_daily.set_index('timestamp', inplace=True)
        
        # Add indicators
        self.add_technical_indicators(df_daily)
        
        signals = []
        trades = []
        
        # Analyze each day in the backtest period
        for i in range(20, len(df_daily)):  # Start from day 20 to have enough data for indicators
            current_day = df_daily.iloc[i]
            date_str = current_day.name.strftime('%Y-%m-%d')
            
            # Create mock multi-timeframe data for this day
            mock_data = {
                'daily': df_daily.iloc[max(0, i-50):i+1],
                'h4': df_daily.iloc[max(0, i-20):i+1],  # Approximate
                'h1': df_daily.iloc[max(0, i-10):i+1],  # Approximate
                'm15': df_daily.iloc[max(0, i-5):i+1]   # Approximate
            }
            
            # Calculate confluence for this day
            confluence = self.calculate_fixed_confluence(mock_data)
            
            analysis = {
                'date': date_str,
                'price': current_day['close'],
                'confluence': confluence['percentage'],
                'signal_generated': confluence['meets_threshold'],
                'confluence_breakdown': confluence['scores']
            }
            
            signals.append(analysis)
            
            # Generate trade if signal triggered
            if confluence['meets_threshold']:
                # Determine direction
                if (current_day['close'] > current_day['sma_20'] and 
                    current_day['rsi'] > 50):
                    trade = {
                        'date': date_str,
                        'direction': 'LONG',
                        'entry_price': current_day['close'],
                        'confluence': confluence['percentage'],
                        'risk_percent': 1.5
                    }
                    trades.append(trade)
                    logger.info(f"🟢 LONG Signal: {date_str} @ ${current_day['close']:,.2f} ({confluence['percentage']:.1f}% confluence)")
                    
                elif (current_day['close'] < current_day['sma_20'] and 
                      current_day['rsi'] < 50):
                    trade = {
                        'date': date_str,
                        'direction': 'SHORT',
                        'entry_price': current_day['close'],
                        'confluence': confluence['percentage'],
                        'risk_percent': 1.5
                    }
                    trades.append(trade)
                    logger.info(f"🔴 SHORT Signal: {date_str} @ ${current_day['close']:,.2f} ({confluence['percentage']:.1f}% confluence)")
        
        # Results summary
        total_signals = len([s for s in signals if s['signal_generated']])
        avg_confluence = np.mean([s['confluence'] for s in signals])
        max_confluence = max([s['confluence'] for s in signals])
        
        results = {
            'backtest_period': f"{days} days",
            'total_days_analyzed': len(signals),
            'total_signals': total_signals,
            'signal_rate': f"{(total_signals/len(signals)*100):.1f}%",
            'average_confluence': f"{avg_confluence:.1f}%",
            'maximum_confluence': f"{max_confluence:.1f}%",
            'threshold_used': f"{self.min_confluence}%",
            'trades_generated': len(trades),
            'signals': signals,
            'trades': trades
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"fixed_institutional_backtest_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Print summary
        print(f"\n🏛️ FIXED INSTITUTIONAL BACKTEST RESULTS")
        print("="*50)
        print(f"📊 Period: {results['backtest_period']}")
        print(f"🎯 Signals Generated: {results['total_signals']}")
        print(f"📈 Signal Rate: {results['signal_rate']}")
        print(f"⚡ Average Confluence: {results['average_confluence']}")
        print(f"🔥 Maximum Confluence: {results['maximum_confluence']}")
        print(f"🎲 Threshold: {results['threshold_used']}")
        print(f"💰 Trades: {results['trades_generated']}")
        print(f"\n📋 Results saved to: {filename}")
        
        return results

def main():
    """Run the fixed institutional trader"""
    trader = FixedInstitutionalTrader()
    
    # Run backtest
    trader.run_fixed_backtest('BTC/USDT', 30)

if __name__ == "__main__":
    main()