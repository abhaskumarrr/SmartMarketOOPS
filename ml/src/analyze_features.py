#!/usr/bin/env python3

"""
Analyze Enhanced Features
Debug why sophisticated strategies aren't generating signals
"""

import pandas as pd
import numpy as np
import ccxt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our enhanced modules
from features.build_features import build_features

def analyze_features_debug():
    """Analyze feature values to understand signal generation constraints."""
    
    # Download real data
    exchange = ccxt.binance({'enableRateLimit': True})
    exchange.load_markets()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=200)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print(f"📊 Real BTC Data Analysis ({len(df)} candles)")
    print(f"📈 Price: ${df['close'].iloc[-1]:.2f} (Range: ${df['close'].min():.0f} - ${df['close'].max():.0f})")
    
    # Build enhanced features
    features = build_features(df)
    print(f"✅ Features built: {features.shape}")
    
    # Analyze key constraints for signal generation
    print("\n🔍 SIGNAL CONSTRAINT ANALYSIS:")
    
    # 1. Market Regime Analysis
    regime_counts = features['market_regime'].value_counts()
    print(f"\n📊 Market Regimes:")
    for regime, count in regime_counts.items():
        print(f"  • {regime}: {count} periods ({count/len(features)*100:.1f}%)")
    
    # 2. HTF Bias Analysis  
    bias_counts = features['bias'].value_counts()
    print(f"\n📊 HTF Bias Distribution:")
    for bias, count in bias_counts.items():
        print(f"  • {bias}: {count} periods ({count/len(features)*100:.1f}%)")
    
    # 3. Volume Expansion Analysis
    vol_expansion_count = features['vol_expansion'].sum()
    print(f"\n📊 Volume Analysis:")
    print(f"  • Volume expansion periods: {vol_expansion_count} ({vol_expansion_count/len(features)*100:.1f}%)")
    
    # 4. Session Analysis
    session_counts = features['session'].value_counts()
    print(f"\n📊 Trading Sessions:")
    for session, count in session_counts.items():
        print(f"  • {session}: {count} periods ({count/len(features)*100:.1f}%)")
    
    key_session_count = features['is_key_session'].sum()
    print(f"  • Key sessions (London/NY): {key_session_count} ({key_session_count/len(features)*100:.1f}%)")
    
    # 5. Confluence Analysis
    confluence_stats = {
        'bullish_0': len(features[features['confluence_bullish'] == 0]),
        'bullish_1': len(features[features['confluence_bullish'] == 1]),
        'bullish_2+': len(features[features['confluence_bullish'] >= 2]),
        'bullish_3+': len(features[features['confluence_bullish'] >= 3]),
        'bearish_0': len(features[features['confluence_bearish'] == 0]),
        'bearish_1': len(features[features['confluence_bearish'] == 1]),
        'bearish_2+': len(features[features['confluence_bearish'] >= 2]),
        'bearish_3+': len(features[features['confluence_bearish'] >= 3])
    }
    
    print(f"\n📊 Confluence Analysis:")
    print(f"  • Bullish confluences 2+: {confluence_stats['bullish_2+']} ({confluence_stats['bullish_2+']/len(features)*100:.1f}%)")
    print(f"  • Bullish confluences 3+: {confluence_stats['bullish_3+']} ({confluence_stats['bullish_3+']/len(features)*100:.1f}%)")
    print(f"  • Bearish confluences 2+: {confluence_stats['bearish_2+']} ({confluence_stats['bearish_2+']/len(features)*100:.1f}%)")
    print(f"  • Bearish confluences 3+: {confluence_stats['bearish_3+']} ({confluence_stats['bearish_3+']/len(features)*100:.1f}%)")
    
    # 6. Order Block Analysis
    ob_support_count = features['order_block_support'].sum()
    ob_resistance_count = features['order_block_resistance'].sum()
    print(f"\n📊 Order Block Analysis:")
    print(f"  • Support order blocks: {ob_support_count} ({ob_support_count/len(features)*100:.1f}%)")
    print(f"  • Resistance order blocks: {ob_resistance_count} ({ob_resistance_count/len(features)*100:.1f}%)")
    
    # 7. Combined Signal Potential Analysis
    print(f"\n🎯 SIGNAL POTENTIAL ANALYSIS:")
    
    # Scalping potential (trending + bias + volume + session + confluence 2+)
    scalping_bullish = len(features[
        (features['market_regime'].isin(['trending_up', 'trending_down'])) &
        (features['bias'] == 'bullish') &
        (features['vol_expansion'] == True) &
        (features['is_key_session'] == True) &
        (features['confluence_bullish'] >= 2)
    ])
    
    scalping_bearish = len(features[
        (features['market_regime'].isin(['trending_up', 'trending_down'])) &
        (features['bias'] == 'bearish') &
        (features['vol_expansion'] == True) &
        (features['is_key_session'] == True) &
        (features['confluence_bearish'] >= 2)
    ])
    
    print(f"  • Scalping potential (bullish): {scalping_bullish} periods")
    print(f"  • Scalping potential (bearish): {scalping_bearish} periods")
    
    # Day trade potential (strictest requirements)
    day_trade_bullish = len(features[
        (features['bias'] == 'bullish') &
        (features['market_regime'] == 'trending_up') &
        (features['confluence_bullish'] >= 3) &
        (features['is_key_session'] == True) &
        (features['vol_expansion'] == True) &
        (features['order_block_support'] == True)
    ])
    
    day_trade_bearish = len(features[
        (features['bias'] == 'bearish') &
        (features['market_regime'] == 'trending_down') &
        (features['confluence_bearish'] >= 3) &
        (features['is_key_session'] == True) &
        (features['vol_expansion'] == True) &
        (features['order_block_resistance'] == True)
    ])
    
    print(f"  • Day trade potential (bullish): {day_trade_bullish} periods")
    print(f"  • Day trade potential (bearish): {day_trade_bearish} periods")
    
    # 8. Specific Recent Analysis
    print(f"\n📅 RECENT 20 PERIODS ANALYSIS:")
    recent = features.tail(20)
    
    recent_trending = len(recent[recent['market_regime'].isin(['trending_up', 'trending_down'])])
    recent_bullish_bias = len(recent[recent['bias'] == 'bullish'])
    recent_bearish_bias = len(recent[recent['bias'] == 'bearish'])
    recent_vol_expansion = recent['vol_expansion'].sum()
    recent_key_sessions = recent['is_key_session'].sum()
    
    print(f"  • Trending periods: {recent_trending}/20")
    print(f"  • Bullish bias: {recent_bullish_bias}/20")
    print(f"  • Bearish bias: {recent_bearish_bias}/20") 
    print(f"  • Volume expansions: {recent_vol_expansion}/20")
    print(f"  • Key sessions: {recent_key_sessions}/20")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if scalping_bullish + scalping_bearish == 0:
        print("  • Scalping: Reduce confluence requirement from 2+ to 1+")
    if day_trade_bullish + day_trade_bearish == 0:
        print("  • Day trading: Reduce confluence requirement from 3+ to 2+")
    if recent_vol_expansion < 5:
        print("  • Volume: Consider relaxing volume expansion criteria")
    if recent_key_sessions < 5:
        print("  • Sessions: Consider adding Asian session trading")
    if recent_trending < 10:
        print("  • Regime: Consider allowing range-bound trading strategies")
    
    return features

if __name__ == "__main__":
    features = analyze_features_debug() 