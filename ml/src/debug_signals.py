#!/usr/bin/env python3
"""
Debug script to identify why enhanced signals are not generating
"""

import pandas as pd
import numpy as np
from features.build_features import build_features
from features.generate_signals import generate_signals, check_confluence
import sys
import os

def create_simple_trending_data():
    """Create simple trending data that should definitely generate signals."""
    
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    
    # Create a clear uptrend
    base_price = 50000
    trend = np.linspace(0, 0.1, 100)  # 10% uptrend
    noise = np.random.normal(0, 0.01, 100)  # 1% noise
    
    prices = base_price * (1 + trend + noise)
    
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = np.maximum(df['open'], df['close']) * 1.01
    df['low'] = np.minimum(df['open'], df['close']) * 0.99
    df['volume'] = np.random.uniform(1000, 5000, 100)
    
    return df

def debug_features(df, features_df):
    """Debug the features to see what's happening."""
    
    print("DEBUGGING FEATURES:")
    print("="*50)
    
    print(f"Data shape: {df.shape}")
    print(f"Features shape: {features_df.shape}")
    
    print(f"\nRegime values: {features_df['regime'].value_counts()}")
    print(f"Bias values: {features_df['bias'].value_counts()}")
    print(f"Order block values: {features_df['order_block'].value_counts()}")
    print(f"Volume expansion: {features_df['vol_expansion'].sum()} periods")
    print(f"Pivot highs: {features_df['pivot_high'].sum()} detected")
    print(f"Pivot lows: {features_df['pivot_low'].sum()} detected")
    
    # Show some sample rows
    print(f"\nSample features (first 5 rows):")
    print(features_df.head())
    
    return features_df

def debug_confluence(features_df, idx=30):
    """Debug confluence checking logic."""
    
    print(f"\nDEBUGGING CONFLUENCE at index {idx}:")
    print("="*50)
    
    if idx >= len(features_df):
        print("Index out of range")
        return False
        
    current = features_df.iloc[idx]
    
    print(f"Current row values:")
    print(f"  bias: {current['bias']}")
    print(f"  vol_expansion: {current['vol_expansion']}")
    print(f"  order_block: {current['order_block']}")
    print(f"  london_session: {current['london_session']}")
    print(f"  ny_session: {current['ny_session']}")
    print(f"  price_change: {current['price_change']}")
    
    # Test buy confluence
    confluence_count = 0
    
    # HTF bias must be bullish
    if current['bias'] == 'bullish':
        confluence_count += 1
        print(f"✓ Bullish bias (+1)")
    else:
        print(f"✗ Bias not bullish: {current['bias']}")
    
    # Volume expansion confirmation
    if current['vol_expansion']:
        confluence_count += 1
        print(f"✓ Volume expansion (+1)")
    else:
        print(f"✗ No volume expansion")
        
    # Order block confirmation
    if current['order_block'] == 1:
        confluence_count += 1
        print(f"✓ Bullish order block (+1)")
    else:
        print(f"✗ No bullish order block: {current['order_block']}")
        
    # Session timing (London/NY)
    if current['london_session'] or current['ny_session']:
        confluence_count += 1
        print(f"✓ Trading session (+1)")
    else:
        print(f"✗ Not in trading session")
    
    print(f"\nTotal confluence score: {confluence_count}/4")
    print(f"Required: 3+ for signal")
    
    return confluence_count >= 3

def simplified_signal_test(df, features_df):
    """Test with simplified signal logic."""
    
    print("\nTESTING SIMPLIFIED SIGNALS:")
    print("="*50)
    
    signals = pd.Series(0, index=features_df.index)
    
    for i in range(20, len(features_df)):
        current = features_df.iloc[i]
        
        # Very simple bullish condition
        if (current['bias'] == 'bullish' and 
            current['regime'] == 'trending' and
            current['vol_expansion']):
            signals.iloc[i] = 1
            print(f"Signal at index {i}: {features_df.index[i]}")
    
    buy_signals = (signals == 1).sum()
    print(f"\nSimplified signals generated: {buy_signals}")
    
    return signals

def main():
    """Main debug function."""
    
    print("DEBUGGING ENHANCED SIGNAL GENERATION")
    print("="*50)
    
    # Create simple trending data
    df = create_simple_trending_data()
    print(f"Created trending data: {len(df)} periods")
    print(f"Price trend: {df['close'].iloc[0]:.2f} -> {df['close'].iloc[-1]:.2f}")
    
    # Build features
    features_df = build_features(df)
    print(f"Built features: {len(features_df)} periods")
    
    # Debug features
    debug_features(df, features_df)
    
    # Debug confluence for a few samples
    for idx in [20, 30, 40]:
        if idx < len(features_df):
            confluence_result = debug_confluence(features_df, idx)
            print(f"Confluence at {idx}: {confluence_result}")
    
    # Test simplified signals
    simplified_signals = simplified_signal_test(df, features_df)
    
    # Test original signals
    print("\nTESTING ORIGINAL SIGNAL GENERATION:")
    print("="*50)
    
    try:
        signals_df = generate_signals(df, features_df)
        print(f"Original signals shape: {signals_df.shape}")
        
        for strategy in signals_df.columns:
            total_signals = (signals_df[strategy] != 0).sum()
            print(f"{strategy}: {total_signals} signals")
            
    except Exception as e:
        print(f"Error in original signal generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 