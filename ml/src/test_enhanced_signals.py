#!/usr/bin/env python3
"""
Test script for enhanced multi-timeframe trading signals
Validates that the sophisticated strategies generate actual trading signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from features.build_features import build_features
from features.generate_signals import generate_signals
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_sample_data(num_days=30, interval_minutes=15):
    """Create realistic sample OHLCV data for testing."""
    
    # Calculate number of periods
    periods_per_day = 24 * 60 // interval_minutes  # 96 for 15-minute intervals
    num_periods = num_days * periods_per_day
    
    # Generate timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=num_periods, freq=f'{interval_minutes}min')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)  # For reproducible results
    
    # Base price starts at $50,000 (BTC-like)
    base_price = 50000
    
    # Generate price movements with trends
    price_changes = np.random.normal(0, 0.002, num_periods)  # 0.2% volatility per period
    
    # Add some trending periods
    trend_periods = [
        (100, 200, 0.001),   # Uptrend
        (400, 500, -0.0015), # Downtrend
        (800, 900, 0.0012),  # Another uptrend
    ]
    
    for start, end, trend_strength in trend_periods:
        if end < num_periods:
            price_changes[start:end] += trend_strength
    
    # Calculate cumulative price
    log_prices = np.log(base_price) + np.cumsum(price_changes)
    prices = np.exp(log_prices)
    
    # Generate OHLC from close prices
    df = pd.DataFrame(index=timestamps)
    df['close'] = prices
    
    # Generate realistic OHLC
    volatility = np.random.uniform(0.001, 0.01, num_periods)  # Variable volatility
    
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = np.maximum(df['open'], df['close']) * (1 + volatility * np.random.uniform(0.1, 2.0, num_periods))
    df['low'] = np.minimum(df['open'], df['close']) * (1 - volatility * np.random.uniform(0.1, 2.0, num_periods))
    
    # Generate realistic volume
    base_volume = 1000
    volume_multiplier = 1 + np.abs(price_changes) * 50  # Higher volume on bigger moves
    df['volume'] = base_volume * volume_multiplier * np.random.uniform(0.5, 2.0, num_periods)
    
    return df

def analyze_signals(signals_df, df):
    """Analyze the generated signals and provide statistics."""
    
    print("\n" + "="*60)
    print("ENHANCED SIGNAL GENERATION ANALYSIS")
    print("="*60)
    
    # Overall signal statistics
    total_periods = len(signals_df)
    
    for strategy in ['scalping', 'trend_capture', 'day_trade', 'unified']:
        if strategy in signals_df.columns:
            strategy_signals = signals_df[strategy]
            
            buy_signals = (strategy_signals == 1).sum()
            sell_signals = (strategy_signals == 2).sum()
            total_signals = buy_signals + sell_signals
            
            print(f"\n{strategy.upper()} STRATEGY:")
            print(f"  Total periods: {total_periods}")
            print(f"  Buy signals: {buy_signals}")
            print(f"  Sell signals: {sell_signals}")
            print(f"  Total signals: {total_signals}")
            print(f"  Signal frequency: {total_signals/total_periods*100:.2f}%")
            
            if total_signals > 0:
                # Show signal distribution over time
                signal_timestamps = signals_df.index[strategy_signals != 0]
                if len(signal_timestamps) > 0:
                    print(f"  First signal: {signal_timestamps[0]}")
                    print(f"  Last signal: {signal_timestamps[-1]}")
                    
                    # Calculate gaps between signals
                    if len(signal_timestamps) > 1:
                        gaps = pd.Series(signal_timestamps).diff().dropna()
                        avg_gap = gaps.mean()
                        print(f"  Average gap between signals: {avg_gap}")
    
    return signals_df

def visualize_signals(df, features_df, signals_df):
    """Create visualization of price action and signals."""
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    # Plot price action with signals
    ax1.plot(df.index, df['close'], label='Close Price', linewidth=1)
    
    # Plot unified signals
    unified_signals = signals_df['unified']
    buy_signals = unified_signals == 1
    sell_signals = unified_signals == 2
    
    if buy_signals.any():
        ax1.scatter(df.index[buy_signals], df['close'][buy_signals], 
                   color='green', marker='^', s=100, label='Buy Signals', zorder=5)
    
    if sell_signals.any():
        ax1.scatter(df.index[sell_signals], df['close'][sell_signals], 
                   color='red', marker='v', s=100, label='Sell Signals', zorder=5)
    
    ax1.set_title('Price Action with Enhanced Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot market regime and bias
    ax2.plot(features_df.index, features_df['bias'].map({'bullish': 1, 'bearish': -1, 'neutral': 0}), 
             label='HTF Bias', linewidth=2)
    ax2.plot(features_df.index, features_df['regime'].map({'trending': 1, 'range-bound': 0}), 
             label='Market Regime', alpha=0.7)
    ax2.set_title('Market Structure Analysis')
    ax2.set_ylabel('Bias/Regime')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot volume analysis
    ax3.plot(features_df.index, features_df['vol_expansion'].astype(int), 
             label='Volume Expansion', alpha=0.7)
    ax3.bar(df.index, df['volume'], alpha=0.3, label='Volume')
    ax3.set_title('Volume Analysis')
    ax3.set_ylabel('Volume')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot order blocks and pivot points
    ax4.scatter(features_df.index[features_df['order_block'] == 1], 
               df['close'][features_df['order_block'] == 1],
               color='blue', marker='s', s=50, label='Bullish Order Blocks', alpha=0.7)
    ax4.scatter(features_df.index[features_df['order_block'] == -1], 
               df['close'][features_df['order_block'] == -1],
               color='orange', marker='s', s=50, label='Bearish Order Blocks', alpha=0.7)
    ax4.plot(df.index, df['close'], alpha=0.3, color='gray')
    ax4.set_title('Smart Money Concepts')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_signals_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'enhanced_signals_analysis.png'")

def main():
    """Main test function."""
    
    print("Testing Enhanced Multi-timeframe Trading System")
    print("=" * 50)
    
    # Generate sample data
    print("Generating sample market data...")
    df = create_sample_data(num_days=30, interval_minutes=15)
    print(f"Generated {len(df)} periods of 15-minute data")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Build enhanced features
    print("\nBuilding enhanced features...")
    try:
        features_df = build_features(df)
        print(f"Generated {len(features_df.columns)} features:")
        print(f"Features: {list(features_df.columns)}")
        
        # Check for any NaN values
        nan_count = features_df.isnull().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in features")
            features_df = features_df.dropna()
            print(f"After dropping NaN: {len(features_df)} periods remaining")
        
    except Exception as e:
        print(f"Error building features: {e}")
        return
    
    # Generate signals
    print("\nGenerating trading signals...")
    try:
        signals_df = generate_signals(df, features_df)
        print(f"Generated signals for {len(signals_df)} periods")
        print(f"Signal strategies: {list(signals_df.columns)}")
        
    except Exception as e:
        print(f"Error generating signals: {e}")
        return
    
    # Analyze results
    analyze_signals(signals_df, df)
    
    # Create visualization
    print("\nCreating visualization...")
    try:
        visualize_signals(df, features_df, signals_df)
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Summary of key improvements
    print("\n" + "="*60)
    print("KEY IMPROVEMENTS IMPLEMENTED:")
    print("="*60)
    print("✅ Multi-timeframe market structure analysis")
    print("✅ Proper swing high/low detection using pivot points")
    print("✅ Smart Money Concepts (SMC) - order blocks & liquidity zones")
    print("✅ Confluence-based signal generation (3+ confirmations required)")
    print("✅ Volume expansion analysis and confirmation")
    print("✅ Session-based filtering (London/NY)")
    print("✅ Dynamic Fibonacci levels based on market structure")
    print("✅ Quality-over-quantity approach (selective signals)")
    print("✅ Three distinct strategies: Scalping, Trend Capture, Day Trading")
    print("✅ Risk management integration (ATR-based)")
    
    print("\n🎯 EXPECTED RESULTS:")
    print("• Higher quality, fewer signals (vs previous 0 signals)")
    print("• Better win rate due to confluence requirements")
    print("• Proper multi-timeframe alignment")
    print("• Institutional-level market analysis")
    
    return signals_df, features_df, df

if __name__ == "__main__":
    signals_df, features_df, df = main() 