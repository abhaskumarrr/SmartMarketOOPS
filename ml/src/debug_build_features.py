#!/usr/bin/env python3
"""
Debug Build Features Function
Isolate and fix issues in the enhanced build_features function
"""

import pandas as pd
import numpy as np
import ccxt
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_test_data():
    """Download a small sample of real data for debugging"""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        exchange.load_markets()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Downloaded test data: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Data types: {df.dtypes}")
        logger.info(f"Sample data:\n{df.head()}")
        
        return df
    except Exception as e:
        logger.error(f"Failed to download test data: {e}")
        return pd.DataFrame()

def debug_get_pivot_points(df):
    """Debug pivot point detection"""
    try:
        logger.info("Testing pivot point detection...")
        
        window = 5
        pivots = pd.DataFrame(index=df.index)
        
        # Calculate pivot highs and lows
        highs = df['high'].rolling(window=window*2+1, center=True).max() == df['high']
        lows = df['low'].rolling(window=window*2+1, center=True).min() == df['low']
        
        pivots['pivot_high'] = highs
        pivots['pivot_low'] = lows
        pivots['swing_high'] = df['high'].where(highs)
        pivots['swing_low'] = df['low'].where(lows)
        
        logger.info(f"Pivot points shape: {pivots.shape}")
        logger.info(f"Pivot highs found: {pivots['pivot_high'].sum()}")
        logger.info(f"Pivot lows found: {pivots['pivot_low'].sum()}")
        
        return pivots
        
    except Exception as e:
        logger.error(f"Pivot point detection failed: {e}")
        return pd.DataFrame()

def debug_market_regime(df):
    """Debug market regime detection"""
    try:
        logger.info("Testing market regime detection...")
        
        # Get pivot points first
        pivots = debug_get_pivot_points(df)
        
        if pivots.empty:
            logger.error("No pivot points - cannot determine regime")
            return pd.Series(['unknown'] * len(df), index=df.index)
        
        # Simple regime detection
        regime = pd.Series(['unknown'] * len(df), index=df.index)
        
        # Count recent swing points to determine trend vs range
        lookback = 20
        for i in range(lookback, len(df)):
            recent_pivots = pivots.iloc[i-lookback:i]
            high_count = recent_pivots['pivot_high'].sum()
            low_count = recent_pivots['pivot_low'].sum()
            
            if high_count >= 2 and low_count >= 2:
                # Check if making higher highs and higher lows
                recent_highs = recent_pivots['swing_high'].dropna()
                recent_lows = recent_pivots['swing_low'].dropna()
                
                if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                    if recent_highs.iloc[-1] > recent_highs.iloc[0] and recent_lows.iloc[-1] > recent_lows.iloc[0]:
                        regime.iloc[i] = 'trending_up'
                    elif recent_highs.iloc[-1] < recent_highs.iloc[0] and recent_lows.iloc[-1] < recent_lows.iloc[0]:
                        regime.iloc[i] = 'trending_down'
                    else:
                        regime.iloc[i] = 'ranging'
                else:
                    regime.iloc[i] = 'ranging'
        
        logger.info(f"Market regime detection completed")
        logger.info(f"Regime distribution: {regime.value_counts()}")
        
        return regime
        
    except Exception as e:
        logger.error(f"Market regime detection failed: {e}")
        return pd.Series(['unknown'] * len(df), index=df.index)

def debug_htf_bias(df):
    """Debug HTF bias calculation"""
    try:
        logger.info("Testing HTF bias calculation...")
        
        # Get pivot points for structure analysis
        pivots = debug_get_pivot_points(df)
        
        if pivots.empty:
            logger.error("No pivot points - cannot determine HTF bias")
            return pd.Series(['neutral'] * len(df), index=df.index)
        
        bias = pd.Series(['neutral'] * len(df), index=df.index)
        
        # Structure-based HTF bias
        lookback = 30
        for i in range(lookback, len(df)):
            recent_data = df.iloc[i-lookback:i+1]
            recent_pivots = pivots.iloc[i-lookback:i+1]
            
            # Get recent swing points
            swing_highs = recent_pivots['swing_high'].dropna()
            swing_lows = recent_pivots['swing_low'].dropna()
            
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Check for higher highs and higher lows (bullish structure)
                latest_high = swing_highs.iloc[-1]
                prev_high = swing_highs.iloc[-2]
                latest_low = swing_lows.iloc[-1] 
                prev_low = swing_lows.iloc[-2]
                
                if latest_high > prev_high and latest_low > prev_low:
                    bias.iloc[i] = 'bullish'
                elif latest_high < prev_high and latest_low < prev_low:
                    bias.iloc[i] = 'bearish'
                else:
                    bias.iloc[i] = 'neutral'
        
        logger.info(f"HTF bias calculation completed")
        logger.info(f"Bias distribution: {bias.value_counts()}")
        
        return bias
        
    except Exception as e:
        logger.error(f"HTF bias calculation failed: {e}")
        return pd.Series(['neutral'] * len(df), index=df.index)

def debug_build_features_step_by_step(df):
    """Debug the build_features function step by step"""
    try:
        logger.info("Starting step-by-step feature building debug...")
        
        # Initialize features DataFrame
        features = df.copy()
        logger.info(f"Initial features shape: {features.shape}")
        
        # Step 1: Pivot points
        logger.info("Step 1: Adding pivot points...")
        pivots = debug_get_pivot_points(df)
        if not pivots.empty:
            for col in ['pivot_high', 'pivot_low', 'swing_high', 'swing_low']:
                if col in pivots.columns:
                    features[col] = pivots[col]
            logger.info("✅ Pivot points added")
        else:
            logger.error("❌ Pivot points failed")
            return pd.DataFrame()
        
        # Step 2: Market regime
        logger.info("Step 2: Adding market regime...")
        regime = debug_market_regime(df)
        features['regime'] = regime
        logger.info("✅ Market regime added")
        
        # Step 3: HTF bias
        logger.info("Step 3: Adding HTF bias...")
        bias = debug_htf_bias(df)
        features['bias'] = bias
        logger.info("✅ HTF bias added")
        
        # Step 4: Simple technical indicators
        logger.info("Step 4: Adding technical indicators...")
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['rsi'] = calculate_rsi(df['close'], 14)
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['vol_expansion'] = df['volume'] > features['volume_ma'] * 1.5
        logger.info("✅ Technical indicators added")
        
        # Step 5: Basic session analysis
        logger.info("Step 5: Adding session analysis...")
        features['hour'] = features.index.hour
        features['ny_session'] = ((features['hour'] >= 14) & (features['hour'] <= 20)).astype(bool)
        features['london_session'] = ((features['hour'] >= 8) & (features['hour'] <= 16)).astype(bool)
        logger.info("✅ Session analysis added")
        
        # Remove rows with insufficient data
        logger.info("Step 6: Cleaning data...")
        initial_len = len(features)
        features = features.dropna()
        final_len = len(features)
        logger.info(f"Cleaned data: {initial_len} -> {final_len} rows")
        
        logger.info(f"Final features shape: {features.shape}")
        logger.info(f"Final feature columns: {list(features.columns)}")
        
        return features
        
    except Exception as e:
        logger.error(f"Step-by-step feature building failed: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return pd.DataFrame()

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception as e:
        logger.error(f"RSI calculation failed: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

def main():
    """Main debug function"""
    logger.info("🔍 Starting build_features debugging...")
    
    # Download test data
    df = download_test_data()
    if df.empty:
        logger.error("No test data available")
        return
    
    # Debug step by step
    features = debug_build_features_step_by_step(df)
    
    if not features.empty:
        logger.info("✅ SUCCESS: Features built successfully!")
        logger.info(f"Final result shape: {features.shape}")
        logger.info("Sample features:")
        logger.info(features[['close', 'regime', 'bias', 'rsi', 'vol_expansion']].tail())
    else:
        logger.error("❌ FAILED: No features generated")

if __name__ == "__main__":
    main() 