import pandas as pd
import numpy as np

def get_market_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Determines the market regime (trending or range-bound)."""
    # A simple approach: use the standard deviation of returns
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=window).std()
    is_trending = volatility > volatility.quantile(0.7) # Top 30% volatility is considered trending
    return is_trending.map({True: 'trending', False: 'range-bound'})

def get_htf_bias(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Determines the higher-timeframe bias (bullish, bearish, neutral)."""
    mavg = df['close'].rolling(window=window).mean()
    bias = pd.Series('neutral', index=df.index)
    bias[df['close'] > mavg] = 'bullish'
    bias[df['close'] < mavg] = 'bearish'
    return bias

def identify_order_blocks(df: pd.DataFrame) -> pd.Series:
    """Identifies potential order blocks (simplified)."""
    # A simple heuristic: a large candle followed by a reversal
    body_size = abs(df['close'] - df['open'])
    is_large_candle = body_size > body_size.rolling(window=10).mean()
    
    # Bullish order block: large down candle followed by a move up
    is_bullish_ob = (df['open'] > df['close']) & is_large_candle & (df['close'].shift(-1) > df['open'].shift(-1))
    
    # Bearish order block: large up candle followed by a move down
    is_bearish_ob = (df['open'] < df['close']) & is_large_candle & (df['close'].shift(-1) < df['open'].shift(-1))
    
    ob = pd.Series(0, index=df.index) # 0: none, 1: bullish, -1: bearish
    ob[is_bullish_ob] = 1
    ob[is_bearish_ob] = -1
    return ob

def get_fibonacci_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Fibonacci retracement and extension levels."""
    high = df['high'].rolling(window=20).max()
    low = df['low'].rolling(window=20).min()
    
    fib_levels = pd.DataFrame(index=df.index)
    fib_levels['fib_0.382'] = low + (high - low) * 0.382
    fib_levels['fib_0.618'] = low + (high - low) * 0.618
    fib_levels['fib_1.272'] = high + (high - low) * 0.272
    fib_levels['fib_1.618'] = high + (high - low) * 0.618
    return fib_levels

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds all features for a given dataframe."""
    features = pd.DataFrame(index=df.index)
    
    # Basic features
    features['price_change'] = df['close'].pct_change()
    features['volume_change'] = df['volume'].pct_change()
    
    # Strategy-specific features
    features['regime'] = get_market_regime(df)
    features['bias'] = get_htf_bias(df)
    features['order_block'] = identify_order_blocks(df)
    
    fib_levels = get_fibonacci_levels(df)
    features = pd.concat([features, fib_levels], axis=1)
    
    # OHLC levels
    features['prev_high'] = df['high'].shift(1)
    features['prev_low'] = df['low'].shift(1)
    features['prev_close'] = df['close'].shift(1)
    
    return features.dropna()

if __name__ == '__main__':
    # Example usage
    data_path = '../../data/raw/BTC_USDT_1h.csv'
    df = pd.read_csv(data_path)
    
    features_df = build_features(df)
    
    print("Generated Features:")
    print(features_df.head())
    
    # Save the features to a new file
    output_path = '../../data/processed/BTC_USDT_1h_features.csv'
    features_df.to_csv(output_path)
    print(f"\nFeatures saved to {output_path}")
