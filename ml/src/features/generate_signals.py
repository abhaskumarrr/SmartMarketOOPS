import pandas as pd
import numpy as np
from .build_features import build_features

def check_confluence(features: pd.DataFrame, idx: int, signal_type: str) -> bool:
    """Checks for confluence of multiple factors for high-quality signals."""
    
    # Get current row data
    current = features.iloc[idx]
    
    if signal_type == 'buy':
        confluence_count = 0
        
        # HTF bias must be bullish
        if current['bias'] == 'bullish':
            confluence_count += 1
        
        # Volume expansion confirmation
        if current['vol_expansion']:
            confluence_count += 1
            
        # Price near support (Fibonacci or previous low)
        price_at_support = (
            abs(current['price_change']) < 0.01 and  # Not too much momentum
            (abs(current['prev_low'] - features.index[idx]) / features.index[idx] < 0.02 if hasattr(features.index[idx], '__float__') else True)
        )
        if price_at_support:
            confluence_count += 1
            
        # Order block confirmation
        if current['order_block'] == 1:
            confluence_count += 1
            
        # Session timing (London/NY)
        if current['london_session'] or current['ny_session']:
            confluence_count += 1
            
        # Require at least 3 confluences for buy signal
        return confluence_count >= 3
        
    elif signal_type == 'sell':
        confluence_count = 0
        
        # HTF bias must be bearish
        if current['bias'] == 'bearish':
            confluence_count += 1
        
        # Volume expansion confirmation
        if current['vol_expansion']:
            confluence_count += 1
            
        # Price near resistance
        price_at_resistance = (
            abs(current['price_change']) < 0.01 and  # Not too much momentum
            (abs(current['prev_high'] - features.index[idx]) / features.index[idx] < 0.02 if hasattr(features.index[idx], '__float__') else True)
        )
        if price_at_resistance:
            confluence_count += 1
            
        # Order block confirmation
        if current['order_block'] == -1:
            confluence_count += 1
            
        # Session timing
        if current['london_session'] or current['ny_session']:
            confluence_count += 1
            
        # Require at least 3 confluences for sell signal
        return confluence_count >= 3
    
    return False

def generate_scalping_signals(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """
    Strategy 1: Scalping with Higher-Timeframe Filter
    - Requires clear HTF momentum
    - Precise entry on pullbacks that respect HTF levels
    - Volume confirmation required
    - Session-based filtering
    """
    signals = pd.Series(0, index=df.index)
    
    for i in range(20, len(features)):  # Need lookback for analysis
        current = features.iloc[i]
        prev = features.iloc[i-1]
        
        # Only scalp in trending markets with clear HTF bias
        if current['market_regime'] not in ['trending_up', 'trending_down']:
            continue
            
        # Buy conditions for scalping
        if current['bias'] == 'bullish':
            # Look for pullback that doesn't break HTF structure
            pullback_condition = (
                df['close'].iloc[i] < df['close'].iloc[i-1] and  # Pullback
                not current['pivot_low']                          # Not breaking structure
            )
            
            # Volume and momentum confirmation (relaxed volume requirement)
            entry_trigger = (
                (current['vol_expansion'] or current['volume'] > current['volume_sma']) and  # Volume confirmation (relaxed)
                current['oc_pct'] > 0.1 and                     # Reduced momentum threshold
                current['is_key_session']                        # Session filter
            )
            
            if pullback_condition and entry_trigger and current['confluence_bullish'] >= 1:  # Reduced from 2 to 1
                signals.iloc[i] = 1
                
        # Sell conditions for scalping
        elif current['bias'] == 'bearish':
            # Look for pullback that doesn't break HTF structure
            pullback_condition = (
                df['close'].iloc[i] > df['close'].iloc[i-1] and  # Pullback up
                not current['pivot_high']                         # Not breaking structure
            )
            
            # Volume and momentum confirmation (relaxed volume requirement)
            entry_trigger = (
                (current['vol_expansion'] or current['volume'] > current['volume_sma']) and  # Volume confirmation (relaxed)
                current['oc_pct'] < -0.1 and                    # Reduced momentum threshold
                current['is_key_session']                        # Session filter
            )
            
            if pullback_condition and entry_trigger and current['confluence_bearish'] >= 1:  # Reduced from 2 to 1
                signals.iloc[i] = 2
    
    return signals

def generate_trend_capture_signals(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """
    Strategy 2: Full-Trend Capture (Swing Trading)
    - Waits for confirmed trend changes
    - Enters on structure breaks with volume
    - Uses Fibonacci targets
    - Holds for complete moves
    """
    signals = pd.Series(0, index=df.index)
    
    for i in range(50, len(features)):  # Need more lookback for trend analysis
        current = features.iloc[i]
        prev = features.iloc[i-1]
        
        # Only trade in trending regimes
        if current['market_regime'] not in ['trending_up', 'trending_down']:
            continue
        
        # Buy signal: New bullish trend confirmed
        if (current['bias'] == 'bullish' and prev['bias'] != 'bullish'):
            # Confirm with structure break
            structure_break = (
                current['pivot_high'] and                        # Breaking above recent high
                df['close'].iloc[i] > df['high'].iloc[i-10:i-1].max()  # Confirmed break
            )
            
            # Volume confirmation (relaxed requirement)
            volume_confirm = (current['vol_expansion'] or current['volume'] > current['volume_sma'])
            
            if structure_break and volume_confirm:
                signals.iloc[i] = 1
                
        # Sell signal: New bearish trend confirmed  
        elif (current['bias'] == 'bearish' and prev['bias'] != 'bearish'):
            # Confirm with structure break
            structure_break = (
                current['pivot_low'] and                         # Breaking below recent low
                df['close'].iloc[i] < df['low'].iloc[i-10:i-1].min()   # Confirmed break
            )
            
            # Volume confirmation (relaxed requirement)
            volume_confirm = (current['vol_expansion'] or current['volume'] > current['volume_sma'])
            
            if structure_break and volume_confirm:
                signals.iloc[i] = 2
    
    return signals

def generate_day_trade_signals(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """
    Strategy 3: High-Quality Day Trades (1–2 Entries)
    - Extremely selective with multiple confirmations
    - Session-based timing
    - Quality over quantity approach
    - Maximum 2 signals per day (simplified as infrequent signals)
    """
    signals = pd.Series(0, index=df.index)
    last_signal_idx = -50  # Minimum gap between signals
    
    for i in range(50, len(features)):
        current = features.iloc[i]
        
        # Skip if too close to last signal (quality over quantity)
        if i - last_signal_idx < 24:  # 24 hour minimum gap
            continue
            
        # Ultra-high-quality buy signal (relaxed from 3+ to 2+ confluence)
        if (current['bias'] == 'bullish' and 
            current['market_regime'] == 'trending_up' and
            current['confluence_bullish'] >= 2 and                   # Reduced from 3 to 2
            current['is_key_session'] and
            (current['vol_expansion'] or current['volume'] > current['volume_sma']) and  # Relaxed volume
            current['order_block_support']):
            
            signals.iloc[i] = 1
            last_signal_idx = i
            
        # Ultra-high-quality sell signal (relaxed from 3+ to 2+ confluence)
        elif (current['bias'] == 'bearish' and 
              current['market_regime'] == 'trending_down' and
              current['confluence_bearish'] >= 2 and               # Reduced from 3 to 2
              current['is_key_session'] and
              (current['vol_expansion'] or current['volume'] > current['volume_sma']) and  # Relaxed volume
              current['order_block_resistance']):
              
            signals.iloc[i] = 2
            last_signal_idx = i
    
    return signals

def generate_unified_signal(df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """
    Generates a unified signal by combining all three strategies.
    Prioritizes quality over quantity.
    """
    scalping = generate_scalping_signals(df, features)
    trend_capture = generate_trend_capture_signals(df, features)
    day_trade = generate_day_trade_signals(df, features)
    
    # Unified signal logic - prioritize day trade signals (highest quality)
    unified = pd.Series(0, index=df.index)
    
    for i in range(len(unified)):
        # Day trade signals have highest priority
        if day_trade.iloc[i] != 0:
            unified.iloc[i] = day_trade.iloc[i]
        # Trend capture signals second priority
        elif trend_capture.iloc[i] != 0:
            unified.iloc[i] = trend_capture.iloc[i]
        # Scalping signals lowest priority
        elif scalping.iloc[i] != 0:
            unified.iloc[i] = scalping.iloc[i]
    
    return unified

def generate_signals(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Generates signals for all strategies with enhanced confluence logic."""
    
    # Align the main dataframe with the features dataframe
    aligned_df = df.loc[features.index]

    signals = pd.DataFrame(index=aligned_df.index)
    signals['scalping'] = generate_scalping_signals(aligned_df, features)
    signals['trend_capture'] = generate_trend_capture_signals(aligned_df, features)
    signals['day_trade'] = generate_day_trade_signals(aligned_df, features)
    signals['unified'] = generate_unified_signal(aligned_df, features)
    
    return signals

if __name__ == '__main__':
    # Example usage
    data_path = '../../data/raw/BTC_USDT_1h.csv'
    df = pd.read_csv(data_path)
    
    features_df = build_features(df)
    signals_df = generate_signals(df, features_df)
    
    print("Generated Signals:")
    print(signals_df.head())
    
    # Save the signals to a new file
    output_path = '../../data/processed/BTC_USDT_1h_signals.csv'
    signals_df.to_csv(output_path)
    print(f"\nSignals saved to {output_path}")
