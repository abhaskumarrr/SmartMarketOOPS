#!/usr/bin/env python3
"""
Simple Real Data Signal Test
Direct CCXT integration to test enhanced multi-timeframe signals
"""

import pandas as pd
import numpy as np
import ccxt
import logging
import sys
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our enhanced modules
try:
    from features.build_features import build_features
    from features.generate_signals import generate_signals
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("Please ensure you're running from the ml/src directory")
    sys.exit(1)

def download_real_data_ccxt(symbol='BTC/USDT', timeframe='1h', limit=500):
    """Download real historical data directly using CCXT"""
    try:
        logger.info(f"📊 Downloading {symbol} {timeframe} data via CCXT...")
        
        # Initialize Binance exchange (no API keys needed for public data)
        exchange = ccxt.binance({
            'sandbox': False,  # Use live data (read-only)
            'enableRateLimit': True,
        })
        
        # Load markets
        exchange.load_markets()
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Downloaded {len(df)} candles for {symbol} {timeframe}")
        logger.info(f"📈 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        logger.info(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"💰 Current price: ${df['close'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to download {symbol} {timeframe}: {e}")
        return pd.DataFrame()

def test_signals_on_real_data(df, symbol, timeframe):
    """Test enhanced signals on real market data"""
    try:
        logger.info(f"\n🧠 Testing enhanced signals for {symbol} {timeframe}...")
        logger.info(f"📊 Data shape: {df.shape}")
        
        # Build enhanced features
        logger.info("🛠️ Building enhanced features...")
        features = build_features(df)
        
        if features.empty:
            logger.warning(f"⚠️ No features generated for {timeframe}")
            return None
        
        logger.info(f"✅ Features built successfully: {features.shape}")
        logger.info(f"📋 Feature columns: {list(features.columns)}")
        
        # Generate sophisticated signals
        logger.info("🎯 Generating sophisticated signals...")
        signals = generate_signals(df, features)
        
        if signals is not None and len(signals) > 0:
            # Analyze results for all strategies
            print(f"\n🎯 Enhanced Signal Results for {symbol} {timeframe}:")
            print(f"Signal columns: {list(signals.columns)}")
            
            for strategy in ['scalping', 'trend_capture', 'day_trade', 'unified']:
                if strategy in signals.columns:
                    buy_signals = len(signals[signals[strategy] == 1])
                    sell_signals = len(signals[signals[strategy] == 2])
                    total_signals = buy_signals + sell_signals
                    
                    print(f"\n📊 {strategy.upper()} Strategy:")
                    print(f"  • Buy signals: {buy_signals}")
                    print(f"  • Sell signals: {sell_signals}")
                    print(f"  • Total signals: {total_signals}")
                    print(f"  • Signal rate: {total_signals/len(df)*100:.1f}%")
                    
                    if total_signals > 0:
                        # Show signal details
                        signal_times = signals[signals[strategy] != 0].index
                        signal_values = signals[signals[strategy] != 0][strategy]
                        print(f"  • Signal timestamps: {list(signal_times[:5])}...")  # First 5
                        print(f"  • Signal types: {list(signal_values[:5])}...")      # First 5
            
            # Overall summary
            total_all_signals = 0
            for strategy in ['scalping', 'trend_capture', 'day_trade', 'unified']:
                if strategy in signals.columns:
                    total_all_signals += len(signals[signals[strategy] != 0])
            
            print(f"\n🎉 TOTAL SIGNALS ACROSS ALL STRATEGIES: {total_all_signals}")
            
            return signals
        else:
            logger.warning(f"⚠️ No signals generated for {symbol} {timeframe}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Error testing signals for {symbol} {timeframe}: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return None

def run_comprehensive_test():
    """Run comprehensive test with multiple symbols and timeframes"""
    logger.info("🚀 Starting comprehensive real data signal test with CCXT...")
    
    # Test configuration
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['1d', '4h', '1h', '15m']
    
    all_results = {}
    
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 Testing {symbol}")
        logger.info(f"{'='*60}")
        
        symbol_results = {}
        
        for timeframe in timeframes:
            logger.info(f"\n📈 Testing {symbol} on {timeframe} timeframe...")
            
            # Download real data
            df = download_real_data_ccxt(symbol, timeframe, limit=300)
            
            if df.empty:
                logger.warning(f"⚠️ No data available for {symbol} {timeframe}")
                continue
            
            # Test signals
            result = test_signals_on_real_data(df, symbol, timeframe)
            
            if result:
                symbol_results[timeframe] = result
            
        if symbol_results:
            all_results[symbol] = symbol_results
    
    # Summary
    logger.info(f"\n🎯 COMPREHENSIVE TEST SUMMARY:")
    logger.info(f"{'='*60}")
    
    total_signals_all = 0
    for symbol, timeframes in all_results.items():
        symbol_total = sum(tf_data['total_signals'] for tf_data in timeframes.values())
        total_signals_all += symbol_total
        
        logger.info(f"• {symbol}:")
        for tf, data in timeframes.items():
            logger.info(f"   - {tf}: {data['total_signals']} signals ({data['signal_rate']:.1f}%)")
    
    logger.info(f"\n📊 OVERALL RESULTS:")
    logger.info(f"   • Total signals across all tests: {total_signals_all}")
    logger.info(f"   • Symbols tested: {len(all_results)}")
    logger.info(f"   • Timeframes per symbol: {len(timeframes)}")
    
    if total_signals_all > 0:
        logger.info("✅ SUCCESS: Enhanced signal system is generating trading signals on real market data!")
        logger.info("💡 The sophisticated multi-timeframe strategies are working properly.")
    else:
        logger.warning("⚠️ NO SIGNALS: The system is not generating any trading signals.")
        logger.info("💡 Possible reasons:")
        logger.info("   • Market conditions don't match strategy criteria")
        logger.info("   • Confluence requirements are too restrictive")
        logger.info("   • Feature engineering needs debugging")
        logger.info("   • Signal generation logic needs adjustment")
    
    return all_results

def main():
    """Main test function"""
    try:
        # Quick single test first
        logger.info("🔍 Quick test with BTC/USDT 1h data...")
        df = download_real_data_ccxt('BTC/USDT', '1h', 200)
        
        if not df.empty:
            result = test_signals_on_real_data(df, 'BTC/USDT', '1h')
            
            if result and result['total_signals'] > 0:
                logger.info("✅ Quick test successful - proceeding with comprehensive test...")
                
                # Run comprehensive test
                run_comprehensive_test()
            else:
                logger.warning("❌ Quick test generated no signals - investigating...")
                logger.info("📊 Data sample:")
                logger.info(df.head())
                logger.info(f"📈 Price stats: min=${df['low'].min():.2f}, max=${df['high'].max():.2f}, avg=${df['close'].mean():.2f}")
        else:
            logger.error("❌ Failed to download test data")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 