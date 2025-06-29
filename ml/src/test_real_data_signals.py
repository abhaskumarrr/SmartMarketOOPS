#!/usr/bin/env python3
"""
Test Enhanced Multi-timeframe Signals with Real Market Data
Uses CCXT to download actual market data from Binance and tests the sophisticated strategies
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our enhanced modules
try:
    from features.build_features import build_features
    from features.generate_signals import generate_signals
    from data.real_market_data_service import RealMarketDataService
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("Please ensure you're running from the ml/src directory")
    sys.exit(1)

class RealDataSignalTester:
    """Test the enhanced signal generation system with real market data"""
    
    def __init__(self):
        self.market_service = None
        self.timeframes = ['1d', '4h', '1h', '15m']  # Multi-timeframe hierarchy
        self.symbols = ['BTC/USDT', 'ETH/USDT']
        self.data_cache = {}
        
    async def initialize(self):
        """Initialize the market data service"""
        try:
            self.market_service = RealMarketDataService()
            await self.market_service.initialize_exchanges()  
            logger.info("✅ Market data service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize market service: {e}")
            raise
    
    async def download_real_data(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Download real historical data from Binance via CCXT"""
        try:
            logger.info(f"📊 Downloading {symbol} {timeframe} data ({limit} candles)...")
            
            # Check cache first
            cache_key = f"{symbol}_{timeframe}_{limit}"
            if cache_key in self.data_cache:
                logger.info(f"📦 Using cached data for {cache_key}")
                return self.data_cache[cache_key].copy()
            
            # Download from Binance
            df = await self.market_service.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                exchange='binance'
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol} {timeframe}")
            
            # Validate data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns in data")
            
            # Cache the data
            self.data_cache[cache_key] = df.copy()
            
            logger.info(f"✅ Downloaded {len(df)} candles for {symbol} {timeframe}")
            logger.info(f"📈 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            logger.info(f"📅 Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to download {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    async def download_multi_timeframe_data(self, symbol: str) -> dict:
        """Download data for all timeframes"""
        logger.info(f"🔄 Downloading multi-timeframe data for {symbol}...")
        
        data = {}
        for timeframe in self.timeframes:
            df = await self.download_real_data(symbol, timeframe, limit=500)
            if not df.empty:
                data[timeframe] = df
            else:
                logger.warning(f"⚠️ No data for {symbol} {timeframe}")
        
        return data
    
    def test_enhanced_signals(self, symbol: str, data: dict) -> dict:
        """Test the enhanced signal generation on real data"""
        logger.info(f"🧠 Testing enhanced signals for {symbol}...")
        
        results = {}
        
        for timeframe in self.timeframes:
            if timeframe not in data:
                continue
                
            df = data[timeframe].copy()
            logger.info(f"\n🔍 Analyzing {symbol} {timeframe} ({len(df)} candles)...")
            
            try:
                # Build enhanced features
                logger.info("🛠️ Building enhanced features...")
                features = build_features(df)
                
                if features.empty:
                    logger.warning(f"⚠️ No features generated for {timeframe}")
                    continue
                
                # Generate sophisticated signals
                logger.info("🎯 Generating sophisticated signals...")
                signals = generate_signals(features)
                
                # Analyze results
                buy_signals = signals[signals['signal'] == 'buy']
                sell_signals = signals[signals['signal'] == 'sell']
                
                # Calculate signal metrics
                total_signals = len(buy_signals) + len(sell_signals)
                signal_rate = (total_signals / len(df)) * 100 if len(df) > 0 else 0
                
                # Store results
                results[timeframe] = {
                    'total_candles': len(df),
                    'total_signals': total_signals,
                    'buy_signals': len(buy_signals),
                    'sell_signals': len(sell_signals),
                    'signal_rate': signal_rate,
                    'features': features,
                    'signals': signals,
                    'price_range': {
                        'min': df['low'].min(),
                        'max': df['high'].max(),
                        'current': df['close'].iloc[-1]
                    }
                }
                
                # Log results
                logger.info(f"📊 {timeframe} Results:")
                logger.info(f"   • Total Signals: {total_signals}")
                logger.info(f"   • Buy Signals: {len(buy_signals)}")
                logger.info(f"   • Sell Signals: {len(sell_signals)}")
                logger.info(f"   • Signal Rate: {signal_rate:.2f}%")
                logger.info(f"   • Price: ${df['close'].iloc[-1]:.2f}")
                
                # Show sample signals if any
                if total_signals > 0:
                    logger.info(f"🎯 Sample signals from {timeframe}:")
                    sample_signals = signals[signals['signal'].isin(['buy', 'sell'])].tail(3)
                    for idx, signal in sample_signals.iterrows():
                        logger.info(f"   • {signal['signal'].upper()} at {idx}: ${signal.get('close', 'N/A'):.2f}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {timeframe}: {e}")
                continue
        
        return results
    
    def analyze_multi_timeframe_confluence(self, results: dict) -> dict:
        """Analyze confluence across timeframes"""
        logger.info("\n🔗 Analyzing multi-timeframe confluence...")
        
        confluence_analysis = {
            'timeframes_with_signals': [],
            'confluence_opportunities': 0,
            'strongest_signals': []
        }
        
        # Find timeframes with signals
        for tf, data in results.items():
            if data['total_signals'] > 0:
                confluence_analysis['timeframes_with_signals'].append(tf)
        
        logger.info(f"📈 Timeframes with signals: {confluence_analysis['timeframes_with_signals']}")
        
        # Look for confluence (signals across multiple timeframes)
        if len(confluence_analysis['timeframes_with_signals']) >= 2:
            confluence_analysis['confluence_opportunities'] = 1
            logger.info("✅ Multi-timeframe confluence detected!")
        else:
            logger.info("❌ No multi-timeframe confluence found")
        
        return confluence_analysis
    
    def create_visualization(self, symbol: str, results: dict):
        """Create visualization of the results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Enhanced Multi-timeframe Analysis: {symbol}', fontsize=16)
            
            plot_idx = 0
            for timeframe, data in results.items():
                if plot_idx >= 4:
                    break
                    
                row = plot_idx // 2
                col = plot_idx % 2
                ax = axes[row, col]
                
                if 'features' in data and not data['features'].empty:
                    df = data['features']
                    
                    # Plot price with signals
                    ax.plot(df.index, df['close'], label='Price', alpha=0.7)
                    
                    # Plot signals if any
                    if 'signals' in data and not data['signals'].empty:
                        signals = data['signals']
                        buy_signals = signals[signals['signal'] == 'buy']
                        sell_signals = signals[signals['signal'] == 'sell']
                        
                        if not buy_signals.empty:
                            ax.scatter(buy_signals.index, buy_signals['close'], 
                                     color='green', marker='^', s=100, label='Buy')
                        
                        if not sell_signals.empty:
                            ax.scatter(sell_signals.index, sell_signals['close'], 
                                     color='red', marker='v', s=100, label='Sell')
                    
                    ax.set_title(f'{timeframe} - {data["total_signals"]} signals')
                    ax.set_ylabel('Price ($)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plot_idx += 1
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"../../output/real_data_signals_{symbol.replace('/', '')}_{timestamp}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Chart saved: {filename}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Visualization error: {e}")
    
    async def run_comprehensive_test(self):
        """Run comprehensive test of the enhanced signal system"""
        logger.info("🚀 Starting comprehensive real data signal test...")
        
        try:
            # Initialize service
            await self.initialize()
            
            all_results = {}
            
            # Test each symbol
            for symbol in self.symbols:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Testing {symbol}")
                logger.info(f"{'='*60}")
                
                # Download multi-timeframe data
                data = await self.download_multi_timeframe_data(symbol)
                
                if not data:
                    logger.warning(f"⚠️ No data available for {symbol}")
                    continue
                
                # Test enhanced signals
                results = self.test_enhanced_signals(symbol, data)
                all_results[symbol] = results
                
                # Analyze confluence
                confluence = self.analyze_multi_timeframe_confluence(results)
                
                # Create visualization
                self.create_visualization(symbol, results)
                
                # Summary for this symbol
                total_signals_all_tf = sum(r['total_signals'] for r in results.values())
                logger.info(f"\n📋 {symbol} Summary:")
                logger.info(f"   • Total signals across all timeframes: {total_signals_all_tf}")
                logger.info(f"   • Confluence opportunities: {confluence['confluence_opportunities']}")
                logger.info(f"   • Active timeframes: {len(results)}")
            
            # Overall summary
            logger.info(f"\n🎯 OVERALL TEST RESULTS:")
            logger.info(f"{'='*60}")
            
            for symbol, results in all_results.items():
                total_signals = sum(r['total_signals'] for r in results.values())
                logger.info(f"• {symbol}: {total_signals} total signals across {len(results)} timeframes")
            
            if all_results:
                logger.info("✅ Enhanced signal system test completed successfully!")
                logger.info("💡 The system is now generating signals on real market data!")
            else:
                logger.warning("⚠️ No results generated - check data availability")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
        finally:
            # Cleanup
            if self.market_service:
                await self.market_service.stop()

async def main():
    """Main test function"""
    tester = RealDataSignalTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1) 