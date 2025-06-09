#!/usr/bin/env python3
"""
Generate comprehensive demo data for SmartMarketOOPS
Creates realistic trading scenarios, sample datasets, and test cases
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoDataGenerator:
    """Generate comprehensive demo data for testing and development"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.sample_data_dir = self.project_root / 'sample_data'
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.sample_data_dir.mkdir(exist_ok=True)
        (self.data_dir / 'demo').mkdir(exist_ok=True)
        (self.data_dir / 'backtesting').mkdir(exist_ok=True)
        (self.data_dir / 'validation').mkdir(exist_ok=True)
    
    def generate_ohlcv_data(self, symbol='BTCUSD', timeframe='1h', days=30):
        """Generate realistic OHLCV data"""
        logger.info(f"Generating {timeframe} OHLCV data for {symbol} ({days} days)")
        
        # Calculate number of candles based on timeframe
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        total_candles = (days * 24 * 60) // minutes
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=total_candles)
        
        # Generate realistic price data with trend and volatility
        base_price = 45000 if 'BTC' in symbol else 2500  # Starting price
        
        # Create price series with realistic movements
        returns = np.random.normal(0, 0.02, total_candles)  # 2% volatility
        
        # Add trend component
        trend = np.linspace(-0.1, 0.1, total_candles)  # Slight upward trend
        returns += trend
        
        # Calculate prices
        prices = [base_price]
        for i in range(1, total_candles):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 100))  # Minimum price floor
        
        # Generate OHLCV data
        data = []
        for i, timestamp in enumerate(timestamps):
            if i == 0:
                open_price = prices[i]
            else:
                open_price = data[-1]['close']
            
            close_price = prices[i]
            
            # Generate high and low based on volatility
            volatility = abs(returns[i]) * 2
            high_price = max(open_price, close_price) * (1 + volatility)
            low_price = min(open_price, close_price) * (1 - volatility)
            
            # Generate volume (higher volume during price movements)
            base_volume = 1000000
            volume_multiplier = 1 + abs(returns[i]) * 5
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def generate_orderbook_data(self, symbol='BTCUSD', depth=20):
        """Generate realistic orderbook data"""
        logger.info(f"Generating orderbook data for {symbol}")
        
        # Current price around 45000 for BTC
        mid_price = 45000 if 'BTC' in symbol else 2500
        
        # Generate bids and asks
        bids = []
        asks = []
        
        for i in range(depth):
            # Bids (decreasing prices)
            bid_price = mid_price * (1 - (i + 1) * 0.0001)
            bid_size = np.random.uniform(0.1, 5.0)
            bids.append([round(bid_price, 2), round(bid_size, 4)])
            
            # Asks (increasing prices)
            ask_price = mid_price * (1 + (i + 1) * 0.0001)
            ask_size = np.random.uniform(0.1, 5.0)
            asks.append([round(ask_price, 2), round(ask_size, 4)])
        
        orderbook = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'bids': bids,
            'asks': asks,
            'mid_price': mid_price
        }
        
        return orderbook
    
    def generate_trade_scenarios(self):
        """Generate realistic trading scenarios for testing"""
        logger.info("Generating trading scenarios")
        
        scenarios = [
            {
                'name': 'Bullish Breakout',
                'description': 'Strong upward momentum with high volume',
                'setup': {
                    'trend': 'bullish',
                    'volatility': 'high',
                    'volume': 'increasing',
                    'confluence_score': 0.85
                },
                'expected_outcome': {
                    'direction': 'long',
                    'probability': 0.75,
                    'risk_reward': 3.0
                }
            },
            {
                'name': 'Bearish Reversal',
                'description': 'Trend reversal with strong selling pressure',
                'setup': {
                    'trend': 'bearish',
                    'volatility': 'medium',
                    'volume': 'high',
                    'confluence_score': 0.78
                },
                'expected_outcome': {
                    'direction': 'short',
                    'probability': 0.68,
                    'risk_reward': 2.5
                }
            },
            {
                'name': 'Range Bound',
                'description': 'Sideways movement with low volatility',
                'setup': {
                    'trend': 'sideways',
                    'volatility': 'low',
                    'volume': 'decreasing',
                    'confluence_score': 0.45
                },
                'expected_outcome': {
                    'direction': 'none',
                    'probability': 0.30,
                    'risk_reward': 1.0
                }
            }
        ]
        
        return scenarios
    
    def generate_all_demo_data(self):
        """Generate all demo data files"""
        logger.info("🚀 Starting comprehensive demo data generation...")
        
        # 1. Generate multi-timeframe OHLCV data
        symbols = ['BTCUSD', 'ETHUSD']
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Generate different amounts of data based on timeframe
                days = {'1m': 2, '5m': 7, '15m': 14, '1h': 30, '4h': 90, '1d': 365}
                
                df = self.generate_ohlcv_data(symbol, timeframe, days.get(timeframe, 30))
                
                # Save to sample_data
                filename = f"{symbol}_{timeframe}.csv"
                filepath = self.sample_data_dir / filename
                df.to_csv(filepath, index=False)
                logger.info(f"✅ Generated {filename}")
        
        # 2. Generate orderbook data
        for symbol in symbols:
            orderbook = self.generate_orderbook_data(symbol)
            filename = f"{symbol}_orderbook.json"
            filepath = self.sample_data_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(orderbook, f, indent=2)
            logger.info(f"✅ Generated {filename}")
        
        # 3. Generate trading scenarios
        scenarios = self.generate_trade_scenarios()
        scenarios_file = self.data_dir / 'demo' / 'trading_scenarios.json'
        
        with open(scenarios_file, 'w') as f:
            json.dump(scenarios, f, indent=2)
        logger.info("✅ Generated trading_scenarios.json")
        
        # 4. Generate backtesting datasets
        for symbol in symbols:
            # Generate 6 months of hourly data for backtesting
            df = self.generate_ohlcv_data(symbol, '1h', 180)
            backtest_file = self.data_dir / 'backtesting' / f"{symbol}_6months.csv"
            df.to_csv(backtest_file, index=False)
            logger.info(f"✅ Generated backtesting data for {symbol}")
        
        # 5. Generate validation datasets
        for symbol in symbols:
            # Generate 1 month of 15m data for validation
            df = self.generate_ohlcv_data(symbol, '15m', 30)
            validation_file = self.data_dir / 'validation' / f"{symbol}_validation.csv"
            df.to_csv(validation_file, index=False)
            logger.info(f"✅ Generated validation data for {symbol}")
        
        logger.info("🎉 Demo data generation completed successfully!")
        logger.info(f"📁 Data saved to:")
        logger.info(f"   - Sample data: {self.sample_data_dir}")
        logger.info(f"   - Demo data: {self.data_dir / 'demo'}")
        logger.info(f"   - Backtesting: {self.data_dir / 'backtesting'}")
        logger.info(f"   - Validation: {self.data_dir / 'validation'}")

if __name__ == "__main__":
    generator = DemoDataGenerator()
    generator.generate_all_demo_data()
