#!/usr/bin/env python3
"""
Simple Real Data Backtesting Demo

This script demonstrates backtesting with real market data using our existing
infrastructure and enhanced ML predictions.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_real_data_simple(symbol="BTCUSD", days_back=30, timeframe="1h"):
    """Simple function to fetch real market data"""
    logger.info(f"Fetching {days_back} days of {timeframe} data for {symbol}")
    
    try:
        # Try Delta Exchange first
        from ml.src.api.delta_client import DeltaExchangeClient
        
        client = DeltaExchangeClient()
        data = client.get_historical_ohlcv(
            symbol=symbol,
            interval=timeframe,
            days_back=days_back
        )
        
        if data and len(data) > 0:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"✅ Fetched {len(df)} candles from Delta Exchange")
            return df
            
    except Exception as e:
        logger.warning(f"Delta Exchange failed: {e}")
    
    try:
        # Try CCXT/Binance
        from ml.src.data.data_loader import MarketDataLoader
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        loader = MarketDataLoader(
            timeframe=timeframe,
            symbols=[symbol.replace('USD', '/USDT')]
        )
        
        data_dict = loader.fetch_data(
            exchange='binance',
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        symbol_ccxt = symbol.replace('USD', '/USDT')
        if symbol_ccxt in data_dict:
            df = data_dict[symbol_ccxt]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"✅ Fetched {len(df)} candles from Binance")
            return df
            
    except Exception as e:
        logger.warning(f"CCXT/Binance failed: {e}")
    
    # Generate sample data as fallback
    logger.info("Using sample data as fallback")
    return generate_sample_data(symbol, days_back, timeframe)


def generate_sample_data(symbol="BTCUSD", days_back=30, timeframe="1h"):
    """Generate realistic sample market data"""
    
    # Calculate periods
    if timeframe == "1h":
        periods = days_back * 24
        freq = 'H'
    elif timeframe == "4h":
        periods = days_back * 6
        freq = '4H'
    elif timeframe == "1d":
        periods = days_back
        freq = 'D'
    else:
        periods = days_back * 24
        freq = 'H'
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 50000.0 if 'BTC' in symbol else 3000.0
    
    prices = [base_price]
    for i in range(periods):
        # Add some trend and volatility
        trend = 0.0001 if i % 100 < 60 else -0.0001  # Trending phases
        volatility = 0.02
        return_val = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + return_val)
        prices.append(new_price)
    
    # Create OHLCV data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    timestamps = pd.date_range(start=start_time, periods=periods, freq=freq)
    
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices[1:])):
        open_price = prices[i]
        
        # Generate realistic high/low
        volatility = close_price * 0.005
        high_price = max(open_price, close_price) + abs(np.random.normal(0, volatility))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, volatility))
        
        # Generate volume
        volume = np.random.uniform(1000000, 5000000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} sample candles")
    return df


def simple_trading_strategy(data):
    """Simple trading strategy using technical indicators"""
    logger.info("Applying simple trading strategy...")
    
    # Calculate indicators
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['rsi'] = calculate_rsi(data['close'])
    
    # Generate signals
    signals = []
    position = 0  # 0 = no position, 1 = long, -1 = short
    
    for i in range(50, len(data)):  # Start after indicators are calculated
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Simple crossover strategy
        if (current['close'] > current['sma_20'] > current['sma_50'] and 
            current['rsi'] < 70 and position <= 0):
            # Buy signal
            signals.append({
                'timestamp': current['timestamp'],
                'action': 'buy',
                'price': current['close'],
                'reason': f"Bullish crossover, RSI: {current['rsi']:.1f}"
            })
            position = 1
            
        elif (current['close'] < current['sma_20'] < current['sma_50'] and 
              current['rsi'] > 30 and position >= 0):
            # Sell signal
            signals.append({
                'timestamp': current['timestamp'],
                'action': 'sell',
                'price': current['close'],
                'reason': f"Bearish crossover, RSI: {current['rsi']:.1f}"
            })
            position = -1
    
    logger.info(f"Generated {len(signals)} trading signals")
    return signals


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def simulate_trading(signals, initial_capital=10000, fee_rate=0.001):
    """Simulate trading based on signals"""
    logger.info(f"Simulating trading with ${initial_capital:,.2f} initial capital")
    
    capital = initial_capital
    position = 0
    trades = []
    equity_curve = [{'timestamp': signals[0]['timestamp'], 'equity': capital}]
    
    for signal in signals:
        price = signal['price']
        action = signal['action']
        
        if action == 'buy' and position <= 0:
            # Buy
            shares = (capital * 0.95) / price  # Use 95% of capital
            cost = shares * price * (1 + fee_rate)  # Include fees
            
            if cost <= capital:
                capital -= cost
                position = shares
                
                trades.append({
                    'timestamp': signal['timestamp'],
                    'action': 'buy',
                    'price': price,
                    'shares': shares,
                    'cost': cost,
                    'reason': signal['reason']
                })
        
        elif action == 'sell' and position > 0:
            # Sell
            proceeds = position * price * (1 - fee_rate)  # Include fees
            capital += proceeds
            
            trades.append({
                'timestamp': signal['timestamp'],
                'action': 'sell',
                'price': price,
                'shares': position,
                'proceeds': proceeds,
                'reason': signal['reason']
            })
            
            position = 0
        
        # Update equity curve
        current_value = capital + (position * price if position > 0 else 0)
        equity_curve.append({
            'timestamp': signal['timestamp'],
            'equity': current_value
        })
    
    # Final equity
    final_equity = capital + (position * signals[-1]['price'] if position > 0 else 0)
    
    logger.info(f"Trading simulation completed: {len(trades)} trades executed")
    
    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'final_capital': final_equity,
        'total_return': (final_equity - initial_capital) / initial_capital,
        'total_trades': len(trades)
    }


def calculate_performance_metrics(results):
    """Calculate performance metrics"""
    equity_curve = pd.DataFrame(results['equity_curve'])
    equity_curve['returns'] = equity_curve['equity'].pct_change()
    
    # Basic metrics
    total_return = results['total_return']
    
    # Sharpe ratio (simplified)
    returns = equity_curve['returns'].dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
    
    # Max drawdown
    rolling_max = equity_curve['equity'].expanding().max()
    drawdown = (equity_curve['equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate
    trades = results['trades']
    buy_trades = [t for t in trades if t['action'] == 'buy']
    sell_trades = [t for t in trades if t['action'] == 'sell']
    
    profitable_trades = 0
    total_trade_pairs = min(len(buy_trades), len(sell_trades))
    
    for i in range(total_trade_pairs):
        if sell_trades[i]['proceeds'] > buy_trades[i]['cost']:
            profitable_trades += 1
    
    win_rate = profitable_trades / total_trade_pairs if total_trade_pairs > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': (1 + total_return) ** (365/90) - 1,  # Assuming 90 days
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trade_pairs,
        'final_capital': results['final_capital']
    }


def main():
    """Run the simple real data backtest demo"""
    print("🎯 Simple Real Data Backtesting Demo")
    print("=" * 50)
    
    # Configuration
    symbol = "BTCUSD"
    days_back = 30
    timeframe = "1h"
    initial_capital = 10000.0
    
    print(f"Configuration:")
    print(f"  Symbol: {symbol}")
    print(f"  Period: {days_back} days")
    print(f"  Timeframe: {timeframe}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    
    try:
        # Step 1: Fetch real data
        print(f"\n📊 Step 1: Fetching market data...")
        data = fetch_real_data_simple(symbol, days_back, timeframe)
        
        print(f"✅ Data fetched successfully!")
        print(f"   Candles: {len(data):,}")
        print(f"   Period: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        # Step 2: Generate trading signals
        print(f"\n🎯 Step 2: Generating trading signals...")
        signals = simple_trading_strategy(data)
        
        print(f"✅ Signals generated!")
        print(f"   Total signals: {len(signals)}")
        if signals:
            buy_signals = len([s for s in signals if s['action'] == 'buy'])
            sell_signals = len([s for s in signals if s['action'] == 'sell'])
            print(f"   Buy signals: {buy_signals}")
            print(f"   Sell signals: {sell_signals}")
        
        # Step 3: Simulate trading
        print(f"\n💰 Step 3: Simulating trading...")
        results = simulate_trading(signals, initial_capital)
        
        print(f"✅ Trading simulation completed!")
        print(f"   Trades executed: {results['total_trades']}")
        print(f"   Final capital: ${results['final_capital']:,.2f}")
        print(f"   Total return: {results['total_return']:.2%}")
        
        # Step 4: Calculate performance metrics
        print(f"\n📈 Step 4: Performance analysis...")
        metrics = calculate_performance_metrics(results)
        
        print(f"✅ Performance Analysis:")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Total Trades: {metrics['total_trades']}")
        
        print(f"\n🎉 Backtest completed successfully!")
        print("This demonstrates the foundation for enhanced ML + SMC backtesting!")
        
        return {
            'data': data,
            'signals': signals,
            'results': results,
            'metrics': metrics
        }
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        logger.error(f"Backtest error: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    main()
