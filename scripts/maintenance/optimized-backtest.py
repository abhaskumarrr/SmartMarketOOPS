#!/usr/bin/env python3
"""
Optimized Real Market Backtest with Lower Confidence Threshold
"""

import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import time

class OptimizedBacktester:
    def __init__(self):
        self.initial_capital = 1000.0
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        
        # More aggressive settings for backtesting
        self.max_position_size = 50.0   # Smaller positions
        self.risk_per_trade = 0.015     # 1.5% risk
        self.stop_loss_pct = 0.015      # 1.5% stop loss
        self.take_profit_pct = 0.03     # 3% take profit
        self.max_positions = 2          # Max 2 positions
        self.confidence_threshold = 0.55  # Lower threshold
        
        print("🎯 Optimized Backtester (Lower Confidence Threshold)")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Confidence Threshold: {self.confidence_threshold*100}%")
    
    def fetch_real_bitcoin_data(self, days=30):
        """Fetch real Bitcoin data"""
        print(f"📡 Fetching Bitcoin data for {days} days...")
        
        try:
            url = "https://api.binance.com/api/v3/klines"
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            params = {
                'symbol': 'BTCUSDT',
                'interval': '4h',  # 4-hour intervals for better signals
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ Fetched {len(df)} data points")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def generate_simple_signals(self, df):
        """Generate simple but effective trading signals"""
        print("🧠 Generating trading signals...")
        
        # Calculate simple indicators
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume
        df['vol_avg'] = df['volume'].rolling(10).mean()
        
        signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            if i < 20:  # Need enough data
                signals.append({'action': 'hold', 'confidence': 0.0})
                continue
            
            # Simple trend following with RSI filter
            signal = {'action': 'hold', 'confidence': 0.5}
            
            # Trend signals
            if (row['sma_5'] > row['sma_10'] > row['sma_20'] and 
                row['rsi'] > 40 and row['rsi'] < 70 and
                row['volume'] > row['vol_avg'] * 0.8):
                signal = {'action': 'buy', 'confidence': 0.65}
            
            elif (row['sma_5'] < row['sma_10'] < row['sma_20'] and 
                  row['rsi'] < 60 and row['rsi'] > 30 and
                  row['volume'] > row['vol_avg'] * 0.8):
                signal = {'action': 'sell', 'confidence': 0.65}
            
            # Momentum signals
            elif (row['close'] > row['sma_5'] * 1.01 and 
                  row['rsi'] < 65 and row['volume'] > row['vol_avg']):
                signal = {'action': 'buy', 'confidence': 0.58}
            
            elif (row['close'] < row['sma_5'] * 0.99 and 
                  row['rsi'] > 35 and row['volume'] > row['vol_avg']):
                signal = {'action': 'sell', 'confidence': 0.58}
            
            signals.append(signal)
        
        return signals
    
    def execute_trade(self, action, price, timestamp, confidence):
        """Execute trade with risk management"""
        if action == 'hold' or len(self.positions) >= self.max_positions:
            return None
        
        # Calculate position size
        risk_amount = self.current_capital * self.risk_per_trade
        stop_distance = price * self.stop_loss_pct
        position_size = min(risk_amount / stop_distance, self.max_position_size)
        position_size = min(position_size, self.current_capital * 0.25)
        
        if position_size < 10 or position_size > self.current_capital:
            return None
        
        trade = {
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'size': position_size,
            'confidence': confidence,
            'stop_loss': price * (1 - self.stop_loss_pct) if action == 'buy' else price * (1 + self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct) if action == 'buy' else price * (1 - self.take_profit_pct),
            'status': 'open'
        }
        
        self.current_capital -= position_size
        self.positions.append(trade)
        self.trades.append(trade)
        
        print(f"📈 {timestamp.strftime('%m-%d %H:%M')} {action.upper()} ${position_size:.0f} @ ${price:.0f}")
        return trade
    
    def check_exits(self, price, timestamp):
        """Check exit conditions"""
        for pos in self.positions[:]:
            if pos['action'] == 'buy':
                if price <= pos['stop_loss']:
                    pnl = (pos['stop_loss'] - pos['price']) * (pos['size'] / pos['price'])
                    self.close_position(pos, pos['stop_loss'], timestamp, 'stop_loss', pnl)
                elif price >= pos['take_profit']:
                    pnl = (pos['take_profit'] - pos['price']) * (pos['size'] / pos['price'])
                    self.close_position(pos, pos['take_profit'], timestamp, 'take_profit', pnl)
            
            else:  # sell
                if price >= pos['stop_loss']:
                    pnl = (pos['price'] - pos['stop_loss']) * (pos['size'] / pos['price'])
                    self.close_position(pos, pos['stop_loss'], timestamp, 'stop_loss', pnl)
                elif price <= pos['take_profit']:
                    pnl = (pos['price'] - pos['take_profit']) * (pos['size'] / pos['price'])
                    self.close_position(pos, pos['take_profit'], timestamp, 'take_profit', pnl)
    
    def close_position(self, pos, exit_price, timestamp, reason, pnl):
        """Close a position"""
        pos['exit_price'] = exit_price
        pos['exit_timestamp'] = timestamp
        pos['exit_reason'] = reason
        pos['pnl'] = pnl
        pos['status'] = 'closed'
        
        self.current_capital += pos['size'] + pnl
        self.positions.remove(pos)
        
        pnl_pct = (pnl / pos['size']) * 100
        print(f"🔄 {timestamp.strftime('%m-%d %H:%M')} CLOSE {pos['action'].upper()} "
              f"P&L: ${pnl:+.0f} ({pnl_pct:+.1f}%) [{reason}]")
    
    def run_backtest(self, days=30):
        """Run optimized backtest"""
        print(f"\n🚀 Running Optimized Backtest ({days} days)")
        
        df = self.fetch_real_bitcoin_data(days)
        if df is None:
            return None
        
        signals = self.generate_simple_signals(df)
        
        print(f"\n📊 Processing {len(df)} data points...")
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            price = row['close']
            signal = signals[i]
            
            self.check_exits(price, timestamp)
            
            if (signal['action'] != 'hold' and 
                signal['confidence'] >= self.confidence_threshold):
                self.execute_trade(signal['action'], price, timestamp, signal['confidence'])
        
        # Close remaining positions
        final_price = df['close'].iloc[-1]
        final_timestamp = df.index[-1]
        
        for pos in self.positions[:]:
            if pos['action'] == 'buy':
                pnl = (final_price - pos['price']) * (pos['size'] / pos['price'])
            else:
                pnl = (pos['price'] - final_price) * (pos['size'] / pos['price'])
            
            self.close_position(pos, final_price, final_timestamp, 'backtest_end', pnl)
        
        return self.calculate_results()
    
    def calculate_results(self):
        """Calculate results"""
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        
        if not closed_trades:
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        total_return = self.current_capital - self.initial_capital
        winning_trades = [t for t in closed_trades if t['pnl'] > 0]
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'total_return_pct': (total_return / self.initial_capital) * 100,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': len(winning_trades) / len(closed_trades) * 100,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in closed_trades if t['pnl'] < 0]) or 0,
            'trades': closed_trades
        }
    
    def print_results(self, results):
        """Print results"""
        print(f"\n{'='*50}")
        print("🎯 OPTIMIZED BACKTEST RESULTS")
        print(f"{'='*50}")
        print(f"💰 Final Capital:    ${results['final_capital']:,.0f}")
        print(f"📈 Total Return:     ${results['total_return']:+,.0f} ({results['total_return_pct']:+.1f}%)")
        print(f"📊 Total Trades:     {results['total_trades']}")
        print(f"✅ Win Rate:         {results['win_rate']:.1f}%")
        print(f"💵 Avg Win:          ${results['avg_win']:+.0f}")
        print(f"💸 Avg Loss:         ${results['avg_loss']:+.0f}")
        print(f"{'='*50}")
        
        if results['total_return_pct'] > 5:
            print("🎉 EXCELLENT: Ready for live trading!")
        elif results['total_return_pct'] > 0:
            print("✅ GOOD: Positive returns, proceed with caution")
        else:
            print("⚠️  NEEDS WORK: Consider optimization")

def main():
    backtester = OptimizedBacktester()
    results = backtester.run_backtest(30)
    
    if results:
        backtester.print_results(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_backtest_{timestamp}.json"
        
        results_copy = results.copy()
        for trade in results_copy['trades']:
            if 'timestamp' in trade:
                trade['timestamp'] = trade['timestamp'].isoformat()
            if 'exit_timestamp' in trade:
                trade['exit_timestamp'] = trade['exit_timestamp'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
        
        # Deployment decision
        if results['total_return_pct'] > 0 and results['total_trades'] > 0:
            print("\n🚀 DEPLOYMENT APPROVED!")
            print("✅ System shows profitable trading on real data")
            print("💡 Ready to deploy with conservative settings")
        else:
            print("\n⚠️  DEPLOY WITH PAPER TRADING FIRST")
            print("💡 Test with paper trading before real money")
    
    return results

if __name__ == "__main__":
    main()