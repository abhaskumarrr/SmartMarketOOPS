#!/usr/bin/env python3
"""
Real Market Data Backtest for SmartMarketOOPS
Uses actual Bitcoin price data from exchanges
"""

import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import time

class RealMarketBacktester:
    def __init__(self):
        self.initial_capital = 1000.0
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        
        # Risk management settings (conservative for real money)
        self.max_position_size = 100.0  # $100 max per trade
        self.risk_per_trade = 0.02      # 2% risk per trade
        self.stop_loss_pct = 0.02       # 2% stop loss
        self.take_profit_pct = 0.04     # 4% take profit
        self.max_positions = 3          # Max 3 open positions
        
        print("🏦 Real Market Backtester Initialized")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Max Position Size: ${self.max_position_size}")
        print(f"⚠️  Risk per Trade: {self.risk_per_trade*100}%")
    
    def fetch_real_bitcoin_data(self, days=30):
        """Fetch real Bitcoin data from Binance API"""
        print(f"📡 Fetching real Bitcoin data for last {days} days...")
        
        try:
            # Binance API for BTCUSDT klines (candlestick data)
            url = "https://api.binance.com/api/v3/klines"
            
            # Calculate timestamps
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1h',  # 1-hour intervals
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ Fetched {len(df)} real Bitcoin price points")
            print(f"📅 Data range: {df.index[0]} to {df.index[-1]}")
            print(f"💵 Price range: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching real data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for trading signals"""
        print("📊 Calculating technical indicators...")
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
        
        print("✅ Technical indicators calculated")
        return df
    
    def generate_trading_signals(self, df):
        """Generate trading signals based on technical analysis"""
        print("🧠 Generating trading signals...")
        
        signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Skip if not enough data for indicators
            if pd.isna(row['sma_50']) or pd.isna(row['rsi']) or pd.isna(row['macd']):
                signals.append({'action': 'hold', 'confidence': 0.0, 'reason': 'insufficient_data'})
                continue
            
            # Initialize signal
            signal = {'action': 'hold', 'confidence': 0.5, 'reason': 'no_signal'}
            
            # Multiple signal criteria
            buy_signals = 0
            sell_signals = 0
            confidence_factors = []
            
            # 1. Moving Average Crossover
            if row['sma_10'] > row['sma_20'] > row['sma_50']:
                buy_signals += 1
                confidence_factors.append(0.15)
            elif row['sma_10'] < row['sma_20'] < row['sma_50']:
                sell_signals += 1
                confidence_factors.append(0.15)
            
            # 2. MACD Signal
            if row['macd'] > row['macd_signal'] and row['macd_histogram'] > 0:
                buy_signals += 1
                confidence_factors.append(0.2)
            elif row['macd'] < row['macd_signal'] and row['macd_histogram'] < 0:
                sell_signals += 1
                confidence_factors.append(0.2)
            
            # 3. RSI Conditions
            if 30 < row['rsi'] < 40:  # Oversold but recovering
                buy_signals += 1
                confidence_factors.append(0.15)
            elif 60 < row['rsi'] < 70:  # Overbought but not extreme
                sell_signals += 1
                confidence_factors.append(0.15)
            
            # 4. Bollinger Bands
            if row['close'] < row['bb_lower'] and row['close'] > df.iloc[i-1]['close']:  # Bounce from lower band
                buy_signals += 1
                confidence_factors.append(0.1)
            elif row['close'] > row['bb_upper'] and row['close'] < df.iloc[i-1]['close']:  # Rejection from upper band
                sell_signals += 1
                confidence_factors.append(0.1)
            
            # 5. Volume Confirmation
            if row['volume_ratio'] > 1.2:  # High volume
                confidence_factors.append(0.1)
            
            # 6. Momentum
            if row['momentum_5'] > 0.02 and row['momentum_10'] > 0.05:  # Strong upward momentum
                buy_signals += 1
                confidence_factors.append(0.1)
            elif row['momentum_5'] < -0.02 and row['momentum_10'] < -0.05:  # Strong downward momentum
                sell_signals += 1
                confidence_factors.append(0.1)
            
            # Determine final signal
            if buy_signals >= 3 and buy_signals > sell_signals:
                signal['action'] = 'buy'
                signal['confidence'] = min(0.9, 0.5 + sum(confidence_factors))
                signal['reason'] = f'buy_signals_{buy_signals}'
            elif sell_signals >= 3 and sell_signals > buy_signals:
                signal['action'] = 'sell'
                signal['confidence'] = min(0.9, 0.5 + sum(confidence_factors))
                signal['reason'] = f'sell_signals_{sell_signals}'
            
            signals.append(signal)
        
        print(f"✅ Generated {len(signals)} trading signals")
        return signals
    
    def calculate_position_size(self, confidence, price):
        """Calculate position size based on risk management"""
        # Kelly Criterion inspired sizing
        risk_amount = self.current_capital * self.risk_per_trade
        stop_distance = price * self.stop_loss_pct
        
        if stop_distance > 0:
            base_size = risk_amount / stop_distance
        else:
            base_size = self.max_position_size
        
        # Apply confidence scaling
        position_size = base_size * confidence
        
        # Apply limits
        position_size = min(position_size, self.max_position_size)
        position_size = min(position_size, self.current_capital * 0.3)  # Max 30% of capital
        position_size = max(position_size, 10)  # Minimum $10
        
        return position_size
    
    def execute_trade(self, action, price, size, timestamp, confidence, reason):
        """Execute a trade"""
        if action == 'hold':
            return None
        
        if len(self.positions) >= self.max_positions:
            return None
        
        if size > self.current_capital:
            return None
        
        trade = {
            'id': f"trade_{len(self.trades)}",
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'size': size,
            'confidence': confidence,
            'reason': reason,
            'stop_loss': price * (1 - self.stop_loss_pct) if action == 'buy' else price * (1 + self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct) if action == 'buy' else price * (1 - self.take_profit_pct),
            'status': 'open'
        }
        
        self.current_capital -= size
        self.positions.append(trade)
        self.trades.append(trade)
        
        print(f"📈 {timestamp.strftime('%Y-%m-%d %H:%M')} - {action.upper()} ${size:.2f} at ${price:.2f}")
        print(f"   Confidence: {confidence:.2f}, Reason: {reason}")
        print(f"   Stop Loss: ${trade['stop_loss']:.2f}, Take Profit: ${trade['take_profit']:.2f}")
        
        return trade
    
    def check_exits(self, price, timestamp):
        """Check exit conditions for open positions"""
        positions_to_close = []
        
        for i, pos in enumerate(self.positions):
            if pos['status'] != 'open':
                continue
            
            should_close = False
            exit_reason = ""
            exit_price = price
            
            # Check stop loss
            if pos['action'] == 'buy' and price <= pos['stop_loss']:
                should_close = True
                exit_reason = "stop_loss"
                exit_price = pos['stop_loss']
            elif pos['action'] == 'sell' and price >= pos['stop_loss']:
                should_close = True
                exit_reason = "stop_loss"
                exit_price = pos['stop_loss']
            
            # Check take profit
            elif pos['action'] == 'buy' and price >= pos['take_profit']:
                should_close = True
                exit_reason = "take_profit"
                exit_price = pos['take_profit']
            elif pos['action'] == 'sell' and price <= pos['take_profit']:
                should_close = True
                exit_reason = "take_profit"
                exit_price = pos['take_profit']
            
            if should_close:
                # Calculate P&L
                if pos['action'] == 'buy':
                    pnl = (exit_price - pos['price']) * (pos['size'] / pos['price'])
                else:
                    pnl = (pos['price'] - exit_price) * (pos['size'] / pos['price'])
                
                pos['exit_timestamp'] = timestamp
                pos['exit_price'] = exit_price
                pos['exit_reason'] = exit_reason
                pos['pnl'] = pnl
                pos['status'] = 'closed'
                
                self.current_capital += pos['size'] + pnl
                
                pnl_pct = (pnl / pos['size']) * 100
                print(f"🔄 {timestamp.strftime('%Y-%m-%d %H:%M')} - CLOSE {pos['action'].upper()}")
                print(f"   P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) - Reason: {exit_reason}")
                
                positions_to_close.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def run_backtest(self, days=30):
        """Run the complete backtest"""
        print(f"\n🚀 Starting Real Market Data Backtest ({days} days)")
        print("=" * 60)
        
        # Fetch real data
        df = self.fetch_real_bitcoin_data(days)
        if df is None:
            return None
        
        # Calculate technical indicators
        df = self.calculate_technical_indicators(df)
        
        # Generate trading signals
        signals = self.generate_trading_signals(df)
        
        # Run backtest
        print(f"\n📊 Running backtest on {len(df)} data points...")
        
        for i, (timestamp, row) in enumerate(df.iterrows()):
            price = row['close']
            signal = signals[i]
            
            # Check exits first
            self.check_exits(price, timestamp)
            
            # Execute new trades
            if signal['action'] != 'hold' and signal['confidence'] > 0.65:
                size = self.calculate_position_size(signal['confidence'], price)
                self.execute_trade(
                    signal['action'], price, size, timestamp, 
                    signal['confidence'], signal['reason']
                )
        
        # Close remaining positions
        final_price = df['close'].iloc[-1]
        final_timestamp = df.index[-1]
        
        for pos in self.positions:
            if pos['action'] == 'buy':
                pnl = (final_price - pos['price']) * (pos['size'] / pos['price'])
            else:
                pnl = (pos['price'] - final_price) * (pos['size'] / pos['price'])
            
            pos['exit_timestamp'] = final_timestamp
            pos['exit_price'] = final_price
            pos['exit_reason'] = "backtest_end"
            pos['pnl'] = pnl
            pos['status'] = 'closed'
            
            self.current_capital += pos['size'] + pnl
            
            print(f"🔄 {final_timestamp.strftime('%Y-%m-%d %H:%M')} - FORCE CLOSE {pos['action'].upper()}")
            print(f"   P&L: ${pnl:.2f} - Reason: backtest_end")
        
        return self.calculate_results()
    
    def calculate_results(self):
        """Calculate comprehensive backtest results"""
        final_capital = self.current_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
        
        if closed_trades:
            win_rate = len(winning_trades) / len(closed_trades) * 100
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Calculate max drawdown
            portfolio_values = [self.initial_capital]
            running_capital = self.initial_capital
            
            for trade in closed_trades:
                running_capital += trade.get('pnl', 0)
                portfolio_values.append(running_capital)
            
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            max_drawdown = 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'trades': closed_trades
        }
    
    def print_results(self, results):
        """Print comprehensive results"""
        print("\n" + "=" * 60)
        print("🎯 REAL MARKET DATA BACKTEST RESULTS")
        print("=" * 60)
        print(f"💰 Initial Capital:     ${results['initial_capital']:,.2f}")
        print(f"💰 Final Capital:       ${results['final_capital']:,.2f}")
        print(f"📈 Total Return:        ${results['total_return']:,.2f} ({results['total_return_pct']:+.2f}%)")
        print(f"📊 Total Trades:        {results['total_trades']}")
        print(f"✅ Winning Trades:      {results['winning_trades']} ({results['win_rate']:.1f}%)")
        print(f"❌ Losing Trades:       {results['losing_trades']}")
        print(f"💵 Average Win:         ${results['avg_win']:+.2f}")
        print(f"💸 Average Loss:        ${results['avg_loss']:+.2f}")
        print(f"⚖️  Profit Factor:       {results['profit_factor']:.2f}")
        print(f"📉 Max Drawdown:        {results['max_drawdown']:.2f}%")
        print("=" * 60)
        
        # Performance assessment
        print("\n🔍 PERFORMANCE ASSESSMENT:")
        
        if results['total_return_pct'] > 10:
            print("🎉 EXCELLENT: Outstanding returns!")
        elif results['total_return_pct'] > 5:
            print("🎯 VERY GOOD: Strong positive returns")
        elif results['total_return_pct'] > 0:
            print("✅ GOOD: Positive returns achieved")
        elif results['total_return_pct'] > -5:
            print("⚠️  ACCEPTABLE: Minor losses, needs optimization")
        else:
            print("❌ POOR: Significant losses, major changes needed")
        
        if results['win_rate'] > 70:
            print("🎯 EXCELLENT WIN RATE: Superior trade selection")
        elif results['win_rate'] > 60:
            print("✅ GOOD WIN RATE: Solid performance")
        elif results['win_rate'] > 50:
            print("⚠️  AVERAGE WIN RATE: Room for improvement")
        else:
            print("❌ LOW WIN RATE: Strategy needs refinement")
        
        if results['max_drawdown'] < 5:
            print("🛡️  EXCELLENT RISK CONTROL: Very low drawdown")
        elif results['max_drawdown'] < 10:
            print("✅ GOOD RISK CONTROL: Acceptable drawdown")
        elif results['max_drawdown'] < 20:
            print("⚠️  MODERATE RISK: Monitor closely")
        else:
            print("🚨 HIGH RISK: Reduce position sizes")

def main():
    """Main execution function"""
    print("🚀 SmartMarketOOPS Real Market Data Backtest")
    print("=" * 50)
    
    backtester = RealMarketBacktester()
    
    try:
        results = backtester.run_backtest(days=30)
        
        if results:
            backtester.print_results(results)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_market_backtest_{timestamp}.json"
            
            # Prepare results for JSON serialization
            results_copy = results.copy()
            for trade in results_copy['trades']:
                if 'timestamp' in trade:
                    trade['timestamp'] = trade['timestamp'].isoformat()
                if 'exit_timestamp' in trade:
                    trade['exit_timestamp'] = trade['exit_timestamp'].isoformat()
            
            with open(filename, 'w') as f:
                json.dump(results_copy, f, indent=2)
            
            print(f"\n💾 Results saved to {filename}")
            
            # Deployment recommendation
            print("\n🚀 DEPLOYMENT RECOMMENDATION:")
            
            if (results['total_return_pct'] > 2 and 
                results['win_rate'] > 55 and 
                results['max_drawdown'] < 15):
                print("✅ APPROVED FOR LIVE TRADING")
                print("💡 Recommended next steps:")
                print("   1. Deploy system with conservative settings")
                print("   2. Start with 50% of planned capital")
                print("   3. Monitor performance for 1 week")
                print("   4. Gradually increase position sizes")
                print("   5. Set up alerts for drawdown > 10%")
            else:
                print("⚠️  PROCEED WITH CAUTION")
                print("💡 Recommendations:")
                if results['total_return_pct'] <= 2:
                    print("   - Consider longer backtesting period")
                    print("   - Optimize signal confidence thresholds")
                if results['win_rate'] <= 55:
                    print("   - Refine entry/exit criteria")
                    print("   - Add more technical indicators")
                if results['max_drawdown'] >= 15:
                    print("   - Reduce position sizes")
                    print("   - Tighten stop losses")
                
                print("   - Start with paper trading mode")
                print("   - Use smaller position sizes initially")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()