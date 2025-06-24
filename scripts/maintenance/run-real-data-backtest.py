#!/usr/bin/env python3
"""
SmartMarketOOPS Real Data Backtest
Comprehensive backtest using real market data before live deployment
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Tuple

# Add ML module to path
sys.path.append('./ml/src')

try:
    from enhanced_ml_model import EnhancedMLModel
    from fibonacci_ml_model import FibonacciMLModel
except ImportError as e:
    print(f"Warning: Could not import ML models: {e}")
    print("Will use mock predictions for backtest")

class RealDataBacktester:
    def __init__(self):
        self.initial_capital = 1000.0
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.daily_pnl = []
        
        # Risk management settings
        self.max_position_size = 100.0
        self.risk_per_trade = 0.02
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        self.max_positions = 3
        
        # Initialize ML models
        try:
            self.enhanced_model = EnhancedMLModel()
            self.fibonacci_model = FibonacciMLModel()
            self.ml_available = True
            print("✅ ML models loaded successfully")
        except Exception as e:
            print(f"⚠️  ML models not available: {e}")
            self.ml_available = False
    
    def fetch_real_data(self, symbol: str = "BTCUSD", days: int = 30) -> pd.DataFrame:
        """Fetch real market data from a public API"""
        print(f"📊 Fetching real market data for {symbol} ({days} days)...")
        
        try:
            # Using CoinGecko API for real Bitcoin data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Convert to timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Fetch data from CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': start_ts,
                'to': end_ts
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            volumes = data['total_volumes']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume data
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('timestamp', inplace=True)
            
            df = df.join(volume_df, how='left')
            
            # Generate OHLC from close prices (simplified)
            df['open'] = df['close'].shift(1)
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))
            
            # Fill NaN values
            df.fillna(method='ffill', inplace=True)
            df.dropna(inplace=True)
            
            print(f"✅ Fetched {len(df)} data points from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching real data: {e}")
            print("📊 Generating synthetic data as fallback...")
            return self.generate_synthetic_data(symbol, days)
    
    def generate_synthetic_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate realistic synthetic data as fallback"""
        periods = days * 24  # Hourly data
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             periods=periods, freq='h')
        
        # Generate realistic price movement
        initial_price = 45000 if symbol == "BTCUSD" else 3000
        returns = np.random.normal(0, 0.02, periods)  # 2% volatility
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(1000, 10000, periods)
        }, index=dates)
        
        # Generate OHLC
        df['open'] = df['close'].shift(1)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, len(df)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, len(df)))
        
        df.fillna(method='ffill', inplace=True)
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame, lookback: int = 50) -> np.ndarray:
        """Prepare features for ML models"""
        features = []
        
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]
            
            # OHLCV features
            ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values
            
            # Technical indicators
            close_prices = window['close'].values
            
            # Moving averages
            ma_short = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else close_prices[-1]
            ma_long = np.mean(close_prices[-20:]) if len(close_prices) >= 20 else close_prices[-1]
            
            # Price momentum
            momentum = (close_prices[-1] / close_prices[0] - 1) if close_prices[0] != 0 else 0
            
            # Volatility
            volatility = np.std(close_prices) / np.mean(close_prices) if np.mean(close_prices) != 0 else 0
            
            # Create feature vector (pad to 20 features)
            feature_row = [
                close_prices[-1], ma_short, ma_long, momentum, volatility,
                window['volume'].iloc[-1], 
                (window['high'].iloc[-1] - window['low'].iloc[-1]) / window['close'].iloc[-1],
                *np.random.random(13)  # Padding features
            ]
            
            features.append(feature_row)
        
        # Reshape for ML models [batch_size, sequence_length, features]
        return np.array(features).reshape(len(features), 1, 20)
    
    def get_ml_predictions(self, features: np.ndarray, current_price: float) -> Dict:
        """Get predictions from ML models"""
        if not self.ml_available:
            # Mock predictions
            return {
                'enhanced': {
                    'action': np.random.choice(['buy', 'sell', 'hold']),
                    'confidence': np.random.uniform(0.5, 0.9)
                },
                'fibonacci': {
                    'action': np.random.choice(['buy', 'sell', 'hold']),
                    'confidence': np.random.uniform(0.5, 0.9),
                    'fibonacci_level': np.random.choice([0.236, 0.382, 0.5, 0.618, 0.786])
                }
            }
        
        try:
            # Get predictions from both models
            enhanced_pred = self.enhanced_model.predict(features[-1:])
            fibonacci_pred = self.fibonacci_model.predict(features[-1:])
            
            return {
                'enhanced': enhanced_pred,
                'fibonacci': fibonacci_pred
            }
        except Exception as e:
            print(f"⚠️  ML prediction error: {e}")
            return self.get_ml_predictions(features, current_price)  # Fallback to mock
    
    def calculate_position_size(self, confidence: float, current_price: float) -> float:
        """Calculate position size based on risk management"""
        # Risk-based position sizing
        risk_amount = self.current_capital * self.risk_per_trade
        stop_loss_distance = current_price * self.stop_loss_pct
        
        if stop_loss_distance > 0:
            position_size = min(
                risk_amount / stop_loss_distance,
                self.max_position_size,
                self.current_capital * 0.3  # Max 30% of capital per position
            )
        else:
            position_size = self.max_position_size
        
        # Adjust by confidence
        position_size *= confidence
        
        return max(10, position_size)  # Minimum $10 position
    
    def execute_trade(self, action: str, price: float, size: float, timestamp: datetime, 
                     confidence: float, model_source: str) -> Dict:
        """Execute a trade and update portfolio"""
        if action == 'hold' or len(self.positions) >= self.max_positions:
            return None
        
        # Check if we have enough capital
        required_capital = size
        if required_capital > self.current_capital:
            return None
        
        trade = {
            'id': f"trade_{len(self.trades)}",
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'size': size,
            'confidence': confidence,
            'model_source': model_source,
            'stop_loss': price * (1 - self.stop_loss_pct) if action == 'buy' else price * (1 + self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct) if action == 'buy' else price * (1 - self.take_profit_pct),
            'status': 'open'
        }
        
        # Update capital and positions
        self.current_capital -= required_capital
        self.positions.append(trade)
        self.trades.append(trade)
        
        return trade
    
    def check_exit_conditions(self, current_price: float, timestamp: datetime):
        """Check if any positions should be closed"""
        positions_to_close = []
        
        for i, position in enumerate(self.positions):
            if position['status'] != 'open':
                continue
            
            should_close = False
            exit_reason = ""
            exit_price = current_price
            
            # Check stop loss
            if position['action'] == 'buy' and current_price <= position['stop_loss']:
                should_close = True
                exit_reason = "stop_loss"
                exit_price = position['stop_loss']
            elif position['action'] == 'sell' and current_price >= position['stop_loss']:
                should_close = True
                exit_reason = "stop_loss"
                exit_price = position['stop_loss']
            
            # Check take profit
            elif position['action'] == 'buy' and current_price >= position['take_profit']:
                should_close = True
                exit_reason = "take_profit"
                exit_price = position['take_profit']
            elif position['action'] == 'sell' and current_price <= position['take_profit']:
                should_close = True
                exit_reason = "take_profit"
                exit_price = position['take_profit']
            
            if should_close:
                # Calculate P&L
                if position['action'] == 'buy':
                    pnl = (exit_price - position['price']) * (position['size'] / position['price'])
                else:
                    pnl = (position['price'] - exit_price) * (position['size'] / position['price'])
                
                # Close position
                position['exit_timestamp'] = timestamp
                position['exit_price'] = exit_price
                position['exit_reason'] = exit_reason
                position['pnl'] = pnl
                position['status'] = 'closed'
                
                # Update capital
                self.current_capital += position['size'] + pnl
                
                positions_to_close.append(i)
        
        # Remove closed positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)
    
    def run_backtest(self, symbol: str = "BTCUSD", days: int = 30) -> Dict:
        """Run the complete backtest"""
        print(f"🚀 Starting backtest for {symbol} over {days} days...")
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        
        # Fetch real data
        df = self.fetch_real_data(symbol, days)
        
        # Prepare ML features
        print("🧠 Preparing ML features...")
        features = self.prepare_ml_features(df)
        
        print(f"📈 Running backtest on {len(features)} data points...")
        
        # Run backtest
        for i, (timestamp, row) in enumerate(df.iloc[50:].iterrows()):  # Skip first 50 for features
            current_price = row['close']
            
            # Check exit conditions for existing positions
            self.check_exit_conditions(current_price, timestamp)
            
            # Get ML predictions
            if i < len(features):
                predictions = self.get_ml_predictions(features, current_price)
                
                # Ensemble decision making
                enhanced_pred = predictions['enhanced']
                fibonacci_pred = predictions['fibonacci']
                
                # Simple ensemble: require agreement or high confidence
                if enhanced_pred['action'] == fibonacci_pred['action']:
                    action = enhanced_pred['action']
                    confidence = (enhanced_pred['confidence'] + fibonacci_pred['confidence']) / 2
                    model_source = "ensemble_agreement"
                elif enhanced_pred['confidence'] > 0.8:
                    action = enhanced_pred['action']
                    confidence = enhanced_pred['confidence']
                    model_source = "enhanced_ml"
                elif fibonacci_pred['confidence'] > 0.8:
                    action = fibonacci_pred['action']
                    confidence = fibonacci_pred['confidence']
                    model_source = "fibonacci_ml"
                else:
                    action = 'hold'
                    confidence = 0.5
                    model_source = "no_consensus"
                
                # Execute trade if conditions are met
                if action != 'hold' and confidence > 0.65:
                    position_size = self.calculate_position_size(confidence, current_price)
                    trade = self.execute_trade(action, current_price, position_size, 
                                             timestamp, confidence, model_source)
                    
                    if trade:
                        print(f"📊 {timestamp.strftime('%Y-%m-%d %H:%M')} - {action.upper()} "
                              f"${position_size:.2f} at ${current_price:.2f} "
                              f"(confidence: {confidence:.2f})")
            
            # Track daily P&L
            if i % 24 == 0:  # Daily summary
                portfolio_value = self.current_capital + sum(
                    pos['size'] + (current_price - pos['price']) * (pos['size'] / pos['price'])
                    if pos['action'] == 'buy' else 
                    pos['size'] + (pos['price'] - current_price) * (pos['size'] / pos['price'])
                    for pos in self.positions
                )
                daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
                self.daily_pnl.append({
                    'date': timestamp.date(),
                    'portfolio_value': portfolio_value,
                    'daily_return': daily_return,
                    'open_positions': len(self.positions)
                })
        
        # Close any remaining positions at final price
        final_price = df['close'].iloc[-1]
        final_timestamp = df.index[-1]
        
        for position in self.positions:
            if position['action'] == 'buy':
                pnl = (final_price - position['price']) * (position['size'] / position['price'])
            else:
                pnl = (position['price'] - final_price) * (position['size'] / position['price'])
            
            position['exit_timestamp'] = final_timestamp
            position['exit_price'] = final_price
            position['exit_reason'] = "backtest_end"
            position['pnl'] = pnl
            position['status'] = 'closed'
            
            self.current_capital += position['size'] + pnl
        
        # Calculate final results
        return self.calculate_results()
    
    def calculate_results(self) -> Dict:
        """Calculate backtest results and metrics"""
        final_capital = self.current_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        winning_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        if self.daily_pnl:
            daily_returns = [d['daily_return'] for d in self.daily_pnl]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365) if np.std(daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        portfolio_values = [d['portfolio_value'] for d in self.daily_pnl] if self.daily_pnl else [self.initial_capital, final_capital]
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        results = {
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
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'trades': closed_trades,
            'daily_pnl': self.daily_pnl
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted backtest results"""
        print("\n" + "="*60)
        print("🎯 BACKTEST RESULTS")
        print("="*60)
        print(f"💰 Initial Capital:     ${results['initial_capital']:,.2f}")
        print(f"💰 Final Capital:       ${results['final_capital']:,.2f}")
        print(f"📈 Total Return:        ${results['total_return']:,.2f} ({results['total_return_pct']:.2f}%)")
        print(f"📊 Total Trades:        {results['total_trades']}")
        print(f"✅ Winning Trades:      {results['winning_trades']} ({results['win_rate']:.1f}%)")
        print(f"❌ Losing Trades:       {results['losing_trades']}")
        print(f"💵 Average Win:         ${results['avg_win']:.2f}")
        print(f"💸 Average Loss:        ${results['avg_loss']:.2f}")
        print(f"⚖️  Profit Factor:       {results['profit_factor']:.2f}")
        print(f"📉 Max Drawdown:        {results['max_drawdown']:.2f}%")
        print(f"📊 Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print("="*60)
        
        # Performance assessment
        if results['total_return_pct'] > 5:
            print("🎉 EXCELLENT: Strong positive returns!")
        elif results['total_return_pct'] > 0:
            print("✅ GOOD: Positive returns achieved")
        elif results['total_return_pct'] > -5:
            print("⚠️  ACCEPTABLE: Small losses, needs optimization")
        else:
            print("❌ POOR: Significant losses, major optimization needed")
        
        if results['win_rate'] > 60:
            print("🎯 HIGH WIN RATE: Good trade selection")
        elif results['win_rate'] > 50:
            print("✅ DECENT WIN RATE: Acceptable performance")
        else:
            print("⚠️  LOW WIN RATE: Need better entry signals")
        
        if results['max_drawdown'] < 10:
            print("🛡️  LOW RISK: Good risk management")
        elif results['max_drawdown'] < 20:
            print("⚠️  MODERATE RISK: Acceptable drawdown")
        else:
            print("🚨 HIGH RISK: Reduce position sizes")
    
    def save_results(self, results: Dict, filename: str = None):
        """Save backtest results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        results_copy = results.copy()
        for trade in results_copy['trades']:
            if 'timestamp' in trade:
                trade['timestamp'] = trade['timestamp'].isoformat()
            if 'exit_timestamp' in trade:
                trade['exit_timestamp'] = trade['exit_timestamp'].isoformat()
        
        for daily in results_copy['daily_pnl']:
            if 'date' in daily:
                daily['date'] = daily['date'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

def main():
    """Main function to run the backtest"""
    print("🚀 SmartMarketOOPS Real Data Backtest")
    print("=====================================")
    
    # Create backtest instance
    backtester = RealDataBacktester()
    
    # Run backtest
    try:
        results = backtester.run_backtest(symbol="BTCUSD", days=30)
        
        # Print results
        backtester.print_results(results)
        
        # Save results
        backtester.save_results(results)
        
        # Recommendation for live trading
        print("\n🔮 LIVE TRADING RECOMMENDATION:")
        if (results['total_return_pct'] > 2 and 
            results['win_rate'] > 55 and 
            results['max_drawdown'] < 15):
            print("✅ APPROVED: System ready for live trading")
            print("💡 Suggested settings:")
            print("   - Start with 50% of planned capital")
            print("   - Monitor closely for first week")
            print("   - Gradually increase position sizes")
        else:
            print("⚠️  CAUTION: Consider optimization before live trading")
            print("💡 Suggestions:")
            if results['total_return_pct'] <= 2:
                print("   - Improve ML model confidence thresholds")
            if results['win_rate'] <= 55:
                print("   - Refine entry/exit signals")
            if results['max_drawdown'] >= 15:
                print("   - Reduce position sizes and risk per trade")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()