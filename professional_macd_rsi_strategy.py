#!/usr/bin/env python3
"""
PROFESSIONAL MACD + RSI + MEAN REVERSION STRATEGY
=================================================

Based on QuantifiedStrategies research showing 73% win rate
Uses MACD, RSI, and a third mean reversion filter for confluence
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

class ProfessionalMACDRSIStrategy:
    def __init__(self):
        self.exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
            'timeout': 30000,
        })
        print("✅ Professional Strategy System Initialized (Binance LIVE)")
    
    def fetch_real_data(self, symbol='BTC/USDT', days_back=60):
        """Fetch comprehensive real data for professional analysis"""
        print(f"📈 Fetching {days_back} days of REAL {symbol} data for professional analysis...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        since = int(start_time.timestamp() * 1000)
        
        all_data = []
        current_since = since
        
        # Fetch sufficient data for proper indicators
        for i in range(15):  # More chunks for better data
            try:
                chunk = self.exchange.fetch_ohlcv(symbol, '1h', current_since, limit=1000)
                if chunk:
                    all_data.extend(chunk)
                    current_since = chunk[-1][0] + (60 * 60 * 1000)
                    print(f"   Chunk {i+1}: {len(chunk)} candles")
                if current_since >= int(end_time.timestamp() * 1000):
                    break
                if len(all_data) > days_back * 24:  # Got enough data
                    break
            except Exception as e:
                print(f"   Chunk failed: {e}")
                break
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Fetched {len(df)} REAL hourly candles")
        print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price Range: ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")
        
        return df
    
    def calculate_advanced_indicators(self, df):
        """Calculate professional indicators for the 73% win rate strategy"""
        print("🔬 Calculating advanced indicators...")
        
        # 1. MACD (12, 26, 9) - Standard settings
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 2. RSI (14) - Standard settings
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. MEAN REVERSION FILTER - The secret sauce!
        # This is what makes the 73% win rate possible
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['bb_period'] = 20
        df['bb_std'] = 2
        df['bb_middle'] = df['close'].rolling(df['bb_period']).mean()
        df['bb_std_dev'] = df['close'].rolling(df['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std_dev'] * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (df['bb_std_dev'] * df['bb_std'])
        
        # Mean reversion indicators
        df['price_bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['rsi_ma'] = df['rsi'].rolling(10).mean()
        df['mean_reversion_signal'] = 0
        
        # Mean reversion conditions (the third filter)
        for i in range(50, len(df)):
            # Oversold mean reversion signal
            if (df.loc[i, 'price_bb_position'] < 0.2 and  # Near lower BB
                df.loc[i, 'rsi'] < 35 and                 # RSI oversold
                df.loc[i, 'close'] < df.loc[i, 'sma_20']):  # Below SMA
                df.loc[i, 'mean_reversion_signal'] = 1  # Bullish mean reversion
            
            # Overbought mean reversion signal  
            elif (df.loc[i, 'price_bb_position'] > 0.8 and  # Near upper BB
                  df.loc[i, 'rsi'] > 65 and                 # RSI overbought
                  df.loc[i, 'close'] > df.loc[i, 'sma_20']):  # Above SMA
                df.loc[i, 'mean_reversion_signal'] = -1  # Bearish mean reversion
        
        # 4. Additional confirmation indicators
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # 5. Trend strength
        df['atr'] = self._calculate_atr(df, 14)
        df['trend_strength'] = abs(df['close'] - df['sma_50']) / df['atr']
        
        print(f"✅ Calculated all professional indicators")
        return df
    
    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window).mean()
    
    def generate_professional_signals(self, df):
        """Generate signals using the 73% win rate strategy"""
        print("⚡ Generating professional trading signals...")
        
        signals = []
        
        for i in range(60, len(df)):  # Start after all indicators are stable
            current = df.iloc[i]
            prev = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            
            signal_strength = 0
            signal_type = None
            reasons = []
            
            # === PRIMARY SIGNAL: MACD Crossover ===
            macd_bullish_cross = (current['macd'] > current['macd_signal'] and 
                                prev['macd'] <= prev['macd_signal'])
            macd_bearish_cross = (current['macd'] < current['macd_signal'] and 
                                prev['macd'] >= prev['macd_signal'])
            
            if macd_bullish_cross:
                signal_strength += 3
                signal_type = "LONG"
                reasons.append("MACD Bullish Crossover")
            elif macd_bearish_cross:
                signal_strength += 3
                signal_type = "SHORT" 
                reasons.append("MACD Bearish Crossover")
            
            # === SECONDARY FILTER: RSI Confirmation ===
            if signal_type == "LONG":
                if current['rsi'] > 30 and current['rsi'] < 70:  # Not extreme
                    signal_strength += 2
                    reasons.append("RSI Neutral Zone")
                if current['rsi'] > prev['rsi']:  # RSI trending up
                    signal_strength += 1
                    reasons.append("RSI Momentum Up")
            
            elif signal_type == "SHORT":
                if current['rsi'] > 30 and current['rsi'] < 70:  # Not extreme
                    signal_strength += 2
                    reasons.append("RSI Neutral Zone")
                if current['rsi'] < prev['rsi']:  # RSI trending down
                    signal_strength += 1
                    reasons.append("RSI Momentum Down")
            
            # === THIRD FILTER: MEAN REVERSION (The Secret Sauce!) ===
            if signal_type == "LONG":
                # Look for bullish mean reversion setup
                if current['mean_reversion_signal'] == 1:
                    signal_strength += 3  # High weight for mean reversion
                    reasons.append("Mean Reversion: Oversold Bounce")
                elif current['price_bb_position'] < 0.3:  # Near lower BB
                    signal_strength += 2
                    reasons.append("Near Bollinger Lower Band")
            
            elif signal_type == "SHORT":
                # Look for bearish mean reversion setup
                if current['mean_reversion_signal'] == -1:
                    signal_strength += 3  # High weight for mean reversion
                    reasons.append("Mean Reversion: Overbought Pullback")
                elif current['price_bb_position'] > 0.7:  # Near upper BB
                    signal_strength += 2
                    reasons.append("Near Bollinger Upper Band")
            
            # === ADDITIONAL FILTERS ===
            # Volume confirmation
            if current['volume_ratio'] > 1.2:  # Above average volume
                signal_strength += 1
                reasons.append("High Volume Confirmation")
            
            # Trend strength filter
            if current['trend_strength'] > 0.5:  # Strong trend
                signal_strength += 1
                reasons.append("Strong Trend Environment")
            
            # === SIGNAL GENERATION ===
            # Require high signal strength for the 73% win rate
            if signal_strength >= 7 and signal_type:  # High threshold
                confidence = min(signal_strength / 10.0, 0.95)
                
                signals.append({
                    'timestamp': current['timestamp'],
                    'signal': signal_type,
                    'price': current['close'],
                    'confidence': confidence,
                    'strength': signal_strength,
                    'reasons': '; '.join(reasons),
                    'rsi': current['rsi'],
                    'macd': current['macd'],
                    'macd_signal': current['macd_signal'],
                    'bb_position': current['price_bb_position'],
                    'mean_rev_signal': current['mean_reversion_signal'],
                    'volume_ratio': current['volume_ratio']
                })
        
        print(f"📊 Generated {len(signals)} HIGH-QUALITY signals")
        print(f"   Average confidence: {np.mean([s['confidence'] for s in signals]):.1%}")
        
        return signals
    
    def simulate_professional_trading(self, df, signals):
        """Simulate trading with professional risk management"""
        print("💼 Simulating professional trading...")
        
        # Professional configuration
        initial_capital = 10000.0
        current_capital = initial_capital
        position_size = 0.06  # 6% per trade (conservative)
        leverage = 2.0  # Conservative leverage
        fee_rate = 0.001  # 0.1% trading fee
        
        # Professional risk management
        stop_loss_atr_multiple = 2.5  # Stop loss based on ATR
        take_profit_ratio = 2.0  # 2:1 risk-reward
        max_open_positions = 1
        max_daily_trades = 3
        
        trades = []
        positions = []
        daily_trades = {}
        
        for signal in signals:
            signal_date = signal['timestamp'].date()
            
            # Daily trade limit
            if daily_trades.get(signal_date, 0) >= max_daily_trades:
                continue
            
            # Position limit
            if len(positions) >= max_open_positions:
                continue
            
            # High confidence threshold (for 73% win rate)
            if signal['confidence'] < 0.8:
                continue
            
            # Calculate position sizing
            entry_price = signal['price']
            position_value = current_capital * position_size
            
            # Find ATR for stop loss
            signal_index = df[df['timestamp'] <= signal['timestamp']].index[-1]
            current_atr = df.iloc[signal_index]['atr']
            
            # Calculate professional stop loss and take profit
            if signal['signal'] == 'LONG':
                stop_loss = entry_price - (current_atr * stop_loss_atr_multiple)
                risk_per_share = entry_price - stop_loss
                take_profit = entry_price + (risk_per_share * take_profit_ratio)
            else:  # SHORT
                stop_loss = entry_price + (current_atr * stop_loss_atr_multiple)
                risk_per_share = stop_loss - entry_price
                take_profit = entry_price - (risk_per_share * take_profit_ratio)
            
            position = {
                'signal': signal['signal'],
                'entry_price': entry_price,
                'entry_time': signal['timestamp'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_value': position_value,
                'quantity': position_value / entry_price,
                'confidence': signal['confidence'],
                'reasons': signal['reasons'],
                'atr': current_atr
            }
            
            positions.append(position)
            daily_trades[signal_date] = daily_trades.get(signal_date, 0) + 1
            
            print(f"   📈 {signal['signal']} position opened at ${entry_price:,.0f} (conf: {signal['confidence']:.1%})")
        
        # Process exits with real market data
        for pos in positions:
            future_data = df[df['timestamp'] > pos['entry_time']].copy()
            
            exit_price = None
            exit_time = None
            exit_reason = None
            
            # Check each subsequent candle
            for _, candle in future_data.iterrows():
                # Stop loss check
                if pos['signal'] == 'LONG' and candle['low'] <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    exit_time = candle['timestamp']
                    exit_reason = "Stop Loss"
                    break
                elif pos['signal'] == 'SHORT' and candle['high'] >= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    exit_time = candle['timestamp']
                    exit_reason = "Stop Loss"
                    break
                
                # Take profit check
                if pos['signal'] == 'LONG' and candle['high'] >= pos['take_profit']:
                    exit_price = pos['take_profit']
                    exit_time = candle['timestamp']
                    exit_reason = "Take Profit"
                    break
                elif pos['signal'] == 'SHORT' and candle['low'] <= pos['take_profit']:
                    exit_price = pos['take_profit']
                    exit_time = candle['timestamp']
                    exit_reason = "Take Profit"
                    break
                
                # Time-based exit (max holding period)
                if (candle['timestamp'] - pos['entry_time']).days >= 10:
                    exit_price = candle['close']
                    exit_time = candle['timestamp']
                    exit_reason = "Time Exit"
                    break
            
            # Default exit if no conditions met
            if exit_price is None:
                exit_price = df['close'].iloc[-1]
                exit_time = df['timestamp'].iloc[-1]
                exit_reason = "End of Data"
            
            # Calculate P&L
            if pos['signal'] == 'LONG':
                pnl_points = exit_price - pos['entry_price']
            else:
                pnl_points = pos['entry_price'] - exit_price
            
            pnl_usd = pnl_points * pos['quantity'] * leverage
            fees = (pos['entry_price'] + exit_price) * pos['quantity'] * fee_rate
            net_pnl = pnl_usd - fees
            
            current_capital += net_pnl
            
            trade = {
                **pos,
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'pnl_usd': net_pnl,
                'pnl_pct': (net_pnl / pos['position_value']) * 100 * leverage,
                'duration_hours': (exit_time - pos['entry_time']).total_seconds() / 3600
            }
            
            trades.append(trade)
        
        # Calculate comprehensive metrics
        total_return = current_capital - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        winning_trades = [t for t in trades if t['pnl_usd'] > 0]
        losing_trades = [t for t in trades if t['pnl_usd'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t['pnl_usd'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_usd'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum([t['pnl_usd'] for t in winning_trades]) / 
                          sum([t['pnl_usd'] for t in losing_trades])) if losing_trades else float('inf')
        
        # Max drawdown calculation
        capital_curve = [initial_capital]
        for trade in trades:
            capital_curve.append(capital_curve[-1] + trade['pnl_usd'])
        
        running_max = np.maximum.accumulate(capital_curve)
        drawdown = (capital_curve - running_max) / running_max * 100
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 1 else 0
        
        # Sharpe ratio (simplified)
        if trades:
            returns = [t['pnl_pct'] for t in trades]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        results = {
            'performance_metrics': {
                'initial_capital': initial_capital,
                'final_capital': current_capital,
                'total_return_usd': total_return,
                'total_return_pct': total_return_pct,
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'largest_win': max([t['pnl_usd'] for t in trades]) if trades else 0,
                'largest_loss': min([t['pnl_usd'] for t in trades]) if trades else 0,
                'avg_trade_duration': np.mean([t['duration_hours'] for t in trades]) if trades else 0
            },
            'trades': trades,
            'strategy': 'MACD + RSI + Mean Reversion (73% Win Rate)',
            'data_source': 'Binance (REAL)'
        }
        
        print(f"✅ Professional simulation completed")
        print(f"   Total trades: {len(trades)}")
        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   Final capital: ${current_capital:,.2f}")
        
        return results

def run_professional_strategy():
    """Run the complete professional MACD+RSI strategy"""
    print("=" * 80)
    print("🎯 PROFESSIONAL MACD + RSI + MEAN REVERSION STRATEGY")
    print("   Based on 73% Win Rate Research (QuantifiedStrategies)")
    print("   Using REAL Binance market data")
    print("=" * 80)
    
    # Initialize strategy
    strategy = ProfessionalMACDRSIStrategy()
    
    # Fetch comprehensive real data
    real_data = strategy.fetch_real_data('BTC/USDT', days_back=90)
    
    # Calculate advanced indicators
    enhanced_data = strategy.calculate_advanced_indicators(real_data)
    
    # Generate professional signals
    signals = strategy.generate_professional_signals(enhanced_data)
    
    # Simulate professional trading
    results = strategy.simulate_professional_trading(enhanced_data, signals)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"professional_strategy_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Display professional results
    print("\n" + "=" * 80)
    print("📊 PROFESSIONAL STRATEGY RESULTS")
    print("=" * 80)
    
    perf = results['performance_metrics']
    
    print(f"\n💰 PERFORMANCE SUMMARY:")
    print(f"   Strategy: {results['strategy']}")
    print(f"   Data Source: {results['data_source']}")
    print(f"   Initial Capital: ${perf['initial_capital']:,.2f}")
    print(f"   Final Capital: ${perf['final_capital']:,.2f}")
    print(f"   Total Return: ${perf['total_return_usd']:,.2f} ({perf['total_return_pct']:.2f}%)")
    print(f"   Max Drawdown: {perf['max_drawdown']:.2f}%")
    print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    
    print(f"\n📈 TRADING STATISTICS:")
    print(f"   Total Trades: {perf['total_trades']}")
    print(f"   Winning Trades: {perf['winning_trades']}")
    print(f"   Losing Trades: {perf['losing_trades']}")
    print(f"   Win Rate: {perf['win_rate']:.1f}% (Target: 73%)")
    print(f"   Profit Factor: {perf['profit_factor']:.2f}")
    print(f"   Average Win: ${perf['avg_win']:,.2f}")
    print(f"   Average Loss: ${perf['avg_loss']:,.2f}")
    print(f"   Largest Win: ${perf['largest_win']:,.2f}")
    print(f"   Largest Loss: ${perf['largest_loss']:,.2f}")
    print(f"   Avg Trade Duration: {perf['avg_trade_duration']:.1f} hours")
    
    # Performance assessment
    if perf['win_rate'] >= 60 and perf['total_return_pct'] > 10 and perf['max_drawdown'] < 15:
        assessment = "EXCELLENT - Approaching research benchmarks"
        emoji = "🎯"
    elif perf['win_rate'] >= 50 and perf['total_return_pct'] > 5 and perf['max_drawdown'] < 20:
        assessment = "GOOD - Strong professional performance"
        emoji = "👍"
    elif perf['total_return_pct'] > 0 and perf['max_drawdown'] < 30:
        assessment = "ACCEPTABLE - Profitable but needs optimization"
        emoji = "✅"
    else:
        assessment = "NEEDS IMPROVEMENT - Requires strategy refinement"
        emoji = "⚠️"
    
    print(f"\n{emoji} OVERALL ASSESSMENT: {assessment}")
    print(f"\n💾 Results saved to: {results_file}")
    
    # Show sample trades
    if results['trades']:
        print(f"\n📋 SAMPLE PROFESSIONAL TRADES:")
        for i, trade in enumerate(results['trades'][:8]):
            pnl_sign = "+" if trade['pnl_usd'] > 0 else ""
            duration = f"{trade['duration_hours']:.1f}h"
            print(f"   {i+1}. {trade['signal']:5s} ${trade['entry_price']:7,.0f} → ${trade['exit_price']:7,.0f} "
                  f"({pnl_sign}${trade['pnl_usd']:6,.0f}) [{trade['exit_reason']}] {duration}")
    
    return results

if __name__ == "__main__":
    run_professional_strategy()
