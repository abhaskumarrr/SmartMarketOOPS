#!/usr/bin/env python3
"""
INSTITUTIONAL GRADE BACKTESTER WITH REAL DATA AND CHARTS
Professional backtesting system showing exactly where and why trades are taken
Demonstrates the difference between spray-and-pray vs institutional quality
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json

# Import our institutional trader
from institutional_grade_trader import InstitutionalGradeTrader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstitutionalBacktester(InstitutionalGradeTrader):
    """Backtesting engine for institutional-grade trading with detailed charting"""
    
    def __init__(self):
        super().__init__()
        
        # Backtesting specific parameters
        self.backtest_start = None
        self.backtest_end = None
        self.trades = []
        self.portfolio_history = []
        self.daily_signals_generated = {}
        
        # Chart settings
        self.chart_width = 16
        self.chart_height = 12
        
        logger.info("🏛️ INSTITUTIONAL BACKTESTER WITH REAL DATA INITIALIZED")
        logger.info(f"   📊 Will show exact entry/exit points and reasoning")
        logger.info(f"   📈 Quality over quantity - max {self.max_daily_signals} signals/day")

    def run_comprehensive_backtest(self, 
                                 symbol: str = 'BTC/USDT',
                                 days_back: int = 30) -> Dict:
        """Run comprehensive backtest with real data and generate detailed analysis"""
        
        logger.info("🎯 STARTING INSTITUTIONAL-GRADE BACKTESTING")
        logger.info("=" * 80)
        logger.info(f"   📊 Symbol: {symbol}")
        logger.info(f"   📅 Period: Last {days_back} days")
        logger.info(f"   🏛️ Institutional confluence threshold: {self.confluence_threshold}%")
        logger.info(f"   🎯 Max daily signals: {self.max_daily_signals}")
        
        # Fetch multi-timeframe data
        data = self.fetch_multi_timeframe_data(symbol)
        if not data:
            logger.error("❌ Failed to fetch market data for backtesting")
            return {}
        
        # Use 15m timeframe for backtesting (entry timeframe)
        backtest_data = data['entry'].copy()
        
        # Limit to specified period
        self.backtest_end = backtest_data.index[-1]
        self.backtest_start = self.backtest_end - timedelta(days=days_back)
        backtest_data = backtest_data[backtest_data.index >= self.backtest_start]
        
        logger.info(f"   📊 Backtesting period: {self.backtest_start} to {self.backtest_end}")
        logger.info(f"   📈 Total candles: {len(backtest_data)}")
        
        # Initialize portfolio tracking
        self.current_capital = self.initial_capital
        portfolio_value = self.initial_capital
        open_positions = []
        
        # Track signals generated per day
        signals_today = 0
        current_date = None
        
        # Process each candle
        for i in range(100, len(backtest_data)):  # Start after sufficient lookback
            current_time = backtest_data.index[i]
            current_candle = backtest_data.iloc[i]
            current_price = current_candle['close']
            
            # Reset daily signal counter
            if current_date is None or current_time.date() != current_date:
                current_date = current_time.date()
                signals_today = 0
                self.daily_signals_generated[current_date] = 0
            
            # Update data context for current timepoint
            current_data = {
                'weekly': data['weekly'][data['weekly'].index <= current_time],
                'daily': data['daily'][data['daily'].index <= current_time],
                'h4': data['h4'][data['h4'].index <= current_time],
                'h1': data['h1'][data['h1'].index <= current_time],
                'entry': backtest_data.iloc[:i+1]
            }
            
            # Skip if insufficient data
            if any(len(timeframe_data) < 20 for timeframe_data in current_data.values()):
                continue
            
            # Check for signal generation (only if under daily limit)
            if signals_today < self.max_daily_signals:
                signal = self.generate_institutional_signal(current_data)
                
                if signal:
                    signals_today += 1
                    self.daily_signals_generated[current_date] += 1
                    
                    # Execute the trade
                    trade = self.execute_backtest_trade(signal, current_time, current_price)
                    if trade:
                        open_positions.append(trade)
                        logger.info(f"🎯 INSTITUTIONAL TRADE EXECUTED at {current_time}")
                        logger.info(f"   📊 {trade['direction'].upper()} at ${current_price:.2f}")
                        logger.info(f"   ⚖️ Confluence: {signal['confluence_score']:.1f}%")
                        logger.info(f"   🎲 Risk: {signal['risk_pct']:.2f}%")
            
            # Check open positions for exit conditions
            positions_to_close = []
            for pos_idx, position in enumerate(open_positions):
                exit_result = self.check_position_exit(position, current_candle, current_time)
                
                if exit_result:
                    exit_price = exit_result['exit_price']
                    exit_reason = exit_result['reason']
                    
                    # Calculate P&L
                    if position['direction'] == 'bullish':
                        pnl = (exit_price - position['entry_price']) * position['position_size']
                    else:
                        pnl = (position['entry_price'] - exit_price) * position['position_size']
                    
                    # Update position
                    position.update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'return_pct': (pnl / (position['entry_price'] * position['position_size'])) * 100,
                        'hold_time': current_time - position['entry_time']
                    })
                    
                    # Update capital
                    self.current_capital += pnl
                    self.trades.append(position)
                    positions_to_close.append(pos_idx)
                    
                    logger.info(f"🏁 TRADE CLOSED at {current_time}")
                    logger.info(f"   💰 P&L: ${pnl:.2f} ({position['return_pct']:.2f}%)")
                    logger.info(f"   📄 Reason: {exit_reason}")
            
            # Remove closed positions
            for idx in reversed(positions_to_close):
                open_positions.pop(idx)
            
            # Track portfolio value
            current_portfolio_value = self.current_capital
            for position in open_positions:
                if position['direction'] == 'bullish':
                    unrealized_pnl = (current_price - position['entry_price']) * position['position_size']
                else:
                    unrealized_pnl = (position['entry_price'] - current_price) * position['position_size']
                current_portfolio_value += unrealized_pnl
            
            self.portfolio_history.append({
                'timestamp': current_time,
                'portfolio_value': current_portfolio_value,
                'capital': self.current_capital,
                'open_positions': len(open_positions)
            })
        
        # Close any remaining open positions
        for position in open_positions:
            final_price = backtest_data.iloc[-1]['close']
            if position['direction'] == 'bullish':
                pnl = (final_price - position['entry_price']) * position['position_size']
            else:
                pnl = (position['entry_price'] - final_price) * position['position_size']
            
            position.update({
                'exit_time': backtest_data.index[-1],
                'exit_price': final_price,
                'exit_reason': 'End of backtest',
                'pnl': pnl,
                'return_pct': (pnl / (position['entry_price'] * position['position_size'])) * 100,
                'hold_time': backtest_data.index[-1] - position['entry_time']
            })
            
            self.current_capital += pnl
            self.trades.append(position)
        
        # Generate comprehensive results
        results = self.generate_backtest_results(symbol, backtest_data)
        
        # Create detailed charts
        self.create_comprehensive_charts(symbol, backtest_data, results)
        
        return results

    def execute_backtest_trade(self, signal: Dict, timestamp: datetime, current_price: float) -> Optional[Dict]:
        """Execute a trade in backtesting environment"""
        
        # Calculate position size based on risk
        risk_amount = abs(signal['entry_price'] - signal['stop_loss'])
        risk_multiplier = 1.0
        
        if signal['confluence_score'] >= 95:
            risk_multiplier = 2.0
        elif signal['confluence_score'] >= 90:
            risk_multiplier = 1.5
        
        position_risk = min(self.max_risk_percent, self.base_risk_percent * risk_multiplier) / 100
        max_loss = self.current_capital * position_risk
        position_size = max_loss / risk_amount
        
        trade = {
            'entry_time': timestamp,
            'entry_price': current_price,
            'direction': signal['direction'],
            'position_size': position_size,
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'confluence_score': signal['confluence_score'],
            'risk_amount': risk_amount,
            'risk_pct': signal['risk_pct'],
            'rr_ratio': signal['rr_ratio'],
            'primary_trend': signal['primary_trend']['trend'],
            'confluence_factors': signal['confluence_factors']
        }
        
        return trade

    def check_position_exit(self, position: Dict, current_candle: pd.Series, current_time: datetime) -> Optional[Dict]:
        """Check if position should be exited"""
        
        current_price = current_candle['close']
        high_price = current_candle['high']
        low_price = current_candle['low']
        
        if position['direction'] == 'bullish':
            # Check take profit
            if high_price >= position['take_profit']:
                return {'exit_price': position['take_profit'], 'reason': 'Take Profit'}
            
            # Check stop loss
            if low_price <= position['stop_loss']:
                return {'exit_price': position['stop_loss'], 'reason': 'Stop Loss'}
        
        else:  # bearish
            # Check take profit
            if low_price <= position['take_profit']:
                return {'exit_price': position['take_profit'], 'reason': 'Take Profit'}
            
            # Check stop loss
            if high_price >= position['stop_loss']:
                return {'exit_price': position['stop_loss'], 'reason': 'Stop Loss'}
        
        return None

    def generate_backtest_results(self, symbol: str, backtest_data: pd.DataFrame) -> Dict:
        """Generate comprehensive backtest results"""
        
        if not self.trades:
            logger.warning("⚠️ No trades executed during backtest period")
            return {
                'symbol': symbol,
                'period': f"{self.backtest_start} to {self.backtest_end}",
                'total_trades': 0,
                'message': 'No institutional-grade setups found with 85%+ confluence'
            }
        
        # Calculate performance metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Daily signal distribution
        total_days = len(set(self.daily_signals_generated.keys()))
        avg_signals_per_day = sum(self.daily_signals_generated.values()) / total_days if total_days > 0 else 0
        
        # Risk metrics
        returns = [t['return_pct'] for t in self.trades]
        max_drawdown = self.calculate_max_drawdown()
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        results = {
            'symbol': symbol,
            'period': f"{self.backtest_start.strftime('%Y-%m-%d')} to {self.backtest_end.strftime('%Y-%m-%d')}",
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_signals_per_day': avg_signals_per_day,
            'total_days': total_days,
            'confluence_threshold': self.confluence_threshold,
            'trades_detail': self.trades,
            'daily_signals': dict(self.daily_signals_generated)
        }
        
        # Print summary
        logger.info("\n🏛️ INSTITUTIONAL BACKTESTING RESULTS")
        logger.info("=" * 60)
        logger.info(f"📊 Symbol: {symbol}")
        logger.info(f"📅 Period: {results['period']}")
        logger.info(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"💰 Final Capital: ${self.current_capital:,.2f}")
        logger.info(f"📈 Total Return: {total_return:.2f}%")
        logger.info(f"🎯 Total Trades: {total_trades}")
        logger.info(f"✅ Win Rate: {win_rate:.1f}%")
        logger.info(f"💵 Profit Factor: {profit_factor:.2f}")
        logger.info(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"📊 Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"🎲 Avg Signals/Day: {avg_signals_per_day:.1f}")
        logger.info("=" * 60)
        
        return results

    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history"""
        if not self.portfolio_history:
            return 0.0
        
        values = [p['portfolio_value'] for p in self.portfolio_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd

    def create_comprehensive_charts(self, symbol: str, backtest_data: pd.DataFrame, results: Dict):
        """Create detailed charts showing entry/exit points and analysis"""
        
        logger.info("📊 Creating comprehensive trading charts...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.chart_width, self.chart_height))
        fig.suptitle(f'🏛️ Institutional Grade Trading Analysis - {symbol}', fontsize=16, fontweight='bold')
        
        # **CHART 1: Price Action with Entry/Exit Points**
        ax1.plot(backtest_data.index, backtest_data['close'], 'b-', linewidth=1, alpha=0.7, label='Price')
        ax1.plot(backtest_data.index, backtest_data['ema_21'], 'orange', linewidth=1, alpha=0.8, label='EMA 21')
        ax1.plot(backtest_data.index, backtest_data['ema_50'], 'red', linewidth=1, alpha=0.8, label='EMA 50')
        
        # Plot trade entry/exit points
        for trade in self.trades:
            entry_color = 'green' if trade['direction'] == 'bullish' else 'red'
            exit_color = 'darkgreen' if trade['pnl'] > 0 else 'darkred'
            
            # Entry point
            ax1.scatter(trade['entry_time'], trade['entry_price'], 
                       color=entry_color, s=100, marker='^' if trade['direction'] == 'bullish' else 'v',
                       zorder=5, label='Entry' if trade == self.trades[0] else '')
            
            # Exit point
            ax1.scatter(trade['exit_time'], trade['exit_price'],
                       color=exit_color, s=100, marker='x', zorder=5,
                       label='Exit' if trade == self.trades[0] else '')
            
            # Draw trade lines
            ax1.plot([trade['entry_time'], trade['exit_time']], 
                    [trade['entry_price'], trade['exit_price']], 
                    color=exit_color, linewidth=2, alpha=0.6)
        
        ax1.set_title('Price Action with Institutional Entry/Exit Points')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # **CHART 2: Portfolio Performance**
        if self.portfolio_history:
            portfolio_times = [p['timestamp'] for p in self.portfolio_history]
            portfolio_values = [p['portfolio_value'] for p in self.portfolio_history]
            portfolio_returns = [(v / self.initial_capital - 1) * 100 for v in portfolio_values]
            
            ax2.plot(portfolio_times, portfolio_returns, 'green', linewidth=2)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.fill_between(portfolio_times, portfolio_returns, alpha=0.3, color='green')
            
        ax2.set_title(f'Portfolio Performance ({results["total_return_pct"]:.1f}% Total Return)')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        
        # **CHART 3: Trade Analysis**
        if self.trades:
            trade_returns = [t['return_pct'] for t in self.trades]
            trade_numbers = list(range(1, len(trade_returns) + 1))
            
            colors = ['green' if r > 0 else 'red' for r in trade_returns]
            ax3.bar(trade_numbers, trade_returns, color=colors, alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax3.set_title(f'Individual Trade Returns (Win Rate: {results["win_rate_pct"]:.1f}%)')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # **CHART 4: Daily Signal Generation**
        if self.daily_signals_generated:
            signal_dates = list(self.daily_signals_generated.keys())
            signal_counts = list(self.daily_signals_generated.values())
            
            ax4.bar(signal_dates, signal_counts, alpha=0.7, color='blue')
            ax4.axhline(y=self.max_daily_signals, color='red', linestyle='--', 
                       label=f'Max Daily Limit ({self.max_daily_signals})')
        
        ax4.set_title(f'Daily Signal Generation (Avg: {results["avg_signals_per_day"]:.1f}/day)')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Signals Generated')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_filename = f"institutional_backtest_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Chart saved as: {chart_filename}")
        
        plt.show()
        
        # Save detailed trade log
        self.save_trade_log(symbol, results)

    def save_trade_log(self, symbol: str, results: Dict):
        """Save detailed trade log to JSON file"""
        
        # Prepare trade data for JSON serialization
        trades_for_json = []
        for trade in self.trades:
            trade_dict = trade.copy()
            # Convert datetime objects to strings
            trade_dict['entry_time'] = trade['entry_time'].isoformat()
            trade_dict['exit_time'] = trade['exit_time'].isoformat()
            trade_dict['hold_time'] = str(trade['hold_time'])
            trades_for_json.append(trade_dict)
        
        # Prepare results for JSON
        results_for_json = results.copy()
        results_for_json['trades_detail'] = trades_for_json
        results_for_json['daily_signals'] = {str(k): v for k, v in results['daily_signals'].items()}
        
        # Save to file
        log_filename = f"institutional_trades_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_filename, 'w') as f:
            json.dump(results_for_json, f, indent=2, default=str)
        
        logger.info(f"💾 Trade log saved as: {log_filename}")


def main():
    """Main execution function"""
    print("🏛️ INSTITUTIONAL GRADE BACKTESTER WITH REAL DATA")
    print("=" * 80)
    print("🎯 Professional-grade backtesting showing EXACTLY where and why trades are taken")
    print("📊 Real market data with institutional confluence requirements")
    print("⚖️ 85%+ confluence threshold - Quality over quantity")
    print("📈 Maximum 5 signals per day - Selective trading")
    print("🏛️ Multi-timeframe analysis with Order Blocks and Fair Value Gaps")
    print()
    
    # Initialize institutional backtester
    backtester = InstitutionalBacktester()
    
    # Run comprehensive backtest
    results = backtester.run_comprehensive_backtest('BTC/USDT', days_back=30)
    
    if results and 'total_trades' in results:
        print(f"\n🎊 INSTITUTIONAL BACKTESTING COMPLETE!")
        print(f"📊 {results['total_trades']} high-quality institutional trades executed")
        print(f"📈 {results['total_return_pct']:.2f}% total return with {results['win_rate_pct']:.1f}% win rate")
        print(f"🎯 Average {results['avg_signals_per_day']:.1f} signals per day (vs 46 in spray-and-pray)")
        print(f"⚖️ All trades met {results['confluence_threshold']}%+ institutional confluence")
        print("\n📊 Charts and detailed trade logs have been generated!")
    else:
        print("\n⏳ No institutional-grade setups found in the backtesting period.")
        print("This demonstrates the selectivity of institutional trading - patience for quality setups.")

if __name__ == "__main__":
    main()