#!/usr/bin/env python3
"""
Chart Analysis Tool

This tool creates visual charts showing:
1. Actual price action with candlesticks
2. Entry points with reasons and confidence
3. Exit points with reasons and P&L
4. Multi-timeframe context
5. Missed opportunities analysis
6. Support/resistance levels

This will help us visually analyze why we have low returns despite good win rate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingChartAnalyzer:
    """
    Visual chart analyzer for trading performance
    """
    
    def __init__(self):
        """Initialize chart analyzer"""
        self.trades = []
        self.price_data = None
        self.htf_data = {}
        
    def analyze_trades_on_chart(self, symbol: str = "BTCUSDT", 
                               start_date: str = "2024-01-01", 
                               end_date: str = "2024-02-01"):
        """Analyze trades visually on charts"""
        
        print("📊 VISUAL CHART ANALYSIS")
        print("=" * 30)
        print("Creating visual analysis of entries and exits on actual charts")
        
        try:
            # Run backtest to get trades
            trades_data = self._run_backtest_for_analysis(symbol, start_date, end_date)
            
            if not trades_data:
                print("❌ No trade data available")
                return None
            
            # Get price data for charting
            price_data = self._get_price_data(symbol, start_date, end_date)
            
            if price_data is None:
                print("❌ No price data available")
                return None
            
            # Create comprehensive charts
            self._create_trading_charts(trades_data, price_data, symbol)
            
            # Analyze trade quality
            analysis = self._analyze_trade_quality(trades_data, price_data)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Chart analysis failed: {e}")
            return None
    
    def _run_backtest_for_analysis(self, symbol: str, start_date: str, end_date: str) -> Optional[Dict]:
        """Run backtest to get trade data"""
        try:
            from multi_timeframe_system_corrected import run_multi_timeframe_optimization
            
            print("🔄 Running backtest to collect trade data...")
            result = run_multi_timeframe_optimization()
            
            if result and 'trades' in result:
                return result
            else:
                print("❌ No trades generated in backtest")
                return None
                
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return None
    
    def _get_price_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get detailed price data for charting"""
        try:
            from production_real_data_backtester import RealDataFetcher
            
            print("📡 Fetching detailed price data for charting...")
            data_fetcher = RealDataFetcher()
            
            # Get 15-minute data for detailed analysis
            price_data = data_fetcher.fetch_real_data(symbol, start_date, end_date, "15m")
            
            if price_data is not None and len(price_data) > 0:
                print(f"✅ Loaded {len(price_data)} price candles")
                return price_data
            else:
                print("❌ No price data retrieved")
                return None
                
        except Exception as e:
            print(f"❌ Price data fetch failed: {e}")
            return None
    
    def _create_trading_charts(self, trades_data: Dict, price_data: pd.DataFrame, symbol: str):
        """Create comprehensive trading charts"""
        
        trades = trades_data.get('trades', [])
        if not trades:
            print("❌ No trades to chart")
            return
        
        # Prepare data
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        price_data = price_data.set_index('timestamp')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'{symbol} Trading Analysis - Entries & Exits on Chart', fontsize=16, fontweight='bold')
        
        # Chart 1: Price action with trades
        self._plot_price_with_trades(ax1, price_data, trades, "Main Chart: Price Action with Trades")
        
        # Chart 2: Profit analysis
        self._plot_profit_analysis(ax2, trades, "Trade Profit Analysis")
        
        # Chart 3: Trade timing analysis
        self._plot_timing_analysis(ax3, trades, price_data, "Trade Timing & Duration Analysis")
        
        plt.tight_layout()
        plt.savefig('trading_analysis_chart.png', dpi=300, bbox_inches='tight')
        print("✅ Chart saved as 'trading_analysis_chart.png'")
        plt.show()
        
        # Create detailed individual trade charts
        self._create_individual_trade_charts(trades, price_data, symbol)
    
    def _plot_price_with_trades(self, ax, price_data: pd.DataFrame, trades: List[Dict], title: str):
        """Plot price action with entry/exit points"""
        
        # Plot candlestick-style price action
        dates = price_data.index
        
        # Plot price line
        ax.plot(dates, price_data['close'], color='black', linewidth=1, alpha=0.7, label='Price')
        
        # Plot high/low range
        ax.fill_between(dates, price_data['low'], price_data['high'], 
                       alpha=0.1, color='gray', label='High-Low Range')
        
        # Plot moving averages for context
        if len(price_data) > 20:
            ma_20 = price_data['close'].rolling(20).mean()
            ma_50 = price_data['close'].rolling(50).mean()
            ax.plot(dates, ma_20, color='blue', alpha=0.5, linewidth=1, label='MA20')
            ax.plot(dates, ma_50, color='red', alpha=0.5, linewidth=1, label='MA50')
        
        # Plot trades
        entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
        exit_trades = [t for t in trades if t['action'] in ['exit', 'partial_exit']]
        
        # Plot entries
        for trade in entry_trades:
            timestamp = pd.to_datetime(trade['timestamp'])
            price = trade['price']
            confidence = trade.get('confidence', 0)
            
            if trade['action'] == 'buy':
                ax.scatter(timestamp, price, color='green', s=100, marker='^', 
                          alpha=0.8, label='Buy Entry' if trade == entry_trades[0] else "")
                ax.annotate(f"BUY\n{confidence:.1%}", (timestamp, price), 
                           xytext=(5, 15), textcoords='offset points', 
                           fontsize=8, ha='left', color='green', fontweight='bold')
            else:  # sell
                ax.scatter(timestamp, price, color='red', s=100, marker='v', 
                          alpha=0.8, label='Sell Entry' if trade == entry_trades[0] else "")
                ax.annotate(f"SELL\n{confidence:.1%}", (timestamp, price), 
                           xytext=(5, -20), textcoords='offset points', 
                           fontsize=8, ha='left', color='red', fontweight='bold')
        
        # Plot exits
        for trade in exit_trades:
            timestamp = pd.to_datetime(trade['timestamp'])
            price = trade['price']
            reason = trade.get('reason', 'exit')
            profit = trade.get('profit', 0)
            
            color = 'darkgreen' if profit > 0 else 'darkred'
            ax.scatter(timestamp, price, color=color, s=80, marker='x', 
                      alpha=0.8, label='Exit' if trade == exit_trades[0] else "")
            ax.annotate(f"EXIT\n{reason}\n{profit:.1%}", (timestamp, price), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=7, ha='left', color=color)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_profit_analysis(self, ax, trades: List[Dict], title: str):
        """Plot profit analysis"""
        
        # Calculate cumulative P&L
        exit_trades = [t for t in trades if t['action'] in ['exit', 'partial_exit']]
        
        if not exit_trades:
            ax.text(0.5, 0.5, 'No completed trades', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        timestamps = [pd.to_datetime(t['timestamp']) for t in exit_trades]
        profits = [t.get('profit', 0) for t in exit_trades]
        cumulative_pnl = np.cumsum(profits)
        
        # Plot cumulative P&L
        ax.plot(timestamps, cumulative_pnl, color='blue', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Color bars based on profit/loss
        colors = ['green' if p > 0 else 'red' for p in profits]
        ax.bar(timestamps, profits, color=colors, alpha=0.6, width=0.5)
        
        # Add profit labels
        for i, (ts, profit) in enumerate(zip(timestamps, profits)):
            ax.annotate(f'{profit:.1%}', (ts, profit), 
                       xytext=(0, 5 if profit > 0 else -15), textcoords='offset points',
                       ha='center', fontsize=8, fontweight='bold')
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Profit/Loss (%)')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_timing_analysis(self, ax, trades: List[Dict], price_data: pd.DataFrame, title: str):
        """Plot trade timing and duration analysis"""
        
        # Analyze trade durations and timing
        entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
        exit_trades = [t for t in trades if t['action'] in ['exit', 'partial_exit']]
        
        if not entry_trades or not exit_trades:
            ax.text(0.5, 0.5, 'Insufficient trade data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Calculate trade durations
        durations = []
        profits = []
        
        for entry in entry_trades:
            entry_time = pd.to_datetime(entry['timestamp'])
            entry_id = entry.get('trade_id', '')
            
            # Find matching exit
            matching_exits = [e for e in exit_trades if e.get('trade_id') == entry_id]
            if matching_exits:
                exit_trade = matching_exits[0]
                exit_time = pd.to_datetime(exit_trade['timestamp'])
                duration_hours = (exit_time - entry_time).total_seconds() / 3600
                profit = exit_trade.get('profit', 0)
                
                durations.append(duration_hours)
                profits.append(profit)
        
        if durations:
            # Scatter plot of duration vs profit
            colors = ['green' if p > 0 else 'red' for p in profits]
            ax.scatter(durations, profits, c=colors, alpha=0.7, s=60)
            
            # Add trend line
            if len(durations) > 1:
                z = np.polyfit(durations, profits, 1)
                p = np.poly1d(z)
                ax.plot(durations, p(durations), "r--", alpha=0.8, linewidth=1)
            
            # Add labels
            for i, (dur, prof) in enumerate(zip(durations, profits)):
                ax.annotate(f'{i+1}', (dur, prof), xytext=(3, 3), textcoords='offset points',
                           fontsize=8, ha='left')
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Trade Duration (hours)')
            ax.set_ylabel('Profit/Loss (%)')
            
            # Add statistics
            avg_duration = np.mean(durations)
            avg_profit = np.mean(profits)
            ax.text(0.02, 0.98, f'Avg Duration: {avg_duration:.1f}h\nAvg Profit: {avg_profit:.2%}', 
                   transform=ax.transAxes, va='top', ha='left', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _create_individual_trade_charts(self, trades: List[Dict], price_data: pd.DataFrame, symbol: str):
        """Create detailed charts for individual trades"""
        
        entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
        exit_trades = [t for t in trades if t['action'] in ['exit', 'partial_exit']]
        
        print(f"\n📈 Creating individual trade analysis charts...")
        
        # Analyze top 3 most significant trades
        significant_trades = []
        
        for entry in entry_trades[:3]:  # Analyze first 3 trades
            entry_time = pd.to_datetime(entry['timestamp'])
            entry_id = entry.get('trade_id', '')
            
            # Find matching exit
            matching_exits = [e for e in exit_trades if e.get('trade_id') == entry_id]
            if matching_exits:
                exit_trade = matching_exits[0]
                exit_time = pd.to_datetime(exit_trade['timestamp'])
                
                # Get price data around this trade
                start_time = entry_time - timedelta(hours=6)
                end_time = exit_time + timedelta(hours=6)
                
                trade_data = price_data[
                    (price_data.index >= start_time) & 
                    (price_data.index <= end_time)
                ].copy()
                
                if len(trade_data) > 10:
                    significant_trades.append({
                        'entry': entry,
                        'exit': exit_trade,
                        'data': trade_data
                    })
        
        # Create individual charts
        for i, trade_info in enumerate(significant_trades):
            self._plot_individual_trade(trade_info, i+1, symbol)
        
        print(f"✅ Created {len(significant_trades)} individual trade charts")
    
    def _plot_individual_trade(self, trade_info: Dict, trade_num: int, symbol: str):
        """Plot detailed analysis of individual trade"""
        
        entry = trade_info['entry']
        exit = trade_info['exit']
        data = trade_info['data']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot price action
        dates = data.index
        ax.plot(dates, data['close'], color='black', linewidth=2, label='Price')
        ax.fill_between(dates, data['low'], data['high'], alpha=0.1, color='gray')
        
        # Plot moving averages
        if len(data) > 10:
            ma_10 = data['close'].rolling(10).mean()
            ma_20 = data['close'].rolling(20).mean()
            ax.plot(dates, ma_10, color='blue', alpha=0.7, linewidth=1, label='MA10')
            ax.plot(dates, ma_20, color='red', alpha=0.7, linewidth=1, label='MA20')
        
        # Mark entry
        entry_time = pd.to_datetime(entry['timestamp'])
        entry_price = entry['price']
        entry_confidence = entry.get('confidence', 0)
        
        if entry['action'] == 'buy':
            ax.scatter(entry_time, entry_price, color='green', s=200, marker='^', 
                      zorder=5, label='Entry')
            ax.axvline(entry_time, color='green', linestyle='--', alpha=0.5)
        else:
            ax.scatter(entry_time, entry_price, color='red', s=200, marker='v', 
                      zorder=5, label='Entry')
            ax.axvline(entry_time, color='red', linestyle='--', alpha=0.5)
        
        # Mark exit
        exit_time = pd.to_datetime(exit['timestamp'])
        exit_price = exit['price']
        profit = exit.get('profit', 0)
        reason = exit.get('reason', 'exit')
        
        exit_color = 'darkgreen' if profit > 0 else 'darkred'
        ax.scatter(exit_time, exit_price, color=exit_color, s=200, marker='x', 
                  zorder=5, label='Exit')
        ax.axvline(exit_time, color=exit_color, linestyle='--', alpha=0.5)
        
        # Draw trade line
        ax.plot([entry_time, exit_time], [entry_price, exit_price], 
               color=exit_color, linewidth=3, alpha=0.7, zorder=4)
        
        # Calculate what could have been achieved
        trade_period_data = data[(data.index >= entry_time) & (data.index <= exit_time)]
        if len(trade_period_data) > 0:
            if entry['action'] == 'buy':
                max_profit_price = trade_period_data['high'].max()
                max_potential_profit = (max_profit_price - entry_price) / entry_price
                ax.axhline(max_profit_price, color='lightgreen', linestyle=':', alpha=0.7, 
                          label=f'Max Potential: {max_potential_profit:.1%}')
            else:
                min_profit_price = trade_period_data['low'].min()
                max_potential_profit = (entry_price - min_profit_price) / entry_price
                ax.axhline(min_profit_price, color='lightgreen', linestyle=':', alpha=0.7, 
                          label=f'Max Potential: {max_potential_profit:.1%}')
        
        # Add annotations
        duration = (exit_time - entry_time).total_seconds() / 3600
        
        ax.set_title(f'Trade #{trade_num} - {symbol}\n'
                    f'{entry["action"].upper()} @ {entry_price:.2f} → EXIT @ {exit_price:.2f}\n'
                    f'Profit: {profit:.2%} | Duration: {duration:.1f}h | Reason: {reason}\n'
                    f'Confidence: {entry_confidence:.1%}', 
                    fontweight='bold', fontsize=12)
        
        ax.set_ylabel('Price ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'trade_{trade_num}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Trade #{trade_num} chart saved as 'trade_{trade_num}_analysis.png'")
        plt.show()
    
    def _analyze_trade_quality(self, trades_data: Dict, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade quality from charts"""
        
        trades = trades_data.get('trades', [])
        entry_trades = [t for t in trades if t['action'] in ['buy', 'sell']]
        exit_trades = [t for t in trades if t['action'] in ['exit', 'partial_exit']]
        
        analysis = {
            'total_trades': len(entry_trades),
            'visual_insights': [],
            'timing_analysis': {},
            'missed_opportunities': [],
            'recommendations': []
        }
        
        # Analyze timing quality
        if entry_trades and exit_trades:
            # Check if entries are at good levels
            good_entries = 0
            for entry in entry_trades:
                entry_time = pd.to_datetime(entry['timestamp'])
                entry_price = entry['price']
                
                # Get surrounding price data
                start_check = entry_time - timedelta(hours=2)
                end_check = entry_time + timedelta(hours=2)
                
                surrounding_data = price_data[
                    (price_data.index >= start_check) & 
                    (price_data.index <= end_check)
                ]
                
                if len(surrounding_data) > 0:
                    if entry['action'] == 'buy':
                        # Good buy if near recent low
                        recent_low = surrounding_data['low'].min()
                        if entry_price <= recent_low * 1.005:  # Within 0.5% of low
                            good_entries += 1
                    else:  # sell
                        # Good sell if near recent high
                        recent_high = surrounding_data['high'].max()
                        if entry_price >= recent_high * 0.995:  # Within 0.5% of high
                            good_entries += 1
            
            entry_quality = good_entries / len(entry_trades) if entry_trades else 0
            analysis['timing_analysis']['entry_quality'] = entry_quality
            
            if entry_quality > 0.6:
                analysis['visual_insights'].append("✅ Good entry timing - entries near local extremes")
            else:
                analysis['visual_insights'].append("❌ Poor entry timing - entries not at optimal levels")
        
        # Analyze exit quality
        early_exits = 0
        for exit in exit_trades:
            exit_time = pd.to_datetime(exit['timestamp'])
            profit = exit.get('profit', 0)
            
            # Check if we exited too early
            post_exit_data = price_data[price_data.index > exit_time].head(20)  # Next 5 hours
            
            if len(post_exit_data) > 0 and profit > 0:
                # For profitable trades, check if price continued in our favor
                exit_price = exit['price']
                max_price_after = post_exit_data['high'].max()
                min_price_after = post_exit_data['low'].min()
                
                # Estimate if we could have made more
                potential_additional_profit = max((max_price_after - exit_price) / exit_price, 
                                                (exit_price - min_price_after) / exit_price)
                
                if potential_additional_profit > profit * 0.5:  # Could have made 50% more
                    early_exits += 1
        
        if exit_trades:
            early_exit_rate = early_exits / len(exit_trades)
            analysis['timing_analysis']['early_exit_rate'] = early_exit_rate
            
            if early_exit_rate > 0.5:
                analysis['visual_insights'].append("❌ High early exit rate - leaving profits on table")
                analysis['recommendations'].append("Consider wider trailing stops or profit targets")
            else:
                analysis['visual_insights'].append("✅ Good exit timing - not exiting too early")
        
        # Overall assessment
        total_return = trades_data.get('total_return', 0)
        win_rate = len([t for t in exit_trades if t.get('profit', 0) > 0]) / len(exit_trades) if exit_trades else 0
        
        if win_rate > 0.6 and total_return < 0.01:
            analysis['visual_insights'].append("🔍 High win rate but low returns - classic overtrading or small profits")
            analysis['recommendations'].append("Focus on fewer, higher-quality trades")
            analysis['recommendations'].append("Increase position sizes on high-confidence trades")
        
        return analysis


def run_chart_analysis():
    """Run comprehensive chart analysis"""
    
    print("📊 COMPREHENSIVE CHART ANALYSIS")
    print("=" * 40)
    print("Visual analysis of entries and exits on actual price charts")
    
    analyzer = TradingChartAnalyzer()
    
    # Run analysis
    result = analyzer.analyze_trades_on_chart()
    
    if result:
        print(f"\n🎯 VISUAL ANALYSIS RESULTS:")
        print("=" * 30)
        
        for insight in result.get('visual_insights', []):
            print(f"{insight}")
        
        print(f"\n💡 CHART-BASED RECOMMENDATIONS:")
        for rec in result.get('recommendations', []):
            print(f"• {rec}")
        
        timing = result.get('timing_analysis', {})
        if timing:
            print(f"\n📊 TIMING ANALYSIS:")
            print(f"Entry Quality: {timing.get('entry_quality', 0):.1%}")
            print(f"Early Exit Rate: {timing.get('early_exit_rate', 0):.1%}")
        
        return result
    
    else:
        print("❌ Chart analysis failed")
        return None


if __name__ == "__main__":
    print("📊 Starting Visual Chart Analysis")
    print("This will create detailed charts showing entries/exits on actual price action")
    
    result = run_chart_analysis()
    
    if result:
        print(f"\n🎉 CHART ANALYSIS COMPLETED!")
        print("Check the generated PNG files for visual analysis")
        print("Charts saved:")
        print("• trading_analysis_chart.png - Main overview")
        print("• trade_1_analysis.png - Detailed trade #1")
        print("• trade_2_analysis.png - Detailed trade #2") 
        print("• trade_3_analysis.png - Detailed trade #3")
    else:
        print(f"\n❌ Chart analysis failed")
        print("Check data availability and try again")
