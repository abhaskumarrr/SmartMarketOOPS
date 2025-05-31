#!/usr/bin/env python3
"""
Professional Entry/Exit Analysis

Analyzing why we have high win rate but low returns:

Common Issues:
1. Taking profits too early (cutting winners short)
2. Not cutting losses quickly enough (letting losers run)
3. Poor position sizing relative to expected move
4. Transaction costs eating into small profits
5. Exit signals triggering too early on noise
6. Not riding trends long enough

This analysis will identify the specific issues and provide solutions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeAnalyzer:
    """
    Professional trade analysis to identify entry/exit issues
    """
    
    def __init__(self):
        """Initialize trade analyzer"""
        self.trades = []
        self.detailed_analysis = {}
        
    def analyze_trade_performance(self, trades: List[Dict], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detailed trade performance"""
        
        if not trades:
            return {"error": "No trades to analyze"}
        
        print("🔍 PROFESSIONAL TRADE ANALYSIS")
        print("=" * 50)
        print("Analyzing entry/exit efficiency and profit optimization")
        
        # Group trades into complete round trips
        round_trips = self._create_round_trips(trades)
        
        if not round_trips:
            return {"error": "No complete round trips found"}
        
        print(f"\n📊 Found {len(round_trips)} complete round trips")
        
        # Analyze each aspect
        results = {
            "round_trips": round_trips,
            "profit_analysis": self._analyze_profit_patterns(round_trips, price_data),
            "timing_analysis": self._analyze_entry_exit_timing(round_trips, price_data),
            "position_sizing": self._analyze_position_sizing(round_trips),
            "transaction_costs": self._analyze_transaction_costs(round_trips),
            "missed_opportunities": self._analyze_missed_opportunities(round_trips, price_data),
            "recommendations": self._generate_recommendations(round_trips, price_data)
        }
        
        return results
    
    def _create_round_trips(self, trades: List[Dict]) -> List[Dict]:
        """Create complete round trip trades"""
        round_trips = []
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x['timestamp'])
        
        i = 0
        while i < len(sorted_trades) - 1:
            entry_trade = sorted_trades[i]
            
            # Find matching exit trade
            for j in range(i + 1, len(sorted_trades)):
                exit_trade = sorted_trades[j]
                
                # Check if this is a valid round trip
                if ((entry_trade['action'] == 'buy' and exit_trade['action'] == 'sell') or
                    (entry_trade['action'] == 'sell' and exit_trade['action'] == 'buy')):
                    
                    # Calculate trade metrics
                    if entry_trade['action'] == 'buy':
                        pnl_pct = (exit_trade['price'] - entry_trade['price']) / entry_trade['price']
                    else:
                        pnl_pct = (entry_trade['price'] - exit_trade['price']) / entry_trade['price']
                    
                    duration = exit_trade['timestamp'] - entry_trade['timestamp']
                    
                    round_trip = {
                        'entry': entry_trade,
                        'exit': exit_trade,
                        'direction': entry_trade['action'],
                        'entry_price': entry_trade['price'],
                        'exit_price': exit_trade['price'],
                        'pnl_pct': pnl_pct,
                        'duration': duration,
                        'duration_hours': duration.total_seconds() / 3600,
                        'entry_confidence': entry_trade.get('confidence', 0),
                        'exit_confidence': exit_trade.get('confidence', 0)
                    }
                    
                    round_trips.append(round_trip)
                    i = j + 1
                    break
            else:
                i += 1
        
        return round_trips
    
    def _analyze_profit_patterns(self, round_trips: List[Dict], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze profit/loss patterns"""
        
        if not round_trips:
            return {}
        
        pnls = [rt['pnl_pct'] for rt in round_trips]
        winners = [pnl for pnl in pnls if pnl > 0]
        losers = [pnl for pnl in pnls if pnl < 0]
        
        analysis = {
            'total_trades': len(round_trips),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(round_trips) if round_trips else 0,
            'avg_winner': np.mean(winners) if winners else 0,
            'avg_loser': np.mean(losers) if losers else 0,
            'largest_winner': max(winners) if winners else 0,
            'largest_loser': min(losers) if losers else 0,
            'profit_factor': abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf'),
            'expectancy': np.mean(pnls) if pnls else 0
        }
        
        # Calculate reward-to-risk ratio
        if analysis['avg_loser'] != 0:
            analysis['reward_risk_ratio'] = abs(analysis['avg_winner'] / analysis['avg_loser'])
        else:
            analysis['reward_risk_ratio'] = float('inf')
        
        print(f"\n💰 PROFIT ANALYSIS")
        print("=" * 20)
        print(f"Win Rate: {analysis['win_rate']:.1%}")
        print(f"Average Winner: {analysis['avg_winner']:.3%}")
        print(f"Average Loser: {analysis['avg_loser']:.3%}")
        print(f"Reward/Risk Ratio: {analysis['reward_risk_ratio']:.2f}")
        print(f"Profit Factor: {analysis['profit_factor']:.2f}")
        print(f"Expectancy: {analysis['expectancy']:.3%}")
        
        # Identify the core issue
        if analysis['win_rate'] > 0.6 and analysis['expectancy'] < 0.005:
            print(f"\n🚨 CORE ISSUE IDENTIFIED:")
            print(f"HIGH WIN RATE ({analysis['win_rate']:.1%}) but LOW EXPECTANCY ({analysis['expectancy']:.3%})")
            
            if analysis['reward_risk_ratio'] < 1.5:
                print(f"❌ CUTTING WINNERS SHORT: R/R ratio {analysis['reward_risk_ratio']:.2f} too low")
                print(f"   Average winner: {analysis['avg_winner']:.3%}")
                print(f"   Average loser: {analysis['avg_loser']:.3%}")
                print(f"   Need R/R ratio > 2.0 for profitable system")
            
            if abs(analysis['avg_loser']) > analysis['avg_winner']:
                print(f"❌ LETTING LOSSES RUN: Losses larger than wins")
        
        return analysis
    
    def _analyze_entry_exit_timing(self, round_trips: List[Dict], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze entry and exit timing efficiency"""
        
        timing_analysis = {
            'early_exits': 0,
            'late_entries': 0,
            'optimal_exits_missed': 0,
            'avg_duration_hours': 0,
            'duration_analysis': {}
        }
        
        durations = [rt['duration_hours'] for rt in round_trips]
        timing_analysis['avg_duration_hours'] = np.mean(durations) if durations else 0
        
        # Analyze each trade's timing
        for rt in round_trips:
            entry_time = rt['entry']['timestamp']
            exit_time = rt['exit']['timestamp']
            
            # Find price data around this trade
            trade_data = price_data[
                (price_data['timestamp'] >= entry_time) & 
                (price_data['timestamp'] <= exit_time + timedelta(hours=24))
            ].copy()
            
            if len(trade_data) < 2:
                continue
            
            entry_price = rt['entry_price']
            exit_price = rt['exit_price']
            
            if rt['direction'] == 'buy':
                # For long trades, find the highest price after entry
                max_price_after_entry = trade_data['high'].max()
                potential_profit = (max_price_after_entry - entry_price) / entry_price
                actual_profit = rt['pnl_pct']
                
                # Check if we exited too early
                if potential_profit > actual_profit * 1.5:  # Could have made 50% more
                    timing_analysis['early_exits'] += 1
                    
            else:  # sell/short trades
                # For short trades, find the lowest price after entry
                min_price_after_entry = trade_data['low'].min()
                potential_profit = (entry_price - min_price_after_entry) / entry_price
                actual_profit = rt['pnl_pct']
                
                if potential_profit > actual_profit * 1.5:
                    timing_analysis['early_exits'] += 1
        
        # Duration analysis
        if durations:
            timing_analysis['duration_analysis'] = {
                'min_duration': min(durations),
                'max_duration': max(durations),
                'median_duration': np.median(durations),
                'std_duration': np.std(durations)
            }
        
        print(f"\n⏰ TIMING ANALYSIS")
        print("=" * 20)
        print(f"Average trade duration: {timing_analysis['avg_duration_hours']:.1f} hours")
        print(f"Early exits detected: {timing_analysis['early_exits']}")
        print(f"Early exit rate: {timing_analysis['early_exits']/len(round_trips):.1%}")
        
        if timing_analysis['early_exits'] / len(round_trips) > 0.3:
            print(f"🚨 HIGH EARLY EXIT RATE: {timing_analysis['early_exits']/len(round_trips):.1%}")
            print(f"   This is likely causing low returns despite high win rate")
        
        return timing_analysis
    
    def _analyze_position_sizing(self, round_trips: List[Dict]) -> Dict[str, Any]:
        """Analyze position sizing efficiency"""
        
        # Extract position sizes (if available)
        position_sizes = []
        confidences = []
        
        for rt in round_trips:
            if 'shares' in rt['entry']:
                # Estimate position size from shares and price
                position_value = rt['entry']['shares'] * rt['entry_price']
                position_sizes.append(position_value)
            
            confidences.append(rt['entry_confidence'])
        
        analysis = {
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'confidence_range': (min(confidences), max(confidences)) if confidences else (0, 0),
            'position_sizing_issues': []
        }
        
        # Check if position sizing correlates with confidence
        if len(confidences) > 5:
            high_conf_trades = [rt for rt in round_trips if rt['entry_confidence'] > 0.7]
            low_conf_trades = [rt for rt in round_trips if rt['entry_confidence'] < 0.5]
            
            if high_conf_trades and low_conf_trades:
                high_conf_returns = [rt['pnl_pct'] for rt in high_conf_trades]
                low_conf_returns = [rt['pnl_pct'] for rt in low_conf_trades]
                
                analysis['high_conf_avg_return'] = np.mean(high_conf_returns)
                analysis['low_conf_avg_return'] = np.mean(low_conf_returns)
                
                if analysis['high_conf_avg_return'] > analysis['low_conf_avg_return']:
                    analysis['position_sizing_issues'].append("Should size larger on high confidence trades")
        
        print(f"\n📏 POSITION SIZING ANALYSIS")
        print("=" * 25)
        print(f"Average entry confidence: {analysis['avg_confidence']:.1%}")
        
        if analysis['position_sizing_issues']:
            for issue in analysis['position_sizing_issues']:
                print(f"⚠️  {issue}")
        
        return analysis
    
    def _analyze_transaction_costs(self, round_trips: List[Dict]) -> Dict[str, Any]:
        """Analyze transaction cost impact"""
        
        # Assume 0.1% transaction cost per trade (0.2% round trip)
        transaction_cost_per_round_trip = 0.002  # 0.2%
        
        gross_pnls = [rt['pnl_pct'] for rt in round_trips]
        net_pnls = [pnl - transaction_cost_per_round_trip for pnl in gross_pnls]
        
        analysis = {
            'gross_expectancy': np.mean(gross_pnls) if gross_pnls else 0,
            'net_expectancy': np.mean(net_pnls) if net_pnls else 0,
            'cost_impact': transaction_cost_per_round_trip,
            'cost_impact_pct': (transaction_cost_per_round_trip / abs(np.mean(gross_pnls))) if gross_pnls and np.mean(gross_pnls) != 0 else 0
        }
        
        print(f"\n💸 TRANSACTION COST ANALYSIS")
        print("=" * 25)
        print(f"Gross expectancy: {analysis['gross_expectancy']:.3%}")
        print(f"Net expectancy: {analysis['net_expectancy']:.3%}")
        print(f"Cost impact: {analysis['cost_impact']:.3%} per round trip")
        
        if analysis['cost_impact_pct'] > 0.5:  # Costs > 50% of gross profits
            print(f"🚨 HIGH COST IMPACT: {analysis['cost_impact_pct']:.1%} of gross profits")
            print(f"   Transaction costs are eating significant profits")
        
        return analysis
    
    def _analyze_missed_opportunities(self, round_trips: List[Dict], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missed profit opportunities"""
        
        missed_analysis = {
            'total_missed_profit': 0,
            'avg_missed_per_trade': 0,
            'biggest_missed_opportunity': 0
        }
        
        total_missed = 0
        missed_opportunities = []
        
        for rt in round_trips:
            entry_time = rt['entry']['timestamp']
            exit_time = rt['exit']['timestamp']
            
            # Get price data for extended period after entry
            extended_data = price_data[
                (price_data['timestamp'] >= entry_time) & 
                (price_data['timestamp'] <= exit_time + timedelta(days=3))
            ].copy()
            
            if len(extended_data) < 2:
                continue
            
            entry_price = rt['entry_price']
            actual_profit = rt['pnl_pct']
            
            if rt['direction'] == 'buy':
                # Find maximum profit opportunity
                max_price = extended_data['high'].max()
                max_potential_profit = (max_price - entry_price) / entry_price
            else:
                # For short trades
                min_price = extended_data['low'].min()
                max_potential_profit = (entry_price - min_price) / entry_price
            
            missed_profit = max_potential_profit - actual_profit
            if missed_profit > 0:
                missed_opportunities.append(missed_profit)
                total_missed += missed_profit
        
        if missed_opportunities:
            missed_analysis['total_missed_profit'] = total_missed
            missed_analysis['avg_missed_per_trade'] = np.mean(missed_opportunities)
            missed_analysis['biggest_missed_opportunity'] = max(missed_opportunities)
        
        print(f"\n🎯 MISSED OPPORTUNITIES")
        print("=" * 20)
        print(f"Average missed profit per trade: {missed_analysis['avg_missed_per_trade']:.3%}")
        print(f"Biggest missed opportunity: {missed_analysis['biggest_missed_opportunity']:.3%}")
        
        if missed_analysis['avg_missed_per_trade'] > 0.01:  # Missing > 1% per trade
            print(f"🚨 SIGNIFICANT MISSED PROFITS: {missed_analysis['avg_missed_per_trade']:.3%} per trade")
            print(f"   This explains low returns despite high win rate")
        
        return missed_analysis
    
    def _generate_recommendations(self, round_trips: List[Dict], price_data: pd.DataFrame) -> List[str]:
        """Generate specific recommendations to improve returns"""
        
        recommendations = []
        
        # Analyze the data to generate specific recommendations
        pnls = [rt['pnl_pct'] for rt in round_trips]
        winners = [pnl for pnl in pnls if pnl > 0]
        losers = [pnl for pnl in pnls if pnl < 0]
        
        win_rate = len(winners) / len(round_trips) if round_trips else 0
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = np.mean(losers) if losers else 0
        
        # Core issue: High win rate, low returns
        if win_rate > 0.6 and np.mean(pnls) < 0.005:
            recommendations.append("🎯 CORE ISSUE: Cutting winners short, letting losers run")
            
            if abs(avg_loser) > avg_winner:
                recommendations.append("✂️ IMPLEMENT TIGHTER STOP LOSSES: Current avg loss too large")
                recommendations.append(f"   Target: Limit losses to {avg_winner:.3%} (same as avg winner)")
            
            if avg_winner < 0.01:  # Less than 1% average winner
                recommendations.append("🚀 LET WINNERS RUN LONGER: Use trailing stops instead of fixed exits")
                recommendations.append("   Target: 2-3x current average winner size")
            
            # Reward-to-risk ratio
            if avg_loser != 0:
                rr_ratio = abs(avg_winner / avg_loser)
                if rr_ratio < 2.0:
                    recommendations.append(f"⚖️ IMPROVE REWARD/RISK RATIO: Current {rr_ratio:.2f}, target >2.0")
        
        # Position sizing recommendations
        recommendations.append("📏 IMPLEMENT DYNAMIC POSITION SIZING:")
        recommendations.append("   • Larger positions on high-confidence trades")
        recommendations.append("   • Smaller positions on low-confidence trades")
        recommendations.append("   • Risk 1-2% of capital per trade maximum")
        
        # Exit strategy improvements
        recommendations.append("🚪 IMPROVE EXIT STRATEGY:")
        recommendations.append("   • Use trailing stops to capture larger moves")
        recommendations.append("   • Exit partial positions at targets, let remainder run")
        recommendations.append("   • Use ATR-based stops instead of fixed percentages")
        
        # Entry improvements
        recommendations.append("🎯 REFINE ENTRY TIMING:")
        recommendations.append("   • Wait for better risk/reward setups")
        recommendations.append("   • Use limit orders to improve entry prices")
        recommendations.append("   • Avoid entering during high volatility periods")
        
        return recommendations


def run_entry_exit_analysis():
    """Run comprehensive entry/exit analysis"""
    
    print("🔍 ENTRY/EXIT ANALYSIS")
    print("=" * 30)
    print("Professional analysis of why high win rate = low returns")
    
    try:
        # Import and run a backtest to get trade data
        from multi_timeframe_system_corrected import run_multi_timeframe_optimization
        
        print(f"\n📊 Running backtest to collect trade data...")
        result = run_multi_timeframe_optimization()
        
        if not result or 'trades' not in result:
            print("❌ No trade data available for analysis")
            return None
        
        trades = result['trades']
        
        # Get price data for analysis
        from production_real_data_backtester import RealDataFetcher
        data_fetcher = RealDataFetcher()
        price_data = data_fetcher.fetch_real_data("BTCUSDT", "2024-01-01", "2024-02-01", "15m")
        
        if price_data is None:
            print("❌ No price data available for analysis")
            return None
        
        # Run detailed analysis
        analyzer = TradeAnalyzer()
        analysis = analyzer.analyze_trade_performance(trades, price_data)
        
        if 'error' in analysis:
            print(f"❌ Analysis error: {analysis['error']}")
            return None
        
        # Display recommendations
        print(f"\n💡 SPECIFIC RECOMMENDATIONS")
        print("=" * 30)
        
        for rec in analysis['recommendations']:
            print(f"{rec}")
        
        print(f"\n🎯 IMPLEMENTATION PRIORITY:")
        print("1. Implement trailing stops for exits")
        print("2. Tighten stop losses to match average winners")
        print("3. Use dynamic position sizing based on confidence")
        print("4. Add partial profit taking with remainder running")
        print("5. Improve entry timing with limit orders")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Entry/exit analysis failed: {e}")
        return None


if __name__ == "__main__":
    run_entry_exit_analysis()
