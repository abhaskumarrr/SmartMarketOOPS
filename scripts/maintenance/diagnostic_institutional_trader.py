#!/usr/bin/env python3
"""
DIAGNOSTIC INSTITUTIONAL TRADER
Exposes exactly what's happening with data and confluence calculations
Reveals why only one day generated signals out of 30 days
"""

import ccxt
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosticInstitutionalTrader:
    """Diagnostic version that shows exactly what's happening daily"""
    
    def __init__(self):
        # Initialize exchange for real data
        self.exchange = ccxt.binance()
        
        # Trading parameters
        self.timeframes = {
            'weekly': '1w',
            'daily': '1d', 
            'h4': '4h',
            'h1': '1h',
            'm15': '15m'
        }
        
        # Confluence factors and weights
        self.confluence_weights = {
            'trend_alignment': 20,      # Multi-timeframe trend alignment
            'momentum': 15,             # RSI/MACD momentum
            'volume': 15,               # Volume confirmation
            'support_resistance': 20,   # Key levels
            'volatility': 10,           # Market volatility
            'market_structure': 10,     # Break of structure
            'risk_reward': 10           # Risk/reward ratio
        }
        
        self.min_confluence = 85  # Current threshold
        
    def fetch_diagnostic_data(self, symbol: str = 'BTC/USDT', days: int = 30) -> Dict:
        """Fetch real data and diagnose each day"""
        logger.info(f"🔍 DIAGNOSTIC: Fetching {days} days of real data for {symbol}")
        
        # Get daily data for the period
        daily_data = self.exchange.fetch_ohlcv(symbol, '1d', limit=days + 10)
        df_daily = pd.DataFrame(daily_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_daily['timestamp'] = pd.to_datetime(df_daily['timestamp'], unit='ms')
        df_daily.set_index('timestamp', inplace=True)
        
        # Get hourly data for detailed analysis
        hourly_data = self.exchange.fetch_ohlcv(symbol, '1h', limit=days * 24 + 100)
        df_hourly = pd.DataFrame(hourly_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp'], unit='ms')
        df_hourly.set_index('timestamp', inplace=True)
        
        diagnostic_results = {
            'symbol': symbol,
            'analysis_period': f"{days} days",
            'daily_data_points': len(df_daily),
            'hourly_data_points': len(df_hourly),
            'daily_analysis': [],
            'summary': {}
        }
        
        # Analyze each day
        logger.info("📊 Analyzing each day's market conditions...")
        
        for i in range(len(df_daily) - 5):  # Need 5 days for indicators
            current_day = df_daily.iloc[i]
            date_str = current_day.name.strftime('%Y-%m-%d')
            
            # Calculate indicators for this day
            daily_slice = df_daily.iloc[max(0, i-20):i+1]  # 20 days for indicators
            hourly_slice = df_hourly[df_hourly.index.date == current_day.name.date()]
            
            if len(daily_slice) < 5 or len(hourly_slice) < 4:
                continue
                
            day_analysis = self.analyze_single_day(date_str, current_day, daily_slice, hourly_slice)
            diagnostic_results['daily_analysis'].append(day_analysis)
            
        # Generate summary
        diagnostic_results['summary'] = self.generate_diagnostic_summary(diagnostic_results['daily_analysis'])
        
        return diagnostic_results
    
    def analyze_single_day(self, date_str: str, current_day, daily_slice, hourly_slice) -> Dict:
        """Detailed analysis of a single day's market conditions"""
        
        # Calculate technical indicators
        daily_slice = daily_slice.copy()
        daily_slice['rsi'] = ta.momentum.RSIIndicator(daily_slice['close']).rsi()
        daily_slice['sma_20'] = ta.trend.SMAIndicator(daily_slice['close'], window=20).sma_indicator()
        daily_slice['sma_50'] = ta.trend.SMAIndicator(daily_slice['close'], window=50).sma_indicator()
        daily_slice['bb_upper'] = ta.volatility.BollingerBands(daily_slice['close']).bollinger_hband()
        daily_slice['bb_lower'] = ta.volatility.BollingerBands(daily_slice['close']).bollinger_lband()
        
        # Get current values
        current_close = current_day['close']
        current_volume = current_day['volume']
        price_change = ((current_close - daily_slice['close'].iloc[-2]) / daily_slice['close'].iloc[-2]) * 100
        
        # Calculate confluence factors
        confluence_scores = {}
        
        # 1. Trend Alignment (20 points)
        trend_score = 0
        if len(daily_slice) >= 50:
            sma_20 = daily_slice['sma_20'].iloc[-1]
            sma_50 = daily_slice['sma_50'].iloc[-1]
            
            if current_close > sma_20 > sma_50:  # Bullish alignment
                trend_score = 20
            elif current_close < sma_20 < sma_50:  # Bearish alignment
                trend_score = 20
            elif current_close > sma_20 or current_close > sma_50:  # Partial alignment
                trend_score = 10
        
        confluence_scores['trend_alignment'] = trend_score
        
        # 2. Momentum (15 points)
        momentum_score = 0
        if len(daily_slice) >= 14:
            rsi = daily_slice['rsi'].iloc[-1]
            if 30 <= rsi <= 70:  # Good momentum zone
                momentum_score = 15
            elif 20 <= rsi <= 80:  # Acceptable zone
                momentum_score = 10
            else:  # Extreme zones
                momentum_score = 5
                
        confluence_scores['momentum'] = momentum_score
        
        # 3. Volume (15 points)
        volume_score = 0
        if len(daily_slice) >= 10:
            avg_volume = daily_slice['volume'].iloc[-10:].mean()
            if current_volume > avg_volume * 1.5:  # High volume
                volume_score = 15
            elif current_volume > avg_volume:  # Above average
                volume_score = 10
            else:  # Below average
                volume_score = 5
                
        confluence_scores['volume'] = volume_score
        
        # 4. Support/Resistance (20 points)
        sr_score = 0
        if len(daily_slice) >= 20:
            # Simple S/R based on recent highs/lows
            recent_high = daily_slice['high'].iloc[-20:].max()
            recent_low = daily_slice['low'].iloc[-20:].min()
            price_position = (current_close - recent_low) / (recent_high - recent_low)
            
            if 0.2 <= price_position <= 0.8:  # Middle zone
                sr_score = 20
            elif 0.1 <= price_position <= 0.9:  # Acceptable zone
                sr_score = 15
            else:  # Extreme zones
                sr_score = 10
                
        confluence_scores['support_resistance'] = sr_score
        
        # 5. Volatility (10 points)
        volatility_score = 10  # Default good score
        if len(daily_slice) >= 20:
            bb_upper = daily_slice['bb_upper'].iloc[-1]
            bb_lower = daily_slice['bb_lower'].iloc[-1]
            if bb_lower <= current_close <= bb_upper:  # Normal volatility
                volatility_score = 10
            else:  # High volatility
                volatility_score = 5
                
        confluence_scores['volatility'] = volatility_score
        
        # 6. Market Structure (10 points)
        structure_score = 10  # Default good score for now
        confluence_scores['market_structure'] = structure_score
        
        # 7. Risk/Reward (10 points)
        rr_score = 10  # Default good score for now
        confluence_scores['risk_reward'] = rr_score
        
        # Calculate total confluence
        total_confluence = sum(confluence_scores.values())
        confluence_percentage = total_confluence  # Already out of 100
        
        # Determine if signal would be generated
        signal_generated = confluence_percentage >= self.min_confluence
        
        return {
            'date': date_str,
            'price': round(current_close, 2),
            'price_change_percent': round(price_change, 2),
            'volume': int(current_volume),
            'confluence_scores': confluence_scores,
            'total_confluence': total_confluence,
            'confluence_percentage': round(confluence_percentage, 1),
            'signal_generated': signal_generated,
            'min_confluence_required': self.min_confluence,
            'meets_threshold': confluence_percentage >= self.min_confluence
        }
    
    def generate_diagnostic_summary(self, daily_analysis: List[Dict]) -> Dict:
        """Generate summary of diagnostic findings"""
        if not daily_analysis:
            return {}
            
        total_days = len(daily_analysis)
        signal_days = sum(1 for day in daily_analysis if day['signal_generated'])
        
        # Find best and worst confluence days
        best_day = max(daily_analysis, key=lambda x: x['confluence_percentage'])
        worst_day = min(daily_analysis, key=lambda x: x['confluence_percentage'])
        
        # Calculate averages
        avg_confluence = np.mean([day['confluence_percentage'] for day in daily_analysis])
        
        # Find days that were close to threshold
        threshold = self.min_confluence
        close_days = [day for day in daily_analysis if threshold - 10 <= day['confluence_percentage'] < threshold]
        
        return {
            'total_days_analyzed': total_days,
            'signal_days': signal_days,
            'signal_percentage': round((signal_days / total_days) * 100, 1),
            'average_confluence': round(avg_confluence, 1),
            'current_threshold': threshold,
            'best_confluence_day': {
                'date': best_day['date'],
                'confluence': best_day['confluence_percentage'],
                'price': best_day['price']
            },
            'worst_confluence_day': {
                'date': worst_day['date'],
                'confluence': worst_day['confluence_percentage'],
                'price': worst_day['price']
            },
            'days_close_to_threshold': len(close_days),
            'recommended_threshold': round(avg_confluence + 10, 0),  # More realistic threshold
            'confluence_distribution': {
                '90-100%': sum(1 for day in daily_analysis if day['confluence_percentage'] >= 90),
                '80-89%': sum(1 for day in daily_analysis if 80 <= day['confluence_percentage'] < 90),
                '70-79%': sum(1 for day in daily_analysis if 70 <= day['confluence_percentage'] < 80),
                '60-69%': sum(1 for day in daily_analysis if 60 <= day['confluence_percentage'] < 70),
                'Below 60%': sum(1 for day in daily_analysis if day['confluence_percentage'] < 60),
            }
        }
    
    def run_diagnostic(self, symbol: str = 'BTC/USDT', days: int = 30):
        """Run complete diagnostic analysis"""
        logger.info("🚀 Starting DIAGNOSTIC INSTITUTIONAL TRADER Analysis")
        logger.info("="*60)
        
        try:
            # Fetch and analyze data
            results = self.fetch_diagnostic_data(symbol, days)
            
            # Save detailed results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"diagnostic_results_{symbol.replace('/', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Print summary
            self.print_diagnostic_summary(results)
            
            logger.info(f"📋 Detailed results saved to: {filename}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Diagnostic failed: {e}")
            return None
    
    def print_diagnostic_summary(self, results: Dict):
        """Print diagnostic summary to console"""
        summary = results['summary']
        
        print("\n" + "="*60)
        print("🔍 DIAGNOSTIC INSTITUTIONAL TRADER RESULTS")
        print("="*60)
        
        print(f"\n📊 ANALYSIS OVERVIEW:")
        print(f"Symbol: {results['symbol']}")
        print(f"Period: {results['analysis_period']}")
        print(f"Total Days: {summary['total_days_analyzed']}")
        print(f"Days with Signals: {summary['signal_days']}")
        print(f"Signal Rate: {summary['signal_percentage']}%")
        print(f"Average Confluence: {summary['average_confluence']}%")
        print(f"Current Threshold: {summary['current_threshold']}%")
        
        print(f"\n🎯 CONFLUENCE ANALYSIS:")
        best = summary['best_confluence_day']
        worst = summary['worst_confluence_day']
        print(f"Best Day: {best['date']} - {best['confluence']}% (${best['price']})")
        print(f"Worst Day: {worst['date']} - {worst['confluence']}% (${worst['price']})")
        print(f"Days Close to Threshold: {summary['days_close_to_threshold']}")
        print(f"Recommended Threshold: {summary['recommended_threshold']}%")
        
        print(f"\n📈 CONFLUENCE DISTRIBUTION:")
        dist = summary['confluence_distribution']
        for range_name, count in dist.items():
            percentage = (count / summary['total_days_analyzed']) * 100
            print(f"{range_name}: {count} days ({percentage:.1f}%)")
        
        print(f"\n💡 DIAGNOSTIC INSIGHTS:")
        if summary['signal_percentage'] < 5:
            print("❌ ISSUE: Very few signals generated - threshold may be too high")
        if summary['average_confluence'] < summary['current_threshold']:
            print("❌ ISSUE: Average confluence below threshold - unrealistic expectations")
        if summary['days_close_to_threshold'] > 5:
            print("⚠️  FINDING: Many days close to threshold - small adjustment could help")
        
        print("\n" + "="*60)

def main():
    """Run the diagnostic analysis"""
    diagnostic = DiagnosticInstitutionalTrader()
    results = diagnostic.run_diagnostic('BTC/USDT', 30)
    
    if results:
        print("\n🔍 Diagnostic Complete! Check the generated JSON file for detailed daily analysis.")
        
        # Print some sample days for immediate review
        daily_analysis = results['daily_analysis']
        if len(daily_analysis) >= 10:
            print("\n📋 SAMPLE DAILY ANALYSIS (Last 10 Days):")
            for day in daily_analysis[-10:]:
                status = "✅ SIGNAL" if day['signal_generated'] else "❌ NO SIGNAL"
                print(f"{day['date']}: ${day['price']:,.2f} ({day['price_change_percent']:+.1f}%) - "
                      f"Confluence: {day['confluence_percentage']}% - {status}")

if __name__ == "__main__":
    main()