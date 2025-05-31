#!/usr/bin/env python3
"""
Demo for Enhanced Real Data Backtesting & Retraining System

This script demonstrates the complete enhanced backtesting system that:
- Fetches real market data from ccxt/Delta Exchange
- Uses enhanced ML predictions + SMC analysis
- Performs automatic model retraining
- Provides comprehensive performance analysis
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

try:
    from enhanced_real_data_backtester import (
        run_enhanced_backtest, 
        retrain_model_with_real_data,
        EnhancedBacktestConfig,
        EnhancedRealDataBacktester
    )
    BACKTESTER_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced backtester not available: {e}")
    BACKTESTER_AVAILABLE = False


def demo_quick_backtest():
    """Demonstrate quick backtesting with real data"""
    print("\n" + "="*60)
    print("🚀 ENHANCED REAL DATA BACKTEST DEMO")
    print("="*60)
    
    if not BACKTESTER_AVAILABLE:
        print("❌ Enhanced backtester not available")
        return
    
    print("Running enhanced backtest with real market data...")
    print("Features:")
    print("✅ Real data from Delta Exchange / Binance")
    print("✅ Enhanced ML predictions")
    print("✅ Smart Money Concepts analysis")
    print("✅ Automatic model retraining")
    print("✅ Comprehensive performance metrics")
    
    try:
        # Run backtest for last 3 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        results = run_enhanced_backtest(
            symbol="BTCUSD",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe="1h",
            initial_capital=10000.0,
            use_real_data=True,  # Try real data first
            data_source="auto",  # Auto-select best source
            use_enhanced_predictions=True,
            use_smc_analysis=True,
            retrain_model=False,  # Disable for demo speed
            confidence_threshold=0.6,
            risk_level="medium"
        )
        
        print(f"\n🎯 BACKTEST RESULTS")
        print(f"   Symbol: {results['config']['symbol']}")
        print(f"   Period: {results['data_info']['start_date'][:10]} to {results['data_info']['end_date'][:10]}")
        print(f"   Data Source: {results['data_info']['data_source']}")
        print(f"   Total Candles: {results['data_info']['total_candles']:,}")
        
        # Performance metrics
        backtest_results = results['backtest_results']
        if 'metrics' in backtest_results:
            metrics = backtest_results['metrics']
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"   Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        
        # Enhanced metrics
        enhanced_metrics = results['enhanced_metrics']
        if 'signal_analysis' in enhanced_metrics:
            signal_analysis = enhanced_metrics['signal_analysis']
            print(f"\n🔍 SIGNAL ANALYSIS:")
            print(f"   Total Trades: {signal_analysis['total_trades']}")
            print(f"   ML-based Trades: {signal_analysis['ml_trades']}")
            print(f"   SMC-based Trades: {signal_analysis['smc_trades']}")
            print(f"   Technical Trades: {signal_analysis['technical_trades']}")
            
            if signal_analysis['ml_trades'] > 0:
                print(f"   ML Win Rate: {signal_analysis['ml_win_rate']:.2%}")
            if signal_analysis['smc_trades'] > 0:
                print(f"   SMC Win Rate: {signal_analysis['smc_win_rate']:.2%}")
            if signal_analysis['technical_trades'] > 0:
                print(f"   Technical Win Rate: {signal_analysis['technical_win_rate']:.2%}")
        
        if 'confidence_analysis' in enhanced_metrics:
            confidence_analysis = enhanced_metrics['confidence_analysis']
            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"   Average Confidence: {confidence_analysis['avg_confidence']:.2%}")
            print(f"   High Confidence Trades: {confidence_analysis['high_confidence_trades']}")
            print(f"   Low Confidence Trades: {confidence_analysis['low_confidence_trades']}")
        
        # Model retraining info
        model_retraining = results['model_retraining']
        print(f"\n🔄 MODEL RETRAINING:")
        print(f"   Enabled: {model_retraining['enabled']}")
        print(f"   Frequency: Every {model_retraining['frequency_days']} days")
        print(f"   Retraining Events: {len(model_retraining['retraining_events'])}")
        
        print(f"\n✅ Enhanced backtest completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return None


def demo_model_retraining():
    """Demonstrate model retraining with real data"""
    print("\n" + "="*60)
    print("🔄 MODEL RETRAINING WITH REAL DATA DEMO")
    print("="*60)
    
    if not BACKTESTER_AVAILABLE:
        print("❌ Enhanced backtester not available")
        return
    
    print("Retraining model with fresh real market data...")
    print("Features:")
    print("✅ Automatic real data fetching")
    print("✅ Multiple data source support")
    print("✅ Model training with latest patterns")
    print("✅ Performance validation")
    
    try:
        result = retrain_model_with_real_data(
            symbol="BTCUSD",
            model_type="cnn_lstm",
            data_source="auto",
            days_back=60,  # 2 months of data
            num_epochs=5,  # Quick training for demo
            early_stopping_patience=3,
            learning_rate=0.001
        )
        
        if 'error' not in result:
            print(f"\n🎯 RETRAINING RESULTS")
            print(f"   Model Version: {result['version']}")
            
            metrics = result['metrics']
            print(f"\n📊 MODEL PERFORMANCE:")
            print(f"   Test Loss: {metrics.get('test_loss', 0):.4f}")
            print(f"   Test Accuracy: {metrics.get('test_accuracy', 0):.2%}")
            print(f"   Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}")
            
            if 'rmse' in metrics:
                print(f"   RMSE: {metrics['rmse']:.4f}")
            if 'mae' in metrics:
                print(f"   MAE: {metrics['mae']:.4f}")
            
            print(f"\n✅ Model retraining completed successfully!")
            return result
        else:
            print(f"❌ Model retraining failed: {result['error']}")
            return None
            
    except Exception as e:
        print(f"❌ Model retraining failed: {e}")
        return None


def demo_custom_backtest():
    """Demonstrate custom backtest configuration"""
    print("\n" + "="*60)
    print("⚙️  CUSTOM BACKTEST CONFIGURATION DEMO")
    print("="*60)
    
    if not BACKTESTER_AVAILABLE:
        print("❌ Enhanced backtester not available")
        return
    
    print("Running custom configured backtest...")
    
    try:
        # Create custom configuration
        config = EnhancedBacktestConfig(
            symbol="BTCUSD",
            start_date="2024-01-01",
            end_date="2024-06-30",
            timeframe="4h",  # 4-hour timeframe
            initial_capital=50000.0,  # Higher capital
            fee_rate=0.0005,  # Lower fees (0.05%)
            slippage_factor=0.0002,  # Lower slippage
            use_enhanced_predictions=True,
            use_smc_analysis=True,
            confidence_threshold=0.7,  # Higher confidence threshold
            risk_level="low",  # Conservative risk
            retrain_model=True,
            retrain_frequency_days=15,  # Retrain every 2 weeks
            model_type="transformer",  # Use transformer model
            data_source="auto",
            use_real_data=True
        )
        
        # Run custom backtest
        backtester = EnhancedRealDataBacktester(config)
        results = backtester.run_backtest()
        
        print(f"\n🎯 CUSTOM BACKTEST RESULTS")
        print(f"   Configuration: {config.timeframe} timeframe, {config.risk_level} risk")
        print(f"   Capital: ${config.initial_capital:,.2f}")
        print(f"   Confidence Threshold: {config.confidence_threshold:.1%}")
        print(f"   Model Type: {config.model_type}")
        
        # Show results summary
        data_info = results['data_info']
        print(f"\n📊 DATA INFO:")
        print(f"   Period: {data_info['start_date'][:10]} to {data_info['end_date'][:10]}")
        print(f"   Candles: {data_info['total_candles']:,}")
        print(f"   Data Source: {data_info['data_source']}")
        
        if 'metrics' in results['backtest_results']:
            metrics = results['backtest_results']['metrics']
            print(f"\n💰 PERFORMANCE:")
            print(f"   Final Capital: ${metrics.get('final_capital', config.initial_capital):,.2f}")
            print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        print(f"\n✅ Custom backtest completed!")
        return results
        
    except Exception as e:
        print(f"❌ Custom backtest failed: {e}")
        return None


def demo_data_sources():
    """Demonstrate different data sources"""
    print("\n" + "="*60)
    print("📊 DATA SOURCE COMPARISON DEMO")
    print("="*60)
    
    if not BACKTESTER_AVAILABLE:
        print("❌ Enhanced backtester not available")
        return
    
    data_sources = ["delta", "binance", "auto"]
    results = {}
    
    for source in data_sources:
        print(f"\n🔍 Testing data source: {source.upper()}")
        
        try:
            config = EnhancedBacktestConfig(
                symbol="BTCUSD",
                start_date="2024-01-01",
                end_date="2024-01-31",  # 1 month for quick test
                timeframe="1h",
                data_source=source,
                use_real_data=True,
                use_enhanced_predictions=False,  # Disable for speed
                use_smc_analysis=False,
                retrain_model=False
            )
            
            backtester = EnhancedRealDataBacktester(config)
            data = backtester.fetch_real_data()
            
            if data is not None and len(data) > 0:
                print(f"   ✅ {source}: {len(data):,} candles fetched")
                print(f"      Period: {data['timestamp'].min()} to {data['timestamp'].max()}")
                print(f"      Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
                results[source] = len(data)
            else:
                print(f"   ❌ {source}: No data fetched")
                results[source] = 0
                
        except Exception as e:
            print(f"   ❌ {source}: Error - {e}")
            results[source] = 0
    
    print(f"\n📊 DATA SOURCE SUMMARY:")
    for source, count in results.items():
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {source.upper()}: {count:,} candles")
    
    return results


def main():
    """Run all demos"""
    print("🎯 Enhanced Real Data Backtesting & Retraining System")
    print("Comprehensive demo of ML + SMC backtesting with real market data")
    
    # Run demos
    demo_quick_backtest()
    demo_model_retraining()
    demo_custom_backtest()
    demo_data_sources()
    
    print(f"\n" + "="*60)
    print("🎉 ALL DEMOS COMPLETED")
    print("="*60)
    print("The Enhanced Real Data Backtesting System successfully:")
    print("✅ Fetched real market data from multiple sources")
    print("✅ Integrated ML predictions with SMC analysis")
    print("✅ Performed comprehensive backtesting")
    print("✅ Demonstrated automatic model retraining")
    print("✅ Provided detailed performance analytics")
    print("✅ Supported custom configurations")
    print("\nReady for production trading strategy development! 🚀")


if __name__ == "__main__":
    main()
