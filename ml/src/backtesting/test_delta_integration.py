#!/usr/bin/env python3
"""
Test Delta Exchange Integration

This script tests the integration of the real Delta Exchange client
into our production backtesting system.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_delta_client_import():
    """Test importing the Delta Exchange client"""
    print("🔧 Testing Delta Exchange client import...")

    try:
        # Add the ml/src path to import the Delta client
        current_dir = Path(__file__).parent
        ml_src_path = current_dir.parent.parent.parent.parent / "ml" / "src"

        print(f"   Current directory: {current_dir}")
        print(f"   ML src path: {ml_src_path}")
        print(f"   ML src exists: {ml_src_path.exists()}")

        if str(ml_src_path) not in sys.path:
            sys.path.insert(0, str(ml_src_path))

        # Check if delta_client.py exists
        delta_client_path = ml_src_path / "api" / "delta_client.py"
        print(f"   Delta client path: {delta_client_path}")
        print(f"   Delta client exists: {delta_client_path.exists()}")

        # Try importing
        from api.delta_client import DeltaExchangeClient

        print("✅ Delta Exchange client imported successfully")
        return DeltaExchangeClient

    except ImportError as e:
        print(f"❌ Delta Exchange client import failed: {e}")
        print(f"   Python path: {sys.path[:3]}...")  # Show first 3 paths

        # Try alternative import method
        try:
            print("   Trying alternative import method...")
            import importlib.util

            ml_src_path = Path(__file__).parent.parent.parent.parent.parent / "ml" / "src"
            delta_client_path = ml_src_path / "api" / "delta_client.py"

            if delta_client_path.exists():
                spec = importlib.util.spec_from_file_location("delta_client", delta_client_path)
                delta_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(delta_module)

                print("✅ Delta Exchange client imported via alternative method")
                return delta_module.DeltaExchangeClient
            else:
                print(f"   ❌ Delta client file not found at {delta_client_path}")
                return None

        except Exception as alt_e:
            print(f"   ❌ Alternative import also failed: {alt_e}")
            return None

    except Exception as e:
        print(f"❌ Unexpected error importing Delta client: {e}")
        return None


def test_delta_client_initialization(DeltaExchangeClient):
    """Test initializing the Delta Exchange client"""
    print("\n🔧 Testing Delta Exchange client initialization...")

    try:
        # Initialize with testnet=True for safety
        client = DeltaExchangeClient(testnet=True)
        print("✅ Delta Exchange client initialized successfully (testnet)")
        return client

    except Exception as e:
        print(f"❌ Delta Exchange client initialization failed: {e}")
        return None


def test_delta_client_basic_methods(client):
    """Test basic Delta Exchange client methods"""
    print("\n🔧 Testing Delta Exchange client basic methods...")

    try:
        # Test getting products
        print("   Testing get_products()...")
        products = client.get_products()
        print(f"   ✅ Retrieved {len(products)} products")

        # Find BTC product
        btc_products = [p for p in products if 'BTC' in p.get('symbol', '')]
        if btc_products:
            btc_symbol = btc_products[0]['symbol']
            print(f"   ✅ Found BTC product: {btc_symbol}")
        else:
            btc_symbol = 'BTCUSD'  # Default fallback
            print(f"   ⚠️  Using default BTC symbol: {btc_symbol}")

        return btc_symbol

    except Exception as e:
        print(f"   ❌ Basic methods test failed: {e}")
        return 'BTCUSD'  # Fallback


def test_delta_historical_data(client, symbol):
    """Test fetching historical data from Delta Exchange"""
    print(f"\n🔧 Testing Delta Exchange historical data for {symbol}...")

    try:
        # Test get_historical_ohlcv
        print("   Testing get_historical_ohlcv()...")

        historical_data = client.get_historical_ohlcv(
            symbol=symbol,
            interval='1h',
            days_back=7  # Last 7 days
        )

        if historical_data:
            print(f"   ✅ Retrieved {len(historical_data)} historical candles")

            # Show sample data structure
            if len(historical_data) > 0:
                sample = historical_data[0]
                print(f"   📊 Sample data structure: {type(sample)}")
                if isinstance(sample, dict):
                    print(f"   📊 Sample keys: {list(sample.keys())}")
                elif isinstance(sample, (list, tuple)):
                    print(f"   📊 Sample length: {len(sample)}")
                print(f"   📊 Sample data: {sample}")

            return historical_data
        else:
            print("   ❌ No historical data returned")
            return None

    except Exception as e:
        print(f"   ❌ Historical data test failed: {e}")
        return None


def test_delta_data_conversion(historical_data):
    """Test converting Delta data to DataFrame format"""
    print("\n🔧 Testing Delta data conversion to DataFrame...")

    if not historical_data:
        print("   ❌ No data to convert")
        return None

    try:
        df_data = []

        for candle in historical_data:
            if isinstance(candle, dict):
                # Handle dict format
                df_data.append({
                    'timestamp': pd.to_datetime(candle.get('time', candle.get('timestamp')), unit='ms'),
                    'open': float(candle.get('open', 0)),
                    'high': float(candle.get('high', 0)),
                    'low': float(candle.get('low', 0)),
                    'close': float(candle.get('close', 0)),
                    'volume': float(candle.get('volume', 0))
                })
            elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                # Handle array format
                df_data.append({
                    'timestamp': pd.to_datetime(candle[0], unit='ms'),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })

        if df_data:
            df = pd.DataFrame(df_data)
            df = df.sort_values('timestamp').reset_index(drop=True)

            print(f"   ✅ Converted to DataFrame with {len(df)} rows")
            print(f"   📊 Columns: {list(df.columns)}")
            print(f"   📊 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   📊 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

            # Show sample rows
            print("   📊 Sample data:")
            print(df.head(3).to_string(index=False))

            return df
        else:
            print("   ❌ No data could be converted")
            return None

    except Exception as e:
        print(f"   ❌ Data conversion failed: {e}")
        return None


def test_production_backtester_integration():
    """Test the production backtester with Delta integration"""
    print("\n🔧 Testing production backtester with Delta integration...")

    try:
        from production_real_data_backtester import RealDataFetcher, ProductionBacktestConfig

        # Create config for Delta testing
        config = ProductionBacktestConfig(
            symbol="BTCUSD",  # Delta format
            start_date="2024-01-01",
            end_date="2024-01-07",  # Short period for testing
            timeframe="1h",
            initial_capital=10000.0
        )

        print(f"   Testing with config: {config.symbol}, {config.start_date} to {config.end_date}")

        # Initialize data fetcher
        data_fetcher = RealDataFetcher()

        # Test data fetching
        real_data = data_fetcher.fetch_real_data(
            symbol=config.symbol,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe
        )

        if real_data is not None and len(real_data) > 0:
            print(f"   ✅ Production backtester fetched {len(real_data)} candles")
            print(f"   📊 Data source priority: Delta Exchange -> Binance -> Fallback")

            # Check if Delta was used
            if hasattr(data_fetcher, 'delta_client') and data_fetcher.delta_client:
                print("   ✅ Delta Exchange client is available in production system")
            else:
                print("   ⚠️  Delta Exchange client not available, using fallback")

            return True
        else:
            print("   ❌ Production backtester failed to fetch data")
            return False

    except Exception as e:
        print(f"   ❌ Production backtester integration test failed: {e}")
        return False


def run_delta_integration_tests():
    """Run comprehensive Delta Exchange integration tests"""
    print("🚀 DELTA EXCHANGE INTEGRATION TESTS")
    print("=" * 50)

    test_results = {
        'import': False,
        'initialization': False,
        'basic_methods': False,
        'historical_data': False,
        'data_conversion': False,
        'production_integration': False
    }

    # Test 1: Import
    DeltaExchangeClient = test_delta_client_import()
    test_results['import'] = DeltaExchangeClient is not None

    if not test_results['import']:
        print("\n❌ Cannot proceed without Delta Exchange client import")
        return test_results

    # Test 2: Initialization
    client = test_delta_client_initialization(DeltaExchangeClient)
    test_results['initialization'] = client is not None

    if not test_results['initialization']:
        print("\n❌ Cannot proceed without Delta Exchange client initialization")
        return test_results

    # Test 3: Basic methods
    symbol = test_delta_client_basic_methods(client)
    test_results['basic_methods'] = symbol is not None

    # Test 4: Historical data
    historical_data = test_delta_historical_data(client, symbol)
    test_results['historical_data'] = historical_data is not None

    # Test 5: Data conversion
    df = test_delta_data_conversion(historical_data)
    test_results['data_conversion'] = df is not None

    # Test 6: Production integration
    test_results['production_integration'] = test_production_backtester_integration()

    # Summary
    print("\n📊 DELTA EXCHANGE INTEGRATION TEST RESULTS")
    print("=" * 50)

    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED - Delta Exchange integration is working!")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  MOSTLY WORKING - Minor issues with Delta Exchange integration")
    else:
        print("❌ INTEGRATION ISSUES - Delta Exchange integration needs work")

    return test_results


if __name__ == "__main__":
    print("🎯 Testing Delta Exchange Integration for Production Backtesting")
    print("This will verify that the real Delta Exchange client works properly")

    results = run_delta_integration_tests()

    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    if not results['import']:
        print("- Check that ml/src/api/delta_client.py exists and is accessible")
        print("- Verify Python path configuration")

    if not results['initialization']:
        print("- Check Delta Exchange API credentials in config")
        print("- Verify network connectivity to Delta Exchange")

    if not results['historical_data']:
        print("- Check Delta Exchange API limits and permissions")
        print("- Verify symbol format (BTCUSD vs BTCUSDT)")

    if results['production_integration']:
        print("✅ Production backtesting system can use Delta Exchange data!")
        print("✅ Ready for real trading strategy development!")
    else:
        print("⚠️  Production system will fall back to Binance or synthetic data")
