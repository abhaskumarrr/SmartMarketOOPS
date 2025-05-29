"""
Test script for the data loader module

This script tests the data loader functionality by fetching
real data from the Delta Exchange API and processing it.
"""

import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import CryptoDataLoader, load_data
from src.api.delta_client import DeltaExchangeClient

def test_delta_client():
    """Test basic Delta Exchange API client functionality"""
    print("Testing Delta Exchange API client...")
    
    client = DeltaExchangeClient()
    
    # Test getting products
    products = client.get_products()
    print(f"Retrieved {len(products)} products from Delta Exchange")
    
    # Test getting OHLCV data for BTC
    btc_data = client.get_historical_ohlcv(symbol="BTCUSD", interval="1h", days_back=7)
    print(f"Retrieved {len(btc_data)} BTC/USD hourly candles")
    
    # Print sample data
    if btc_data:
        print("\nSample BTC/USD candle:")
        print(btc_data[0])
    
    return len(products) > 0 and len(btc_data) > 0

def test_data_loader():
    """Test basic data loader functionality"""
    print("\nTesting data loader...")
    
    loader = CryptoDataLoader()
    
    # Test getting data with caching
    btc_data = loader.get_data(symbol="BTCUSD", interval="1h", days_back=7, use_cache=True)
    print(f"Loaded {len(btc_data)} rows of BTC/USD data")
    
    # Print sample data
    if not btc_data.empty:
        print("\nSample BTC/USD data:")
        print(btc_data.head(3))
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in btc_data.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        
        print("\nData columns:", btc_data.columns.tolist())
        print("Data types:")
        print(btc_data.dtypes)
    
    return not btc_data.empty

def test_technical_indicators():
    """Test adding technical indicators to OHLCV data"""
    print("\nTesting technical indicators...")
    
    loader = CryptoDataLoader()
    btc_data = loader.get_data(symbol="BTCUSD", interval="1h", days_back=30, use_cache=True)
    
    # Add technical indicators
    data_with_indicators = loader.add_technical_indicators(btc_data)
    
    # Print list of added indicators
    original_columns = set(btc_data.columns)
    new_columns = set(data_with_indicators.columns) - original_columns
    
    print(f"Added {len(new_columns)} technical indicators:")
    for col in sorted(new_columns):
        print(f"  - {col}")
    
    # Plot some indicators for visual verification
    if not data_with_indicators.empty:
        plt.figure(figsize=(12, 8))
        
        # Price and Moving Averages
        plt.subplot(2, 1, 1)
        plt.plot(data_with_indicators.index[-100:], data_with_indicators['close'][-100:], label='Close')
        plt.plot(data_with_indicators.index[-100:], data_with_indicators['ma7'][-100:], label='MA7')
        plt.plot(data_with_indicators.index[-100:], data_with_indicators['ma30'][-100:], label='MA30')
        plt.title('Price and Moving Averages (Last 100 hours)')
        plt.legend()
        
        # RSI
        plt.subplot(2, 1, 2)
        plt.plot(data_with_indicators.index[-100:], data_with_indicators['rsi'][-100:])
        plt.axhline(y=70, color='r', linestyle='-')
        plt.axhline(y=30, color='g', linestyle='-')
        plt.title('RSI (Last 100 hours)')
        
        # Save plot to file
        plt.tight_layout()
        os.makedirs('../../logs', exist_ok=True)
        plt.savefig('../../logs/technical_indicators.png')
        print("Plot saved to logs/technical_indicators.png")
    
    return len(new_columns) > 0

def test_preprocessing():
    """Test data preprocessing pipeline"""
    print("\nTesting data preprocessing...")
    
    # Use convenience function
    processed_data = load_data(
        symbol="BTCUSD", 
        interval="1h",
        days_back=30,
        use_cache=True,
        preprocess=True,
        add_features=True,
        normalize=True,
        target_column='close',
        sequence_length=24,
        train_split=0.8
    )
    
    # Check processed data structure
    expected_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'feature_columns', 'target_column', 'scaler_params']
    missing_keys = [key for key in expected_keys if key not in processed_data]
    
    if missing_keys:
        print(f"Missing keys in processed data: {missing_keys}")
        return False
    
    # Print shapes and statistics
    print("\nPreprocessed data statistics:")
    print(f"  X_train shape: {processed_data['X_train'].shape}")
    print(f"  y_train shape: {processed_data['y_train'].shape}")
    print(f"  X_val shape: {processed_data['X_val'].shape}")
    print(f"  y_val shape: {processed_data['y_val'].shape}")
    print(f"  Number of features: {len(processed_data['feature_columns'])}")
    print(f"  Target column: {processed_data['target_column']}")
    
    # Check if data is normalized
    for dataset, name in [(processed_data['X_train'], 'X_train'), (processed_data['X_val'], 'X_val')]:
        min_val = dataset.min()
        max_val = dataset.max()
        print(f"  {name} value range: [{min_val:.4f}, {max_val:.4f}]")
    
    return all(key in processed_data for key in expected_keys)

def test_multiple_symbols():
    """Test loading data for multiple symbols"""
    print("\nTesting multiple symbols data loading...")
    
    loader = CryptoDataLoader()
    symbols = ["BTCUSD", "ETHUSD"]  # Add more symbols as needed
    
    # Load data for multiple symbols
    data_dict = loader.load_multiple_symbols(
        symbols=symbols,
        interval="1h",
        days_back=7,
        use_cache=True
    )
    
    # Check if we loaded data for all symbols
    print(f"Loaded data for {len(data_dict)} out of {len(symbols)} symbols:")
    for symbol, df in data_dict.items():
        print(f"  - {symbol}: {len(df)} rows")
    
    return len(data_dict) == len(symbols)

def main():
    """Run all tests"""
    print("=== Data Ingestion Module Tests ===\n")
    
    # Run tests
    client_ok = test_delta_client()
    loader_ok = test_data_loader()
    indicators_ok = test_technical_indicators()
    preprocessing_ok = test_preprocessing()
    multi_symbol_ok = test_multiple_symbols()
    
    # Summarize results
    print("\n=== Test Results ===")
    print(f"Delta API Client: {'✅ PASS' if client_ok else '❌ FAIL'}")
    print(f"Data Loader: {'✅ PASS' if loader_ok else '❌ FAIL'}")
    print(f"Technical Indicators: {'✅ PASS' if indicators_ok else '❌ FAIL'}")
    print(f"Data Preprocessing: {'✅ PASS' if preprocessing_ok else '❌ FAIL'}")
    print(f"Multiple Symbols: {'✅ PASS' if multi_symbol_ok else '❌ FAIL'}")
    
    all_tests_passed = all([client_ok, loader_ok, indicators_ok, preprocessing_ok, multi_symbol_ok])
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 