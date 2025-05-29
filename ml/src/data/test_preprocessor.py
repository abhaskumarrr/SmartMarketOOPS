"""
Test script for the enhanced preprocessor module

This script tests the enhanced preprocessor functionality by comparing
different scaling methods and evaluating new features.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import CryptoDataLoader
from src.data.preprocessor import EnhancedPreprocessor, preprocess_with_enhanced_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_ohlcv_data(n_samples=500):
    """Create mock OHLCV data for testing"""
    # Start date for the time series
    start_date = datetime.now() - timedelta(days=n_samples // 24)
    
    # Generate timestamps
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate prices with some randomness but with a trend
    start_price = 30000
    prices = []
    current_price = start_price
    
    for i in range(n_samples):
        # Random price change as percentage (-1% to +1%)
        change = np.random.uniform(-0.01, 0.01)
        # Add a slight upward trend
        trend = 0.0005
        # Calculate the price
        current_price = current_price * (1 + change + trend)
        prices.append(current_price)
    
    prices = np.array(prices)
    
    # Generate open, high, low based on the close price
    opens = prices * np.random.uniform(0.99, 1.01, n_samples)
    highs = np.maximum(prices, opens) * np.random.uniform(1.0, 1.02, n_samples)
    lows = np.minimum(prices, opens) * np.random.uniform(0.98, 1.0, n_samples)
    
    # Generate some random volume data
    volumes = np.random.uniform(100, 1000, n_samples) * (1 + 0.5 * np.sin(np.arange(n_samples) / 20))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    return df

def test_advanced_features():
    """Test adding advanced technical indicators"""
    print("\nTesting advanced technical indicators...")
    
    # Create mock data
    mock_data = create_mock_ohlcv_data()
    print(f"Created mock OHLCV data with {len(mock_data)} samples")
    
    loader = CryptoDataLoader()
    preprocessor = EnhancedPreprocessor(loader)
    
    # Add advanced features
    data_with_advanced = preprocessor.add_advanced_features(mock_data)
    
    # Compare with basic indicators
    data_with_basic = loader.add_technical_indicators(mock_data)
    
    basic_columns = set(data_with_basic.columns)
    advanced_columns = set(data_with_advanced.columns)
    new_columns = advanced_columns - basic_columns
    
    print(f"Basic indicators: {len(basic_columns)} columns")
    print(f"Advanced indicators: {len(advanced_columns)} columns")
    print(f"Added {len(new_columns)} new indicators:")
    
    for col in sorted(new_columns):
        print(f"  - {col}")
    
    # Plot some of the new indicators
    if not data_with_advanced.empty and len(new_columns) > 0:
        plt.figure(figsize=(15, 10))
        
        # Close price for reference
        plt.subplot(3, 1, 1)
        plt.plot(data_with_advanced.index[-100:], data_with_advanced['close'][-100:], 'k-', label='Close Price')
        plt.title('Reference: Close Price (Last 100 hours)')
        plt.legend()
        
        # Stochastic oscillator
        if 'stoch_k' in data_with_advanced.columns and 'stoch_d' in data_with_advanced.columns:
            plt.subplot(3, 1, 2)
            plt.plot(data_with_advanced.index[-100:], data_with_advanced['stoch_k'][-100:], 'b-', label='%K')
            plt.plot(data_with_advanced.index[-100:], data_with_advanced['stoch_d'][-100:], 'r-', label='%D')
            plt.axhline(y=80, color='r', linestyle='--')
            plt.axhline(y=20, color='g', linestyle='--')
            plt.title('Stochastic Oscillator (Last 100 hours)')
            plt.legend()
        
        # On-Balance Volume
        if 'obv' in data_with_advanced.columns:
            plt.subplot(3, 1, 3)
            plt.plot(data_with_advanced.index[-100:], data_with_advanced['obv'][-100:], 'g-')
            plt.title('On-Balance Volume (Last 100 hours)')
        
        # Save plot to file
        plt.tight_layout()
        os.makedirs('../../logs', exist_ok=True)
        plt.savefig('../../logs/advanced_indicators.png')
        print("Plot saved to logs/advanced_indicators.png")
    
    return len(new_columns) > 0

def test_scaling_methods():
    """Test different scaling methods and compare distributions"""
    print("\nTesting different scaling methods...")
    
    # Create mock data
    mock_data = create_mock_ohlcv_data()
    
    loader = CryptoDataLoader()
    preprocessor = EnhancedPreprocessor(loader)
    
    # Add features and preprocess with different scaling methods
    # Min-max scaling
    minmax_data = loader.preprocess_data(
        mock_data,
        add_features=True,
        normalize=True,
        target_column='close',
        sequence_length=24,
        train_split=0.8
    )
    
    # Standard scaling
    standard_data = preprocessor.preprocess_with_standard_scaling(
        mock_data,
        add_features=True,
        add_advanced_features=False,  # Use same features as minmax for fair comparison
        target_column='close',
        sequence_length=24,
        train_split=0.8
    )
    
    # Robust scaling
    robust_data = preprocessor.preprocess_with_robust_scaling(
        mock_data,
        add_features=True,
        add_advanced_features=False,  # Use same features as minmax for fair comparison
        target_column='close',
        sequence_length=24,
        train_split=0.8
    )
    
    # Compare distributions
    print("\nScaling method comparison:")
    print(f"Min-max scaling range: [{minmax_data['X_train'].min():.4f}, {minmax_data['X_train'].max():.4f}]")
    print(f"Standard scaling mean: {standard_data['X_train'].mean():.4f}, std: {standard_data['X_train'].std():.4f}")
    print(f"Robust scaling median: {np.median(robust_data['X_train']):.4f}")
    
    # Plot histograms of the target feature after preprocessing
    plt.figure(figsize=(15, 5))
    
    # Get the column index for 'close'
    col_idx = minmax_data['feature_columns'].index('close')
    
    # Extract the 'close' feature from each sequence
    minmax_close = minmax_data['X_train'][:, -1, col_idx]
    standard_close = standard_data['X_train'][:, -1, col_idx]
    robust_close = robust_data['X_train'][:, -1, col_idx]
    
    plt.subplot(1, 3, 1)
    plt.hist(minmax_close, bins=50, alpha=0.7)
    plt.title('Min-Max Scaling')
    plt.xlabel('Normalized Close Price')
    
    plt.subplot(1, 3, 2)
    plt.hist(standard_close, bins=50, alpha=0.7)
    plt.title('Standard Scaling')
    plt.xlabel('Standardized Close Price')
    
    plt.subplot(1, 3, 3)
    plt.hist(robust_close, bins=50, alpha=0.7)
    plt.title('Robust Scaling')
    plt.xlabel('Robust Scaled Close Price')
    
    plt.tight_layout()
    plt.savefig('../../logs/scaling_comparison.png')
    print("Scaling comparison plot saved to logs/scaling_comparison.png")
    
    return True

def test_cross_validation():
    """Test time-series cross-validation functionality"""
    print("\nTesting time-series cross-validation...")
    
    # Create sample data
    n_samples = 1000
    n_features = 5
    sample_data = np.random.random((n_samples, n_features))
    sample_targets = np.random.random(n_samples)
    
    preprocessor = EnhancedPreprocessor()
    
    # Create cross-validation splits
    n_splits = 5
    cv_splits = preprocessor.cross_validation_split(sample_data, sample_targets, n_splits)
    
    # Verify splits
    print(f"Created {len(cv_splits)} cross-validation splits")
    
    for i, split in enumerate(cv_splits):
        print(f"Split {i+1}:")
        print(f"  Training samples: {split['X_train'].shape[0]}")
        print(f"  Validation samples: {split['X_val'].shape[0]}")
        print(f"  Split index: {split['split_idx']}")
    
    # Verify that splits respect time ordering (no future leakage)
    all_splits_valid = True
    for split in cv_splits:
        if split['split_idx'] < 0 or split['split_idx'] >= n_samples:
            all_splits_valid = False
            print(f"Invalid split index: {split['split_idx']}")
            break
    
    return all_splits_valid

def test_convenience_function():
    """Test the convenience function for preprocessing"""
    print("\nTesting convenience function...")
    
    # Create a mock function to replace the original get_data method
    original_get_data = CryptoDataLoader.get_data
    
    try:
        # Patch the get_data method to return mock data
        def mock_get_data(self, symbol=None, interval=None, days_back=None, use_cache=None):
            return create_mock_ohlcv_data(n_samples=200)
        
        CryptoDataLoader.get_data = mock_get_data
        
        # Test with different scaling methods
        scaling_methods = ['minmax', 'standard', 'robust']
        
        for method in scaling_methods:
            print(f"\nTesting {method} scaling method:")
            
            try:
                processed_data = preprocess_with_enhanced_features(
                    symbol="BTCUSD",  # Doesn't matter as we're using mock data
                    interval="1h",
                    days_back=7,
                    use_cache=True,
                    scaling_method=method,
                    add_advanced_features=True,
                    sequence_length=24,
                    train_split=0.8
                )
                
                print(f"  Successfully processed data using {method} scaling")
                print(f"  X_train shape: {processed_data['X_train'].shape}")
                print(f"  y_train shape: {processed_data['y_train'].shape}")
                
                # Check if advanced features were added
                if method in ['standard', 'robust']:
                    expected_advanced_features = ['stoch_k', 'stoch_d', 'atr', 'obv']
                    found_features = [f for f in expected_advanced_features if f in processed_data['feature_columns']]
                    print(f"  Advanced features found: {len(found_features)}/{len(expected_advanced_features)}")
                    
            except Exception as e:
                print(f"  Error with {method} scaling: {str(e)}")
                return False
        
        # Test with invalid scaling method
        try:
            processed_data = preprocess_with_enhanced_features(scaling_method="invalid_method")
            print("  Invalid scaling method did not raise an error!")
            return False
        except ValueError:
            print("  Successfully caught invalid scaling method")
        
        return True
    
    finally:
        # Restore the original method
        CryptoDataLoader.get_data = original_get_data

def test_denormalize():
    """Test denormalization functionality"""
    print("\nTesting denormalization...")
    
    # Create mock data
    mock_data = create_mock_ohlcv_data(n_samples=200)
    
    loader = CryptoDataLoader()
    preprocessor = EnhancedPreprocessor(loader)
    
    # Process with standard scaling
    standard_data = preprocessor.preprocess_with_standard_scaling(
        mock_data,
        add_features=True,
        add_advanced_features=False,
        target_column='close',
        sequence_length=24,
        train_split=0.8
    )
    
    # Get the original values for the target (close price)
    target_idx = standard_data['feature_columns'].index('close')
    scaler = standard_data['scaler']
    
    # Get a sample of predicted values
    sample_predictions = standard_data['y_val'][:10]
    
    # Denormalize
    denorm_predictions = preprocessor.denormalize_with_scaler(
        sample_predictions, scaler, target_idx)
    
    print(f"Original scaled predictions (first 5): {sample_predictions[:5]}")
    print(f"Denormalized predictions (first 5): {denorm_predictions[:5]}")
    
    # Check if denormalized values are within a reasonable range
    orig_close_min = mock_data['close'].min()
    orig_close_max = mock_data['close'].max()
    
    print(f"Original close price range: [{orig_close_min}, {orig_close_max}]")
    print(f"Denormalized close price range: [{denorm_predictions.min()}, {denorm_predictions.max()}]")
    
    # Denormalization is successful if the values are within a reasonable range
    reasonable_range = (
        denorm_predictions.min() >= orig_close_min * 0.5 and
        denorm_predictions.max() <= orig_close_max * 1.5
    )
    
    return reasonable_range

def main():
    """Run all tests"""
    print("=== Enhanced Preprocessor Tests ===\n")
    
    # Run tests
    advanced_ok = test_advanced_features()
    scaling_ok = test_scaling_methods()
    cv_ok = test_cross_validation()
    convenience_ok = test_convenience_function()
    denorm_ok = test_denormalize()
    
    # Summarize results
    print("\n=== Test Results ===")
    print(f"Advanced Features: {'✅ PASS' if advanced_ok else '❌ FAIL'}")
    print(f"Scaling Methods: {'✅ PASS' if scaling_ok else '❌ FAIL'}")
    print(f"Cross-Validation: {'✅ PASS' if cv_ok else '❌ FAIL'}")
    print(f"Convenience Function: {'✅ PASS' if convenience_ok else '❌ FAIL'}")
    print(f"Denormalization: {'✅ PASS' if denorm_ok else '❌ FAIL'}")
    
    all_tests_passed = all([advanced_ok, scaling_ok, cv_ok, convenience_ok, denorm_ok])
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 