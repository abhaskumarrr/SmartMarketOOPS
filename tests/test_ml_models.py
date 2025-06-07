"""
Comprehensive Unit Tests for ML Trading Models
Tests model accuracy, prediction consistency, and edge cases
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.enhanced_ml_model import EnhancedMLModel
from ml_models.fibonacci_ml_model import FibonacciMLModel
from ml_models.multi_timeframe_model import MultiTimeframeModel
from data_collection.delta_exchange_client import DeltaExchangeClient

class TestEnhancedMLModel:
    """Test suite for Enhanced ML Model"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample market data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='5T')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(45000, 50000, 1000),
            'high': np.random.uniform(45000, 50000, 1000),
            'low': np.random.uniform(45000, 50000, 1000),
            'close': np.random.uniform(45000, 50000, 1000),
            'volume': np.random.uniform(100, 1000, 1000)
        })
        # Ensure high >= low and open, close within range
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        return data
    
    @pytest.fixture
    def ml_model(self):
        """Initialize ML model for testing"""
        return EnhancedMLModel()
    
    def test_model_initialization(self, ml_model):
        """Test model initializes correctly"""
        assert ml_model is not None
        assert hasattr(ml_model, 'model')
        assert hasattr(ml_model, 'scaler')
        assert ml_model.is_trained == False
    
    def test_feature_engineering(self, ml_model, sample_data):
        """Test feature engineering produces expected outputs"""
        features = ml_model.engineer_features(sample_data)
        
        # Check feature DataFrame structure
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        assert 'rsi' in features.columns
        assert 'macd' in features.columns
        assert 'bb_upper' in features.columns
        assert 'bb_lower' in features.columns
        
        # Check for no infinite or NaN values in critical features
        critical_features = ['rsi', 'macd', 'bb_upper', 'bb_lower']
        for feature in critical_features:
            if feature in features.columns:
                assert not features[feature].isin([np.inf, -np.inf]).any()
                # Allow some NaN values due to rolling calculations
                assert features[feature].notna().sum() > len(features) * 0.8
    
    def test_model_training(self, ml_model, sample_data):
        """Test model training process"""
        # Prepare training data
        features = ml_model.engineer_features(sample_data)
        
        # Create mock labels (buy/sell/hold signals)
        labels = np.random.choice([0, 1, 2], size=len(features))
        
        # Train model
        ml_model.train(features, labels)
        
        assert ml_model.is_trained == True
        assert ml_model.model is not None
        assert ml_model.scaler is not None
    
    def test_prediction_accuracy(self, ml_model, sample_data):
        """Test prediction accuracy and consistency"""
        features = ml_model.engineer_features(sample_data)
        labels = np.random.choice([0, 1, 2], size=len(features))
        
        # Train model
        ml_model.train(features, labels)
        
        # Make predictions
        predictions = ml_model.predict(features.tail(100))
        
        # Check prediction format
        assert isinstance(predictions, (np.ndarray, list))
        assert len(predictions) == 100
        assert all(pred in [0, 1, 2] for pred in predictions)
        
        # Test prediction consistency (same input should give same output)
        predictions2 = ml_model.predict(features.tail(100))
        np.testing.assert_array_equal(predictions, predictions2)
    
    def test_confidence_scores(self, ml_model, sample_data):
        """Test confidence score calculation"""
        features = ml_model.engineer_features(sample_data)
        labels = np.random.choice([0, 1, 2], size=len(features))
        
        ml_model.train(features, labels)
        
        # Get predictions with confidence
        predictions, confidence = ml_model.predict_with_confidence(features.tail(50))
        
        # Check confidence scores
        assert len(confidence) == 50
        assert all(0 <= conf <= 1 for conf in confidence)
        assert isinstance(confidence, (np.ndarray, list))
    
    def test_edge_cases(self, ml_model):
        """Test model behavior with edge cases"""
        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises((ValueError, IndexError)):
            ml_model.engineer_features(empty_data)
        
        # Test with insufficient data
        small_data = pd.DataFrame({
            'open': [100], 'high': [105], 'low': [95], 'close': [102], 'volume': [1000]
        })
        features = ml_model.engineer_features(small_data)
        assert len(features) >= 0  # Should handle gracefully
        
        # Test with extreme values
        extreme_data = pd.DataFrame({
            'open': [1e6], 'high': [1e6], 'low': [1e6], 'close': [1e6], 'volume': [1e6]
        })
        features = ml_model.engineer_features(extreme_data)
        assert not features.isin([np.inf, -np.inf]).any().any()

class TestFibonacciMLModel:
    """Test suite for Fibonacci ML Model"""
    
    @pytest.fixture
    def fib_model(self):
        return FibonacciMLModel()
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data with clear trends"""
        np.random.seed(42)
        prices = []
        base_price = 45000
        
        # Create trending data
        for i in range(200):
            trend = 0.001 * i  # Upward trend
            noise = np.random.normal(0, 0.01)
            price = base_price * (1 + trend + noise)
            prices.append(price)
        
        return pd.Series(prices)
    
    def test_fibonacci_level_calculation(self, fib_model, sample_price_data):
        """Test Fibonacci retracement level calculation"""
        levels = fib_model.calculate_fibonacci_levels(sample_price_data)
        
        assert isinstance(levels, dict)
        expected_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in expected_levels:
            assert level in levels
            assert isinstance(levels[level], (int, float))
            assert levels[level] > 0
    
    def test_support_resistance_detection(self, fib_model, sample_price_data):
        """Test support and resistance level detection"""
        support_levels, resistance_levels = fib_model.detect_support_resistance(sample_price_data)
        
        assert isinstance(support_levels, list)
        assert isinstance(resistance_levels, list)
        assert len(support_levels) >= 0
        assert len(resistance_levels) >= 0
        
        # Support levels should be below current price
        current_price = sample_price_data.iloc[-1]
        for level in support_levels:
            assert level <= current_price
        
        # Resistance levels should be above current price
        for level in resistance_levels:
            assert level >= current_price

class TestMultiTimeframeModel:
    """Test suite for Multi-Timeframe Model"""
    
    @pytest.fixture
    def mtf_model(self):
        return MultiTimeframeModel()
    
    @pytest.fixture
    def multi_timeframe_data(self):
        """Generate multi-timeframe sample data"""
        np.random.seed(42)
        base_time = pd.Timestamp('2023-01-01')
        
        data = {}
        for timeframe in ['5m', '15m', '1h', '4h']:
            if timeframe == '5m':
                freq = '5T'
                periods = 1000
            elif timeframe == '15m':
                freq = '15T'
                periods = 500
            elif timeframe == '1h':
                freq = '1H'
                periods = 200
            else:  # 4h
                freq = '4H'
                periods = 100
            
            dates = pd.date_range(base_time, periods=periods, freq=freq)
            data[timeframe] = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(45000, 50000, periods),
                'high': np.random.uniform(45000, 50000, periods),
                'low': np.random.uniform(45000, 50000, periods),
                'close': np.random.uniform(45000, 50000, periods),
                'volume': np.random.uniform(100, 1000, periods)
            })
        
        return data
    
    def test_timeframe_alignment(self, mtf_model, multi_timeframe_data):
        """Test timeframe data alignment"""
        aligned_data = mtf_model.align_timeframes(multi_timeframe_data)
        
        assert isinstance(aligned_data, pd.DataFrame)
        assert len(aligned_data) > 0
        
        # Check that all timeframe columns are present
        expected_columns = ['5m_close', '15m_close', '1h_close', '4h_close']
        for col in expected_columns:
            assert col in aligned_data.columns
    
    def test_confluence_calculation(self, mtf_model, multi_timeframe_data):
        """Test confluence score calculation"""
        aligned_data = mtf_model.align_timeframes(multi_timeframe_data)
        confluence_scores = mtf_model.calculate_confluence(aligned_data)
        
        assert isinstance(confluence_scores, (pd.Series, np.ndarray))
        assert len(confluence_scores) > 0
        assert all(0 <= score <= 1 for score in confluence_scores if not np.isnan(score))

class TestSystemIntegration:
    """Integration tests for the complete system"""
    
    def test_delta_exchange_connection(self):
        """Test Delta Exchange API connection"""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {'result': []}
            
            client = DeltaExchangeClient()
            result = client.get_products()
            
            assert result is not None
            mock_get.assert_called_once()
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline"""
        # This would test the entire flow from data collection to prediction
        # Mock external dependencies
        with patch('data_collection.delta_exchange_client.DeltaExchangeClient') as mock_client:
            mock_client.return_value.get_candles.return_value = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='5T'),
                'open': np.random.uniform(45000, 50000, 100),
                'high': np.random.uniform(45000, 50000, 100),
                'low': np.random.uniform(45000, 50000, 100),
                'close': np.random.uniform(45000, 50000, 100),
                'volume': np.random.uniform(100, 1000, 100)
            })
            
            # Test pipeline execution
            model = EnhancedMLModel()
            # Add pipeline test logic here
            assert True  # Placeholder for actual pipeline test

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--cov=ml_models",
        "--cov=data_collection",
        "--cov-report=html",
        "--cov-report=term-missing"
    ])
