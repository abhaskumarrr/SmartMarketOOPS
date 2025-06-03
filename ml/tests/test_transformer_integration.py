"""
Comprehensive Tests for Task #24 and Task #25
Tests Transformer Model Integration and Enhanced Signal Quality System
"""

import unittest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.memory_efficient_transformer import MemoryEfficientTransformer
from src.data.transformer_preprocessor import TransformerPreprocessor
from src.ensemble.enhanced_signal_quality_system import (
    EnhancedSignalQualitySystem, TradingSignal, SignalType, SignalQuality
)
from src.integration.transformer_ml_pipeline import TransformerMLPipeline


class TestTransformerIntegration(unittest.TestCase):
    """Test suite for Transformer model integration (Task #24)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 20
        self.output_dim = 1
        self.seq_len = 50
        self.forecast_horizon = 1
        self.batch_size = 8  # Small batch for testing
        
        # Create sample market data
        self.sample_data = self._create_sample_market_data()
        
        # Initialize models
        self.transformer_model = MemoryEfficientTransformer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            seq_len=self.seq_len,
            forecast_horizon=self.forecast_horizon,
            d_model=64,  # Smaller for testing
            num_layers=2,
            nhead=4
        )
        
        self.preprocessor = TransformerPreprocessor(
            sequence_length=self.seq_len,
            forecast_horizon=self.forecast_horizon
        )
    
    def _create_sample_market_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Create sample market data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=num_samples, freq='1H')
        
        # Generate realistic price data
        np.random.seed(42)
        price_base = 50000
        returns = np.random.normal(0, 0.02, num_samples)
        prices = [price_base]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(100, 1000, num_samples)
        })
        
        return data
    
    def test_memory_efficient_transformer_initialization(self):
        """Test memory-efficient Transformer initialization"""
        self.assertIsNotNone(self.transformer_model)
        self.assertEqual(self.transformer_model.input_dim, self.input_dim)
        self.assertEqual(self.transformer_model.output_dim, self.output_dim)
        self.assertEqual(self.transformer_model.seq_len, self.seq_len)
        self.assertTrue(self.transformer_model.use_gradient_checkpointing)
    
    def test_transformer_forward_pass(self):
        """Test Transformer forward pass"""
        batch_size = 4
        x = torch.randn(batch_size, self.seq_len, self.input_dim)
        
        with torch.no_grad():
            output = self.transformer_model(x)
        
        self.assertEqual(output.shape, (batch_size, self.output_dim))
        self.assertFalse(torch.isnan(output).any())
    
    def test_transformer_preprocessor(self):
        """Test Transformer data preprocessing"""
        processed_data = self.preprocessor.fit_transform(
            self.sample_data,
            target_column='close'
        )
        
        self.assertIn('X_train', processed_data)
        self.assertIn('y_train', processed_data)
        self.assertIn('X_val', processed_data)
        self.assertIn('y_val', processed_data)
        self.assertIn('feature_names', processed_data)
        
        # Check shapes
        X_train = processed_data['X_train']
        self.assertEqual(len(X_train.shape), 3)  # [samples, seq_len, features]
        self.assertEqual(X_train.shape[1], self.seq_len)
    
    def test_memory_efficient_training(self):
        """Test memory-efficient training process"""
        # Prepare small dataset for quick training
        small_data = self.sample_data.head(200)
        processed_data = self.preprocessor.fit_transform(small_data, target_column='close')
        
        train_loader, val_loader = self.preprocessor.create_data_loaders(
            processed_data['X_train'],
            processed_data['y_train'],
            processed_data['X_val'],
            processed_data['y_val'],
            batch_size=self.batch_size
        )
        
        # Quick training test (1 epoch)
        results = self.transformer_model.fit_model_memory_efficient(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            lr=0.001
        )
        
        self.assertIn('best_val_loss', results)
        self.assertIn('train_losses', results)
        self.assertIn('val_losses', results)
        self.assertGreater(len(results['train_losses']), 0)
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring"""
        memory_stats = self.transformer_model.get_memory_usage()
        
        self.assertIn('cpu_memory', memory_stats)
        self.assertIsInstance(memory_stats['cpu_memory'], float)
        self.assertGreater(memory_stats['cpu_memory'], 0)
    
    def test_prediction_with_confidence(self):
        """Test prediction with confidence scoring"""
        # Create dummy input
        x = np.random.randn(10, self.seq_len, self.input_dim)
        
        predictions, confidences = self.transformer_model.predict(
            x, return_confidence=True
        )
        
        self.assertEqual(len(predictions), 10)
        self.assertEqual(len(confidences), 10)
        self.assertTrue(all(0 <= c <= 1 for c in confidences))


class TestEnhancedSignalQuality(unittest.TestCase):
    """Test suite for Enhanced Signal Quality System (Task #25)"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock models
        self.transformer_model = self._create_mock_transformer()
        self.ensemble_models = [self._create_mock_model() for _ in range(2)]
        
        # Initialize signal quality system
        self.signal_system = EnhancedSignalQualitySystem(
            transformer_model=self.transformer_model,
            ensemble_models=self.ensemble_models,
            confidence_threshold=0.6
        )
        
        # Create sample market data
        self.market_data = self._create_sample_market_data()
    
    def _create_mock_transformer(self):
        """Create mock Transformer model"""
        class MockTransformer:
            def predict(self, data, return_confidence=False):
                if return_confidence:
                    return np.array([0.7]), np.array([0.8])
                return np.array([0.7])
        
        return MockTransformer()
    
    def _create_mock_model(self):
        """Create mock ensemble model"""
        class MockModel:
            def predict(self, data):
                return 0.6
            
            @property
            def confidence(self):
                return 0.7
        
        return MockModel()
    
    def _create_sample_market_data(self) -> pd.DataFrame:
        """Create sample market data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        prices = np.cumsum(np.random.normal(0, 1, 100)) + 50000
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 100)
        })
    
    def test_signal_generation(self):
        """Test trading signal generation"""
        signal = self.signal_system.generate_signal(
            market_data=self.market_data,
            symbol='BTCUSD',
            current_price=50000.0
        )
        
        self.assertIsInstance(signal, TradingSignal)
        self.assertIsInstance(signal.signal_type, SignalType)
        self.assertIsInstance(signal.quality, SignalQuality)
        self.assertGreaterEqual(signal.confidence, 0)
        self.assertLessEqual(signal.confidence, 1)
        self.assertEqual(signal.symbol, 'BTCUSD')
        self.assertEqual(signal.price, 50000.0)
    
    def test_signal_quality_assessment(self):
        """Test signal quality assessment"""
        signal = self.signal_system.generate_signal(
            market_data=self.market_data,
            symbol='BTCUSD',
            current_price=50000.0
        )
        
        if signal:  # Signal might be None if below threshold
            self.assertIn(signal.quality, [
                SignalQuality.EXCELLENT,
                SignalQuality.GOOD,
                SignalQuality.FAIR,
                SignalQuality.POOR
            ])
            
            # Check quality consistency with confidence
            if signal.confidence >= 0.9:
                self.assertEqual(signal.quality, SignalQuality.EXCELLENT)
            elif signal.confidence >= 0.7:
                self.assertEqual(signal.quality, SignalQuality.GOOD)
    
    def test_risk_metrics_calculation(self):
        """Test risk metrics calculation"""
        signal = self.signal_system.generate_signal(
            market_data=self.market_data,
            symbol='BTCUSD',
            current_price=50000.0
        )
        
        if signal and signal.signal_type != SignalType.HOLD:
            self.assertIsNotNone(signal.stop_loss)
            self.assertIsNotNone(signal.take_profit)
            self.assertIsNotNone(signal.position_size)
            self.assertIsNotNone(signal.risk_reward_ratio)
            
            # Check risk metrics are reasonable
            self.assertGreater(signal.position_size, 0)
            self.assertLess(signal.position_size, 1)  # Should be less than 100%
            self.assertGreater(signal.risk_reward_ratio, 0)
    
    def test_performance_tracking(self):
        """Test performance tracking"""
        # Generate multiple signals
        for i in range(5):
            signal = self.signal_system.generate_signal(
                market_data=self.market_data,
                symbol='BTCUSD',
                current_price=50000.0 + i * 100
            )
        
        # Get performance report
        report = self.signal_system.get_performance_report()
        
        self.assertIn('total_signals', report)
        self.assertIn('average_confidence', report)
        self.assertIn('quality_distribution', report)
        
        if report['total_signals'] > 0:
            self.assertGreaterEqual(report['average_confidence'], 0)
            self.assertLessEqual(report['average_confidence'], 1)
    
    def test_signal_outcome_update(self):
        """Test signal outcome updating"""
        signal = self.signal_system.generate_signal(
            market_data=self.market_data,
            symbol='BTCUSD',
            current_price=50000.0
        )
        
        if signal:
            # Update signal outcome
            signal_id = str(signal.timestamp)
            self.signal_system.update_signal_outcome(signal_id, was_profitable=True)
            
            # Check that metrics were updated
            report = self.signal_system.get_performance_report()
            self.assertGreaterEqual(report['total_signals'], 1)


class TestTransformerMLPipeline(unittest.TestCase):
    """Test suite for complete ML pipeline integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = TransformerMLPipeline(
            use_memory_efficient=True,
            enable_ensemble=True
        )
        
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for pipeline testing"""
        dates = pd.date_range(start='2023-01-01', periods=500, freq='1H')
        np.random.seed(42)
        
        prices = np.cumsum(np.random.normal(0, 0.01, 500)) + 50000
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': np.random.uniform(100, 1000, 500)
        })
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.pipeline.initialize_models(
            input_dim=10,
            output_dim=1,
            seq_len=50,
            forecast_horizon=1
        )
        
        self.assertIsNotNone(self.pipeline.transformer_model)
        self.assertIsNotNone(self.pipeline.preprocessor)
        self.assertIsNotNone(self.pipeline.signal_quality_system)
        
        if self.pipeline.enable_ensemble:
            self.assertGreater(len(self.pipeline.legacy_models), 0)
    
    def test_pipeline_training(self):
        """Test complete pipeline training"""
        self.pipeline.initialize_models(
            input_dim=5,  # Basic OHLCV
            output_dim=1,
            seq_len=30,  # Shorter for testing
            forecast_horizon=1
        )
        
        # Quick training test
        results = self.pipeline.train_pipeline(
            train_data=self.sample_data,
            num_epochs=1,  # Single epoch for testing
            batch_size=8
        )
        
        self.assertIn('transformer_results', results)
        self.assertIn('evaluation_results', results)
        self.assertIn('performance_metrics', results)
        self.assertIn('training_time', results)
        
        # Check that training completed
        self.assertGreater(results['training_time'], 0)
    
    def test_pipeline_prediction(self):
        """Test pipeline prediction"""
        self.pipeline.initialize_models(
            input_dim=5,
            output_dim=1,
            seq_len=30,
            forecast_horizon=1
        )
        
        # Test direct prediction
        prediction = self.pipeline.predict(
            market_data=self.sample_data,
            symbol='BTCUSD',
            return_signal=False
        )
        
        self.assertIsNotNone(prediction)
        
        # Test signal generation
        signal = self.pipeline.predict(
            market_data=self.sample_data,
            symbol='BTCUSD',
            return_signal=True
        )
        
        # Signal might be None if below confidence threshold
        if signal:
            self.assertIsInstance(signal, TradingSignal)
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        summary = self.pipeline.get_performance_summary()
        
        self.assertIn('performance_metrics', summary)
        self.assertIn('model_info', summary)
        self.assertIn('signal_quality_metrics', summary)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
