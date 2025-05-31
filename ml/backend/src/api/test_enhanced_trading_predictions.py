#!/usr/bin/env python3
"""
Test suite for Enhanced Trading Predictions API

This module contains comprehensive tests for the Enhanced Trading Predictions system
that integrates ML predictions with Smart Money Concepts analysis.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from unittest.mock import Mock, patch

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from enhanced_trading_predictions import (
        EnhancedTradingPredictor, TradingPredictionInput, TradingPredictionOutput,
        TradingSignal
    )
    from ml.src.api.model_service import ModelService
    ENHANCED_PREDICTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced trading predictions not available: {e}")
    ENHANCED_PREDICTIONS_AVAILABLE = False


@unittest.skipIf(not ENHANCED_PREDICTIONS_AVAILABLE, "Enhanced trading predictions not available")
class TestEnhancedTradingPredictions(unittest.TestCase):
    """Test cases for Enhanced Trading Predictions"""
    
    def setUp(self):
        """Set up test environment"""
        # Create mock model service
        self.mock_model_service = Mock(spec=ModelService)
        
        # Configure mock model service to return sample predictions
        self.mock_model_service.predict.return_value = {
            'symbol': 'BTC/USDT',
            'predictions': [0.2, 0.3, 0.5],  # [down, neutral, up]
            'confidence': 0.7,
            'predicted_direction': 'up',
            'prediction_time': datetime.now().isoformat(),
            'model_version': 'test-v1.0'
        }
        
        # Initialize enhanced trading predictor
        self.predictor = EnhancedTradingPredictor(self.mock_model_service)
    
    def test_predictor_initialization(self):
        """Test EnhancedTradingPredictor initialization"""
        self.assertIsInstance(self.predictor, EnhancedTradingPredictor)
        self.assertEqual(self.predictor.model_service, self.mock_model_service)
        self.assertIsInstance(self.predictor.risk_configs, dict)
        self.assertIn('low', self.predictor.risk_configs)
        self.assertIn('medium', self.predictor.risk_configs)
        self.assertIn('high', self.predictor.risk_configs)
    
    def test_trading_prediction_input_validation(self):
        """Test TradingPredictionInput validation"""
        # Valid input
        valid_input = TradingPredictionInput(
            symbol="BTC/USDT",
            timeframe="15m",
            include_smc=True,
            include_confluence=True,
            confidence_threshold=0.6,
            risk_level="medium"
        )
        
        self.assertEqual(valid_input.symbol, "BTC/USDT")
        self.assertEqual(valid_input.timeframe, "15m")
        self.assertTrue(valid_input.include_smc)
        self.assertTrue(valid_input.include_confluence)
        self.assertEqual(valid_input.confidence_threshold, 0.6)
        self.assertEqual(valid_input.risk_level, "medium")
    
    def test_sample_ohlcv_data_generation(self):
        """Test sample OHLCV data generation"""
        sample_data = self.predictor._generate_sample_ohlcv_data(100)
        
        self.assertIsInstance(sample_data, pd.DataFrame)
        self.assertEqual(len(sample_data), 100)
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, sample_data.columns)
        
        # Check OHLC relationships
        for _, row in sample_data.iterrows():
            self.assertGreaterEqual(row['high'], row['open'])
            self.assertGreaterEqual(row['high'], row['close'])
            self.assertLessEqual(row['low'], row['open'])
            self.assertLessEqual(row['low'], row['close'])
            self.assertGreater(row['volume'], 0)
    
    def test_ml_prediction_integration(self):
        """Test ML prediction integration"""
        input_data = TradingPredictionInput(
            symbol="BTC/USDT",
            timeframe="15m"
        )
        
        ml_prediction = self.predictor._get_ml_prediction(input_data)
        
        self.assertIsInstance(ml_prediction, dict)
        self.assertIn('direction', ml_prediction)
        self.assertIn('confidence', ml_prediction)
        self.assertIn('probabilities', ml_prediction)
        self.assertIn('model_version', ml_prediction)
        
        # Verify mock was called
        self.mock_model_service.predict.assert_called_once()
    
    def test_confidence_calculation(self):
        """Test combined confidence calculation"""
        ml_prediction = {
            'direction': 'up',
            'confidence': 0.7
        }
        
        smc_analysis = {
            'smc_bias': 'bullish',
            'institutional_activity': 'medium'
        }
        
        confluence_analysis = {
            'confluence_score': 0.8,
            'market_timing': 0.7
        }
        
        combined_confidence = self.predictor._calculate_combined_confidence(
            ml_prediction, smc_analysis, confluence_analysis
        )
        
        self.assertIsInstance(combined_confidence, float)
        self.assertGreaterEqual(combined_confidence, 0.0)
        self.assertLessEqual(combined_confidence, 1.0)
        self.assertGreater(combined_confidence, 0.7)  # Should be higher than base ML confidence
    
    def test_direction_determination(self):
        """Test primary direction determination"""
        ml_prediction = {'direction': 'up'}
        smc_analysis = {'smc_bias': 'bullish'}
        confluence_analysis = {'best_signal': {'type': 'buy'}}
        
        direction = self.predictor._determine_primary_direction(
            ml_prediction, smc_analysis, confluence_analysis
        )
        
        self.assertEqual(direction, 'buy')
        
        # Test conflicting signals
        ml_prediction = {'direction': 'down'}
        smc_analysis = {'smc_bias': 'bearish'}
        confluence_analysis = {'best_signal': {'type': 'buy'}}
        
        direction = self.predictor._determine_primary_direction(
            ml_prediction, smc_analysis, confluence_analysis
        )
        
        self.assertIn(direction, ['buy', 'sell', 'hold'])
    
    def test_trading_signal_creation(self):
        """Test trading signal creation"""
        ml_prediction = {'direction': 'up', 'confidence': 0.7}
        smc_analysis = {'smc_bias': 'bullish', 'institutional_activity': 'medium'}
        confluence_analysis = {'confluence_score': 0.8}
        
        input_data = TradingPredictionInput(
            symbol="BTC/USDT",
            timeframe="15m"
        )
        
        signal = self.predictor._create_trading_signal(
            'buy', 0.75, ml_prediction, smc_analysis, confluence_analysis, input_data
        )
        
        self.assertIsInstance(signal, TradingSignal)
        self.assertEqual(signal.signal_type, 'buy')
        self.assertEqual(signal.confidence, 0.75)
        self.assertIn(signal.strength, ['weak', 'moderate', 'strong', 'very_strong'])
        self.assertIsInstance(signal.reasoning, list)
        self.assertGreater(len(signal.reasoning), 0)
    
    def test_risk_assessment(self):
        """Test risk assessment functionality"""
        # Create a sample signal
        signal = TradingSignal(
            signal_type='buy',
            confidence=0.7,
            strength='moderate',
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            risk_reward_ratio=2.0,
            timeframe='15m',
            reasoning=['Test signal']
        )
        
        signals = {'primary': signal, 'alternatives': []}
        
        input_data = TradingPredictionInput(
            symbol="BTC/USDT",
            risk_level="medium"
        )
        
        risk_assessment = self.predictor._assess_risk(signals, input_data)
        
        self.assertIsInstance(risk_assessment, dict)
        self.assertIn('risk_level', risk_assessment)
        self.assertIn('max_risk_per_trade', risk_assessment)
        self.assertIn('confidence_threshold', risk_assessment)
        self.assertIn('signal_meets_threshold', risk_assessment)
        self.assertIn('recommended_position_size', risk_assessment)
        self.assertIn('risk_factors', risk_assessment)
        self.assertIn('risk_mitigation', risk_assessment)
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        signal = TradingSignal(
            signal_type='buy',
            confidence=0.8,
            strength='strong',
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            risk_reward_ratio=2.0,
            timeframe='15m',
            reasoning=['Test signal']
        )
        
        position_size = self.predictor._calculate_position_size(signal, 0.02)
        
        self.assertIsInstance(position_size, dict)
        self.assertIn('percentage_of_portfolio', position_size)
        self.assertIn('reasoning', position_size)
        
        # Test hold signal
        hold_signal = TradingSignal(
            signal_type='hold',
            confidence=0.5,
            strength='weak',
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            risk_reward_ratio=None,
            timeframe='15m',
            reasoning=['Hold signal']
        )
        
        hold_position_size = self.predictor._calculate_position_size(hold_signal, 0.02)
        self.assertEqual(hold_position_size['percentage_of_portfolio'], 0.0)
    
    def test_market_regime_determination(self):
        """Test market regime determination"""
        # Create trending up data
        trending_up_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105] * 10,
            'timestamp': pd.date_range(start='2023-01-01', periods=60, freq='15T')
        })
        
        regime = self.predictor._determine_market_regime(trending_up_data)
        self.assertEqual(regime, 'trending_up')
        
        # Test insufficient data
        small_data = pd.DataFrame({
            'close': [100, 101, 102],
            'timestamp': pd.date_range(start='2023-01-01', periods=3, freq='15T')
        })
        
        regime = self.predictor._determine_market_regime(small_data)
        self.assertEqual(regime, 'insufficient_data')
    
    @patch('ml.backend.src.api.enhanced_trading_predictions.SMC_AVAILABLE', True)
    def test_comprehensive_prediction_flow(self):
        """Test the complete prediction flow"""
        input_data = TradingPredictionInput(
            symbol="BTC/USDT",
            timeframe="15m",
            include_smc=False,  # Disable SMC to avoid import issues in test
            include_confluence=False,  # Disable confluence to avoid import issues in test
            confidence_threshold=0.6,
            risk_level="medium"
        )
        
        # Mock the prediction methods to avoid SMC dependencies
        with patch.object(self.predictor, '_get_smc_analysis', return_value=None), \
             patch.object(self.predictor, '_get_confluence_analysis', return_value=None):
            
            result = self.predictor.predict_trading_signals(input_data)
            
            self.assertIsInstance(result, TradingPredictionOutput)
            self.assertEqual(result.symbol, "BTC/USDT")
            self.assertEqual(result.primary_timeframe, "15m")
            self.assertIsInstance(result.ml_prediction, dict)
            self.assertIsInstance(result.primary_signal, TradingSignal)
            self.assertIsInstance(result.alternative_signals, list)
            self.assertIsInstance(result.risk_assessment, dict)
            self.assertIsInstance(result.market_context, dict)
    
    def test_alternative_signal_generation(self):
        """Test alternative signal generation"""
        ml_prediction = {'direction': 'up', 'confidence': 0.5}  # Low confidence
        
        input_data = TradingPredictionInput(
            symbol="BTC/USDT",
            timeframe="15m"
        )
        
        alternatives = self.predictor._generate_alternative_signals(
            ml_prediction, None, None, input_data
        )
        
        self.assertIsInstance(alternatives, list)
        # Should generate alternatives for low confidence signal
        self.assertGreater(len(alternatives), 0)
    
    def test_risk_factor_identification(self):
        """Test risk factor identification"""
        # Low confidence signal
        low_confidence_signal = TradingSignal(
            signal_type='buy',
            confidence=0.4,  # Low confidence
            strength='weak',
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=51000.0,  # Poor R:R ratio
            risk_reward_ratio=1.0,
            timeframe='1m',  # Short timeframe
            reasoning=['Test signal']
        )
        
        input_data = TradingPredictionInput(
            symbol="BTC/USDT",
            timeframe="1m"
        )
        
        risk_factors = self.predictor._identify_risk_factors(low_confidence_signal, input_data)
        
        self.assertIsInstance(risk_factors, list)
        self.assertGreater(len(risk_factors), 0)
        
        # Check for expected risk factors
        risk_text = ' '.join(risk_factors)
        self.assertIn('confidence', risk_text.lower())
        self.assertIn('timeframe', risk_text.lower())


class TestTradingPredictionModels(unittest.TestCase):
    """Test cases for trading prediction data models"""
    
    def test_trading_signal_model(self):
        """Test TradingSignal model"""
        signal = TradingSignal(
            signal_type='buy',
            confidence=0.75,
            strength='strong',
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
            risk_reward_ratio=2.0,
            timeframe='15m',
            reasoning=['ML prediction: up', 'SMC bias: bullish']
        )
        
        self.assertEqual(signal.signal_type, 'buy')
        self.assertEqual(signal.confidence, 0.75)
        self.assertEqual(signal.strength, 'strong')
        self.assertEqual(signal.entry_price, 50000.0)
        self.assertEqual(signal.stop_loss, 49000.0)
        self.assertEqual(signal.take_profit, 52000.0)
        self.assertEqual(signal.risk_reward_ratio, 2.0)
        self.assertEqual(signal.timeframe, '15m')
        self.assertIsInstance(signal.reasoning, list)
        self.assertEqual(len(signal.reasoning), 2)


if __name__ == '__main__':
    # Run the tests
    print("Running Enhanced Trading Predictions Tests...")
    print("=" * 60)
    
    if not ENHANCED_PREDICTIONS_AVAILABLE:
        print("⚠️  Enhanced trading predictions not available - skipping tests")
        print("This is expected if dependencies are not installed")
    else:
        unittest.main(verbosity=2, exit=False)
