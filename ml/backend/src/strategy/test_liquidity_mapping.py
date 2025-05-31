#!/usr/bin/env python3
"""
Test suite for the Liquidity Level Mapping System

This module contains comprehensive tests for the enhanced Liquidity Mapping System
that implements Smart Money Concepts for institutional liquidity level identification,
including equal highs/lows, BSL/SSL detection, and liquidity sweep analysis.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liquidity_mapping import (
    LiquidityMapper, LiquidityLevel, LiquiditySweep, LiquidityType, LiquidityStatus, SweepType,
    create_sample_liquidity_data, get_enhanced_liquidity_levels, get_liquidity_sweeps,
    integrate_liquidity_mapper_with_smc
)


class TestLiquidityMapping(unittest.TestCase):
    """Test cases for Liquidity Level Mapping System"""
    
    def setUp(self):
        """Set up test data and mapper instance"""
        # Create sample OHLCV data with intentional liquidity levels
        self.sample_data = create_sample_liquidity_data(300)
        self.mapper = LiquidityMapper(
            self.sample_data,
            equal_tolerance_pct=0.002,
            min_touches=2,
            lookback_period=50
        )
    
    def test_liquidity_mapper_initialization(self):
        """Test LiquidityMapper initialization"""
        self.assertIsInstance(self.mapper, LiquidityMapper)
        self.assertIsInstance(self.mapper.ohlcv, pd.DataFrame)
        self.assertTrue(len(self.mapper.ohlcv) > 0)
        self.assertIn('swing_high', self.mapper.ohlcv.columns)
        self.assertIn('swing_low', self.mapper.ohlcv.columns)
        self.assertIn('volume_ratio', self.mapper.ohlcv.columns)
        self.assertIn('atr', self.mapper.ohlcv.columns)
    
    def test_equal_levels_identification(self):
        """Test equal highs and lows identification"""
        equal_levels = self.mapper.identify_equal_levels()
        
        # Should return a list
        self.assertIsInstance(equal_levels, list)
        
        # Check if any equal levels were detected
        if equal_levels:
            # Test first equal level structure
            level = equal_levels[0]
            self.assertIsInstance(level, LiquidityLevel)
            self.assertIn(level.type, [LiquidityType.EQUAL_HIGHS, LiquidityType.EQUAL_LOWS])
            self.assertGreater(level.strength, 0)
            self.assertGreaterEqual(level.touches, self.mapper.min_touches)
            self.assertGreaterEqual(level.equal_level_count, 1)
    
    def test_bsl_ssl_identification(self):
        """Test Buy-Side and Sell-Side Liquidity identification"""
        bsl_ssl_levels = self.mapper.identify_bsl_ssl_levels()
        
        # Should return a list
        self.assertIsInstance(bsl_ssl_levels, list)
        
        # Check if any BSL/SSL levels were detected
        if bsl_ssl_levels:
            # Test first BSL/SSL level structure
            level = bsl_ssl_levels[0]
            self.assertIsInstance(level, LiquidityLevel)
            self.assertIn(level.type, [LiquidityType.BUY_SIDE, LiquidityType.SELL_SIDE])
            self.assertGreater(level.strength, 0)
            self.assertGreaterEqual(level.touches, self.mapper.min_touches)
    
    def test_comprehensive_liquidity_mapping(self):
        """Test comprehensive liquidity mapping functionality"""
        levels_by_type = self.mapper.map_all_liquidity_levels()
        
        # Should return dictionary with expected keys
        expected_keys = ['equal_highs', 'equal_lows', 'buy_side', 'sell_side']
        for key in expected_keys:
            self.assertIn(key, levels_by_type)
            self.assertIsInstance(levels_by_type[key], list)
        
        # Should have populated instance variables
        self.assertIsInstance(self.mapper.liquidity_levels, list)
        self.assertIsInstance(self.mapper.liquidity_sweeps, list)
    
    def test_liquidity_sweep_detection(self):
        """Test liquidity sweep detection and classification"""
        # First map levels
        self.mapper.map_all_liquidity_levels()
        
        # Check sweeps
        sweeps = self.mapper.liquidity_sweeps
        self.assertIsInstance(sweeps, list)
        
        if sweeps:
            # Test first sweep structure
            sweep = sweeps[0]
            self.assertIsInstance(sweep, LiquiditySweep)
            self.assertIsInstance(sweep.level, LiquidityLevel)
            self.assertIn(sweep.sweep_type, [SweepType.STOP_HUNT, SweepType.LIQUIDITY_GRAB, 
                                           SweepType.BREAKOUT, SweepType.FALSE_BREAKOUT])
            self.assertGreaterEqual(sweep.reversal_strength, 0)
            self.assertGreaterEqual(sweep.volume_spike, 0)
            self.assertGreaterEqual(sweep.institutional_signature, 0)
            self.assertLessEqual(sweep.institutional_signature, 1)
    
    def test_liquidity_statistics(self):
        """Test liquidity statistics calculation"""
        # Map levels first
        self.mapper.map_all_liquidity_levels()
        stats = self.mapper.get_liquidity_statistics()
        
        # Should return dictionary with expected keys
        expected_keys = [
            'total_levels', 'levels_by_type', 'total_sweeps', 'sweeps_by_type',
            'average_level_strength', 'stop_hunt_rate', 'institutional_signature_avg',
            'active_levels', 'swept_levels'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Values should be non-negative
        self.assertGreaterEqual(stats['total_levels'], 0)
        self.assertGreaterEqual(stats['total_sweeps'], 0)
        self.assertGreaterEqual(stats['average_level_strength'], 0)
        self.assertGreaterEqual(stats['stop_hunt_rate'], 0)
        self.assertLessEqual(stats['stop_hunt_rate'], 1)
    
    def test_active_levels_near_price(self):
        """Test active levels near price functionality"""
        # Map levels first
        self.mapper.map_all_liquidity_levels()
        
        if self.mapper.liquidity_levels:
            # Test with current price
            current_price = self.sample_data['close'].iloc[-1]
            active_levels = self.mapper.get_active_levels_near_price(current_price)
            
            self.assertIsInstance(active_levels, list)
            
            # All returned levels should be active and within proximity
            for level in active_levels:
                self.assertEqual(level.status, LiquidityStatus.ACTIVE)
                proximity = abs(level.price - current_price) / current_price
                self.assertLessEqual(proximity, 0.05)  # Default 5% proximity
    
    def test_stop_hunt_analysis(self):
        """Test stop hunt pattern analysis"""
        # Map levels first
        self.mapper.map_all_liquidity_levels()
        analysis = self.mapper.analyze_stop_hunt_patterns()
        
        # Should return dictionary with expected keys
        expected_keys = [
            'total_stop_hunts', 'average_reversal_strength', 'average_volume_spike',
            'success_rate', 'institutional_signature_avg', 'by_level_type'
        ]
        
        for key in expected_keys:
            self.assertIn(key, analysis)
        
        # Values should be non-negative
        self.assertGreaterEqual(analysis['total_stop_hunts'], 0)
        self.assertGreaterEqual(analysis['success_rate'], 0)
        self.assertLessEqual(analysis['success_rate'], 1)
    
    def test_level_quality_validation(self):
        """Test level quality validation"""
        # Map levels first
        self.mapper.map_all_liquidity_levels()
        
        if self.mapper.liquidity_levels:
            level = self.mapper.liquidity_levels[0]
            validation = self.mapper.validate_level_quality(level)
            
            # Should return dictionary with expected structure
            self.assertIn('overall_strength', validation)
            self.assertIn('criteria', validation)
            self.assertIn('metrics', validation)
            self.assertIn('status_info', validation)
            self.assertIn('quality_level', validation)
            
            # Quality level should be valid
            self.assertIn(validation['quality_level'], ['high', 'medium', 'low'])
    
    def test_convenience_functions(self):
        """Test convenience functions for getting liquidity data"""
        enhanced_levels = get_enhanced_liquidity_levels(self.sample_data)
        liquidity_sweeps = get_liquidity_sweeps(self.sample_data)
        
        self.assertIsInstance(enhanced_levels, list)
        self.assertIsInstance(liquidity_sweeps, list)
        
        if enhanced_levels:
            level = enhanced_levels[0]
            # Should have all expected fields
            expected_fields = [
                'type', 'price', 'strength', 'formation_timestamp', 'touches',
                'status', 'equal_level_count', 'volume_context'
            ]
            
            for field in expected_fields:
                self.assertIn(field, level)
        
        if liquidity_sweeps:
            sweep = liquidity_sweeps[0]
            # Should have all expected fields
            expected_fields = [
                'level_type', 'level_price', 'sweep_type', 'reversal_strength',
                'volume_spike', 'institutional_signature', 'is_stop_hunt'
            ]
            
            for field in expected_fields:
                self.assertIn(field, sweep)


class TestLiquidityLevelDataClass(unittest.TestCase):
    """Test cases for LiquidityLevel data class"""
    
    def setUp(self):
        """Set up test liquidity level"""
        self.level = LiquidityLevel(
            type=LiquidityType.EQUAL_HIGHS,
            price=50000.0,
            strength=0.75,
            formation_timestamp=datetime.now(),
            formation_index=100,
            touches=3,
            equal_level_count=3,
            volume_context={'avg_volume_ratio': 1.5}
        )
    
    def test_level_creation(self):
        """Test LiquidityLevel creation and basic properties"""
        self.assertEqual(self.level.type, LiquidityType.EQUAL_HIGHS)
        self.assertEqual(self.level.price, 50000.0)
        self.assertEqual(self.level.strength, 0.75)
        self.assertEqual(self.level.touches, 3)
        self.assertEqual(self.level.status, LiquidityStatus.ACTIVE)
    
    def test_price_near_level_method(self):
        """Test is_price_near_level method"""
        # Price near level
        self.assertTrue(self.level.is_price_near_level(50050.0, 0.002))  # 0.2% tolerance
        
        # Price far from level
        self.assertFalse(self.level.is_price_near_level(51000.0, 0.002))
        
        # Price at exact level
        self.assertTrue(self.level.is_price_near_level(50000.0, 0.002))
    
    def test_strength_calculation(self):
        """Test calculate_strength method"""
        calculated_strength = self.level.calculate_strength()
        
        # Should return a value between 0 and 1
        self.assertGreaterEqual(calculated_strength, 0)
        self.assertLessEqual(calculated_strength, 1)
        
        # Should consider touches, equal levels, and volume
        self.assertGreater(calculated_strength, 0)  # Should have some strength


class TestSampleDataGeneration(unittest.TestCase):
    """Test cases for sample data generation"""
    
    def test_create_sample_liquidity_data(self):
        """Test sample liquidity data creation"""
        data = create_sample_liquidity_data(200)
        
        # Should return DataFrame
        self.assertIsInstance(data, pd.DataFrame)
        
        # Should have correct number of rows (approximately)
        self.assertGreater(len(data), 100)  # Should have substantial data
        
        # Should have required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # OHLC relationships should be valid
        for _, row in data.iterrows():
            self.assertGreaterEqual(row['high'], row['open'])
            self.assertGreaterEqual(row['high'], row['close'])
            self.assertLessEqual(row['low'], row['open'])
            self.assertLessEqual(row['low'], row['close'])
            self.assertGreater(row['volume'], 0)


class TestIntegrationFunction(unittest.TestCase):
    """Test cases for SMC integration"""
    
    def test_integration_decorator(self):
        """Test the SMC integration decorator"""
        # Create a mock SMC detector class
        class MockSMCDetector:
            def __init__(self, ohlcv):
                self.ohlcv = ohlcv
            
            def detect_all(self):
                return {'order_blocks': [], 'fvg': []}
        
        # Apply the integration decorator
        EnhancedSMCDetector = integrate_liquidity_mapper_with_smc(MockSMCDetector)
        
        # Test enhanced detector
        sample_data = create_sample_liquidity_data(100)
        enhanced_detector = EnhancedSMCDetector(sample_data)
        
        # Should have liquidity mapper
        self.assertIsInstance(enhanced_detector.liquidity_mapper, LiquidityMapper)
        
        # Should have enhanced methods
        self.assertTrue(hasattr(enhanced_detector, 'detect_liquidity_levels'))
        self.assertTrue(hasattr(enhanced_detector, 'get_liquidity_statistics'))
        self.assertTrue(hasattr(enhanced_detector, 'get_active_liquidity_near_price'))
        self.assertTrue(hasattr(enhanced_detector, 'analyze_stop_hunts'))
        self.assertTrue(hasattr(enhanced_detector, 'detect_all_enhanced'))


if __name__ == '__main__':
    # Run the tests
    print("Running Liquidity Level Mapping System Tests...")
    print("=" * 60)
    
    # Run tests using unittest.main() for modern approach
    unittest.main(verbosity=2, exit=False)
