#!/usr/bin/env python3
"""
Test suite for the Fair Value Gap (FVG) Detection System

This module contains comprehensive tests for the enhanced FVG Detection System
that implements Smart Money Concepts for institutional Fair Value Gap identification.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fvg_detection import (
    FVGDetector, FairValueGap, FVGType, FVGStatus,
    create_sample_fvg_data, get_enhanced_fvgs, integrate_fvg_detector_with_smc
)


class TestFVGDetection(unittest.TestCase):
    """Test cases for Fair Value Gap Detection System"""
    
    def setUp(self):
        """Set up test data and detector instance"""
        # Create sample OHLCV data with intentional FVGs
        self.sample_data = create_sample_fvg_data(200)
        self.detector = FVGDetector(
            self.sample_data,
            min_gap_percentage=0.001,
            min_impulse_strength=0.015,
            volume_threshold_percentile=70
        )
    
    def test_fvg_detector_initialization(self):
        """Test FVGDetector initialization"""
        self.assertIsInstance(self.detector, FVGDetector)
        self.assertIsInstance(self.detector.ohlcv, pd.DataFrame)
        self.assertTrue(len(self.detector.ohlcv) > 0)
        self.assertIn('atr', self.detector.ohlcv.columns)
        self.assertIn('bullish_impulse', self.detector.ohlcv.columns)
        self.assertIn('bearish_impulse', self.detector.ohlcv.columns)
    
    def test_fvg_identification(self):
        """Test FVG identification functionality"""
        fvgs = self.detector.identify_fvgs()
        
        # Should return a list
        self.assertIsInstance(fvgs, list)
        
        # Check if any FVGs were detected
        if fvgs:
            # Test first FVG structure
            fvg = fvgs[0]
            self.assertIsInstance(fvg, FairValueGap)
            self.assertIn(fvg.type, [FVGType.BULLISH, FVGType.BEARISH])
            self.assertGreater(fvg.size, 0)
            self.assertGreater(fvg.percentage, 0)
            self.assertGreaterEqual(fvg.validation_score, 0)
            self.assertLessEqual(fvg.validation_score, 1)
    
    def test_fvg_fill_tracking(self):
        """Test FVG fill tracking functionality"""
        fvgs = self.detector.identify_fvgs()
        fill_stats = self.detector.track_fvg_fills()
        
        # Should return dictionary with expected keys
        expected_keys = [
            'total_fvgs', 'active_fvgs', 'partially_filled', 
            'fully_filled', 'expired', 'average_fill_time', 'fill_success_rate'
        ]
        
        for key in expected_keys:
            self.assertIn(key, fill_stats)
        
        # Values should be non-negative
        self.assertGreaterEqual(fill_stats['total_fvgs'], 0)
        self.assertGreaterEqual(fill_stats['active_fvgs'], 0)
        self.assertGreaterEqual(fill_stats['fill_success_rate'], 0)
        self.assertLessEqual(fill_stats['fill_success_rate'], 1)
    
    def test_fvg_statistics(self):
        """Test FVG statistics calculation"""
        fvgs = self.detector.identify_fvgs()
        stats = self.detector.get_fvg_statistics()
        
        # Should return dictionary with expected keys
        expected_keys = [
            'total_fvgs', 'bullish_fvgs', 'bearish_fvgs',
            'average_gap_size', 'average_validation_score', 'average_impulse_strength',
            'status_distribution', 'size_distribution', 'reaction_statistics'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Values should be consistent
        self.assertEqual(stats['total_fvgs'], len(fvgs))
        self.assertEqual(
            stats['bullish_fvgs'] + stats['bearish_fvgs'], 
            stats['total_fvgs']
        )
    
    def test_active_fvgs_filtering(self):
        """Test active FVGs filtering by price level"""
        fvgs = self.detector.identify_fvgs()
        
        if fvgs:
            # Test without price filter
            active_fvgs = self.detector.get_active_fvgs()
            self.assertIsInstance(active_fvgs, list)
            
            # Test with price filter
            test_price = self.sample_data['close'].iloc[-1]
            filtered_fvgs = self.detector.get_active_fvgs(test_price)
            self.assertIsInstance(filtered_fvgs, list)
            
            # Filtered list should be subset of all active FVGs
            self.assertLessEqual(len(filtered_fvgs), len(active_fvgs))
    
    def test_fvg_quality_validation(self):
        """Test FVG quality validation"""
        fvgs = self.detector.identify_fvgs()
        
        if fvgs:
            fvg = fvgs[0]
            validation = self.detector.validate_fvg_quality(fvg)
            
            # Should return dictionary with expected structure
            self.assertIn('overall_score', validation)
            self.assertIn('criteria', validation)
            self.assertIn('metrics', validation)
            self.assertIn('status_info', validation)
            self.assertIn('quality_level', validation)
            
            # Quality level should be valid
            self.assertIn(validation['quality_level'], ['high', 'medium', 'low'])
    
    def test_convenience_function(self):
        """Test the convenience function for getting enhanced FVGs"""
        enhanced_fvgs = get_enhanced_fvgs(self.sample_data)
        
        self.assertIsInstance(enhanced_fvgs, list)
        
        if enhanced_fvgs:
            fvg = enhanced_fvgs[0]
            # Should have all expected fields
            expected_fields = [
                'type', 'top', 'bottom', 'size', 'percentage', 'midpoint',
                'formation_index', 'impulse_strength', 'validation_score',
                'status', 'fill_percentage', 'touches', 'max_reaction'
            ]
            
            for field in expected_fields:
                self.assertIn(field, fvg)


class TestFairValueGapDataClass(unittest.TestCase):
    """Test cases for FairValueGap data class"""
    
    def setUp(self):
        """Set up test FVG"""
        self.fvg = FairValueGap(
            type=FVGType.BULLISH,
            top=50200.0,
            bottom=50000.0,
            size=200.0,
            percentage=0.004,
            formation_index=100,
            formation_timestamp=datetime.now(),
            candle_1_index=99,
            candle_2_index=100,
            candle_3_index=101,
            impulse_strength=0.025,
            volume_context={'average_volume': 1000000, 'above_threshold': True}
        )
    
    def test_fvg_creation(self):
        """Test FairValueGap creation and basic properties"""
        self.assertEqual(self.fvg.type, FVGType.BULLISH)
        self.assertEqual(self.fvg.top, 50200.0)
        self.assertEqual(self.fvg.bottom, 50000.0)
        self.assertEqual(self.fvg.size, 200.0)
        self.assertEqual(self.fvg.status, FVGStatus.ACTIVE)
    
    def test_midpoint_calculation(self):
        """Test get_midpoint method"""
        midpoint = self.fvg.get_midpoint()
        expected_midpoint = (50200.0 + 50000.0) / 2
        self.assertEqual(midpoint, expected_midpoint)
    
    def test_price_in_fvg_method(self):
        """Test is_price_in_fvg method"""
        # Price within FVG
        self.assertTrue(self.fvg.is_price_in_fvg(50100.0))
        
        # Price outside FVG
        self.assertFalse(self.fvg.is_price_in_fvg(49000.0))
        self.assertFalse(self.fvg.is_price_in_fvg(51000.0))
        
        # Price at boundaries
        self.assertTrue(self.fvg.is_price_in_fvg(50000.0))
        self.assertTrue(self.fvg.is_price_in_fvg(50200.0))
    
    def test_fill_percentage_calculation(self):
        """Test calculate_fill_percentage method"""
        # No fill
        fill_pct = self.fvg.calculate_fill_percentage(50300.0, 50250.0)
        self.assertEqual(fill_pct, 0.0)
        
        # Partial fill
        fill_pct = self.fvg.calculate_fill_percentage(50300.0, 50100.0)
        self.assertGreater(fill_pct, 0)
        self.assertLess(fill_pct, 100)
        
        # Full fill
        fill_pct = self.fvg.calculate_fill_percentage(50300.0, 49900.0)
        self.assertEqual(fill_pct, 100.0)


class TestSampleDataGeneration(unittest.TestCase):
    """Test cases for sample data generation"""
    
    def test_create_sample_fvg_data(self):
        """Test sample FVG data creation"""
        data = create_sample_fvg_data(100)
        
        # Should return DataFrame
        self.assertIsInstance(data, pd.DataFrame)
        
        # Should have correct number of rows (approximately, due to FVG creation logic)
        self.assertGreater(len(data), 50)  # Should have at least some data
        
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
                return {'fvg': [], 'order_blocks': []}
        
        # Apply the integration decorator
        EnhancedSMCDetector = integrate_fvg_detector_with_smc(MockSMCDetector)
        
        # Test enhanced detector
        sample_data = create_sample_fvg_data(100)
        enhanced_detector = EnhancedSMCDetector(sample_data)
        
        # Should have FVG detector
        self.assertIsInstance(enhanced_detector.fvg_detector, FVGDetector)
        
        # Should have enhanced methods
        self.assertTrue(hasattr(enhanced_detector, 'detect_fvg_enhanced'))
        self.assertTrue(hasattr(enhanced_detector, 'get_fvg_statistics'))
        self.assertTrue(hasattr(enhanced_detector, 'get_active_fvgs_near_price'))
        self.assertTrue(hasattr(enhanced_detector, 'detect_all_enhanced'))


if __name__ == '__main__':
    # Run the tests
    print("Running Fair Value Gap Detection System Tests...")
    print("=" * 60)
    
    # Run tests using unittest.main() for modern approach
    unittest.main(verbosity=2, exit=False)
