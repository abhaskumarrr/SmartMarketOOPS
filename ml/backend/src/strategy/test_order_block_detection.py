#!/usr/bin/env python3
"""
Test suite for the Order Block Detection Engine

This module contains comprehensive tests for the enhanced Order Block Detection Engine
that implements Smart Money Concepts for institutional order block identification.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smc_detection import OrderBlockDetector, OrderBlock, SMCDector, create_sample_data, get_enhanced_order_blocks


class TestOrderBlockDetection(unittest.TestCase):
    """Test cases for Order Block Detection Engine"""

    def setUp(self):
        """Set up test data and detector instance"""
        # Create sample OHLCV data for testing
        self.sample_data = create_sample_data(200)
        self.detector = OrderBlockDetector(self.sample_data)
        self.smc_detector = SMCDector(self.sample_data)

    def test_order_block_detector_initialization(self):
        """Test OrderBlockDetector initialization"""
        self.assertIsInstance(self.detector, OrderBlockDetector)
        self.assertIsInstance(self.detector.ohlcv, pd.DataFrame)
        self.assertTrue(len(self.detector.ohlcv) > 0)
        self.assertIn('swing_high', self.detector.ohlcv.columns)
        self.assertIn('swing_low', self.detector.ohlcv.columns)
        self.assertIn('atr', self.detector.ohlcv.columns)

    def test_swing_detection(self):
        """Test swing high and low detection"""
        swing_highs = self.detector.ohlcv['swing_high'].sum()
        swing_lows = self.detector.ohlcv['swing_low'].sum()

        # Should detect some swing points in sample data
        self.assertGreater(swing_highs, 0, "Should detect at least one swing high")
        self.assertGreater(swing_lows, 0, "Should detect at least one swing low")

    def test_order_block_detection(self):
        """Test order block detection functionality"""
        order_blocks = self.detector.detect_order_blocks()

        # Should return a list
        self.assertIsInstance(order_blocks, list)

        # Check if any order blocks were detected
        if order_blocks:
            # Test first order block structure
            ob = order_blocks[0]
            self.assertIsInstance(ob, OrderBlock)
            self.assertIn(ob.type, ['bullish', 'bearish'])
            self.assertGreater(ob.strength, 0)
            self.assertGreaterEqual(ob.touches, 0)

    def test_order_block_validation(self):
        """Test order block validation logic"""
        order_blocks = self.detector.detect_order_blocks()

        for ob in order_blocks:
            if ob.is_valid:
                # Valid order blocks should meet basic criteria
                self.assertGreater(ob.volume, 0)
                self.assertGreater(ob.strength, 0)
                self.assertIsNotNone(ob.formation_context)

    def test_smc_detector_integration(self):
        """Test SMC detector integration with enhanced order blocks"""
        results = self.smc_detector.detect_all()

        # Should return dictionary with expected keys
        self.assertIn('order_blocks', results)
        self.assertIn('fvg', results)
        self.assertIn('liquidity_zones', results)

        # Order blocks should be in dictionary format
        order_blocks = results['order_blocks']
        self.assertIsInstance(order_blocks, list)

        if order_blocks:
            ob = order_blocks[0]
            self.assertIn('type', ob)
            self.assertIn('strength', ob)
            self.assertIn('price', ob)
            self.assertIn('volume', ob)

    def test_order_block_statistics(self):
        """Test order block statistics calculation"""
        stats = self.smc_detector.get_order_block_statistics()

        # Should return dictionary with expected keys
        expected_keys = [
            'total_detected', 'valid_blocks', 'bullish_blocks',
            'bearish_blocks', 'average_strength', 'average_touches',
            'strength_distribution'
        ]

        for key in expected_keys:
            self.assertIn(key, stats)

        # Values should be non-negative
        self.assertGreaterEqual(stats['total_detected'], 0)
        self.assertGreaterEqual(stats['valid_blocks'], 0)
        self.assertGreaterEqual(stats['average_strength'], 0)

    def test_convenience_function(self):
        """Test the convenience function for getting enhanced order blocks"""
        order_blocks = get_enhanced_order_blocks(self.sample_data)

        self.assertIsInstance(order_blocks, list)

        if order_blocks:
            ob = order_blocks[0]
            # Should have all expected fields
            expected_fields = [
                'type', 'price', 'top', 'bottom', 'volume',
                'strength', 'timestamp', 'touches', 'is_valid'
            ]

            for field in expected_fields:
                self.assertIn(field, ob)


class TestOrderBlockDataClass(unittest.TestCase):
    """Test cases for OrderBlock data class"""

    def setUp(self):
        """Set up test order block"""
        self.order_block = OrderBlock(
            type='bullish',
            top=50100.0,
            bottom=50000.0,
            high=50150.0,
            low=49950.0,
            volume=1000000.0,
            timestamp=datetime.now(),
            index=100,
            strength=0.75,
            formation_context={'test': True}
        )

    def test_order_block_creation(self):
        """Test OrderBlock creation and basic properties"""
        self.assertEqual(self.order_block.type, 'bullish')
        self.assertEqual(self.order_block.top, 50100.0)
        self.assertEqual(self.order_block.bottom, 50000.0)
        self.assertEqual(self.order_block.strength, 0.75)
        self.assertTrue(self.order_block.is_valid)

    def test_price_range_method(self):
        """Test get_price_range method"""
        bottom, top = self.order_block.get_price_range()
        self.assertEqual(bottom, 50000.0)
        self.assertEqual(top, 50100.0)

    def test_price_in_block_method(self):
        """Test is_price_in_block method"""
        # Price within block
        self.assertTrue(self.order_block.is_price_in_block(50050.0))

        # Price outside block
        self.assertFalse(self.order_block.is_price_in_block(51000.0))
        self.assertFalse(self.order_block.is_price_in_block(49000.0))

        # Price at edges (with tolerance)
        self.assertTrue(self.order_block.is_price_in_block(50000.0))
        self.assertTrue(self.order_block.is_price_in_block(50100.0))


class TestSampleDataGeneration(unittest.TestCase):
    """Test cases for sample data generation"""

    def test_create_sample_data(self):
        """Test sample data creation"""
        data = create_sample_data(100)

        # Should return DataFrame
        self.assertIsInstance(data, pd.DataFrame)

        # Should have correct number of rows
        self.assertEqual(len(data), 100)

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


if __name__ == '__main__':
    # Run the tests
    print("Running Order Block Detection Engine Tests...")
    print("=" * 60)

    # Run tests using unittest.main() for modern approach
    unittest.main(verbosity=2, exit=False)
