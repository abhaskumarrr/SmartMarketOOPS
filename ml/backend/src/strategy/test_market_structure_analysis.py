#!/usr/bin/env python3
"""
Test suite for the Market Structure Analysis System

This module contains comprehensive tests for the Market Structure Analysis System
that implements Break of Structure (BOS) and Change of Character (ChoCH) detection
for institutional trading analysis.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_structure_analysis import (
    MarketStructureAnalyzer, SwingPoint, StructureBreak, MarketStructure,
    StructureType, TrendDirection, StructureStrength,
    create_sample_market_structure_data, get_enhanced_market_structure
)


class TestMarketStructureAnalysis(unittest.TestCase):
    """Test cases for Market Structure Analysis System"""
    
    def setUp(self):
        """Set up test data and analyzer instance"""
        # Create sample OHLCV data with market structure patterns
        self.sample_data = create_sample_market_structure_data(300)
        self.analyzer = MarketStructureAnalyzer(
            self.sample_data,
            swing_lookback=5,
            min_structure_distance=0.005,
            volume_threshold=1.2
        )
    
    def test_analyzer_initialization(self):
        """Test MarketStructureAnalyzer initialization"""
        self.assertIsInstance(self.analyzer, MarketStructureAnalyzer)
        self.assertIsInstance(self.analyzer.ohlcv, pd.DataFrame)
        self.assertTrue(len(self.analyzer.ohlcv) > 0)
        self.assertIn('volume_ratio', self.analyzer.ohlcv.columns)
        self.assertIn('atr', self.analyzer.ohlcv.columns)
        self.assertIn('rsi', self.analyzer.ohlcv.columns)
        self.assertIn('ema_fast', self.analyzer.ohlcv.columns)
        self.assertIn('ema_slow', self.analyzer.ohlcv.columns)
    
    def test_swing_point_detection(self):
        """Test swing point detection functionality"""
        swing_points = self.analyzer.detect_swing_points()
        
        # Should return a list
        self.assertIsInstance(swing_points, list)
        
        # Check if any swing points were detected
        if swing_points:
            # Test first swing point structure
            point = swing_points[0]
            self.assertIsInstance(point, SwingPoint)
            self.assertIn(point.type, [StructureType.SWING_HIGH, StructureType.SWING_LOW])
            self.assertGreater(point.strength, 0)
            self.assertLessEqual(point.strength, 1)
            self.assertIsInstance(point.confirmed, bool)
    
    def test_structure_break_detection(self):
        """Test BOS and ChoCH detection functionality"""
        # First detect swing points
        self.analyzer.detect_swing_points()
        
        # Then detect structure breaks
        structure_breaks = self.analyzer.detect_structure_breaks()
        
        # Should return a list
        self.assertIsInstance(structure_breaks, list)
        
        # Check if any structure breaks were detected
        if structure_breaks:
            # Test first structure break
            break_event = structure_breaks[0]
            self.assertIsInstance(break_event, StructureBreak)
            self.assertIn(break_event.type, [StructureType.BOS, StructureType.CHOCH])
            self.assertIn(break_event.direction, [TrendDirection.BULLISH, TrendDirection.BEARISH])
            self.assertIn(break_event.strength, [
                StructureStrength.WEAK, StructureStrength.MODERATE,
                StructureStrength.STRONG, StructureStrength.VERY_STRONG
            ])
            self.assertGreaterEqual(break_event.volume_confirmation, 0)
            self.assertGreaterEqual(break_event.impulse_strength, 0)
            self.assertLessEqual(break_event.impulse_strength, 1)
            self.assertGreaterEqual(break_event.institutional_signature, 0)
            self.assertLessEqual(break_event.institutional_signature, 1)
    
    def test_comprehensive_market_structure_analysis(self):
        """Test comprehensive market structure analysis"""
        market_structure = self.analyzer.analyze_market_structure()
        
        # Should return MarketStructure object
        self.assertIsInstance(market_structure, MarketStructure)
        
        # Check structure properties
        self.assertIn(market_structure.current_trend, [
            TrendDirection.BULLISH, TrendDirection.BEARISH, 
            TrendDirection.NEUTRAL, TrendDirection.TRANSITIONING
        ])
        self.assertGreaterEqual(market_structure.trend_strength, 0)
        self.assertLessEqual(market_structure.trend_strength, 1)
        self.assertGreaterEqual(market_structure.structure_quality, 0)
        self.assertLessEqual(market_structure.structure_quality, 1)
        self.assertIsInstance(market_structure.active_swing_highs, list)
        self.assertIsInstance(market_structure.active_swing_lows, list)
        self.assertIsInstance(market_structure.structure_breaks, list)
    
    def test_structure_statistics(self):
        """Test structure statistics calculation"""
        # Analyze structure first
        self.analyzer.analyze_market_structure()
        stats = self.analyzer.get_structure_statistics()
        
        # Should return dictionary with expected keys
        expected_keys = [
            'total_swing_points', 'swing_highs', 'swing_lows', 'confirmed_swings',
            'total_structure_breaks', 'bos_breaks', 'choch_breaks',
            'bullish_breaks', 'bearish_breaks', 'current_trend',
            'trend_strength', 'structure_quality', 'strength_distribution',
            'average_institutional_signature', 'follow_through_rate'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Values should be non-negative
        self.assertGreaterEqual(stats['total_swing_points'], 0)
        self.assertGreaterEqual(stats['total_structure_breaks'], 0)
        self.assertGreaterEqual(stats['trend_strength'], 0)
        self.assertLessEqual(stats['trend_strength'], 1)
        self.assertGreaterEqual(stats['structure_quality'], 0)
        self.assertLessEqual(stats['structure_quality'], 1)
        self.assertGreaterEqual(stats['follow_through_rate'], 0)
        self.assertLessEqual(stats['follow_through_rate'], 1)
    
    def test_swing_point_methods(self):
        """Test SwingPoint class methods"""
        # Create test swing points
        high_point = SwingPoint(
            type=StructureType.SWING_HIGH,
            price=50000.0,
            timestamp=datetime.now(),
            index=100,
            volume=1000000,
            strength=0.8,
            confirmed=True
        )
        
        low_point = SwingPoint(
            type=StructureType.SWING_LOW,
            price=49000.0,
            timestamp=datetime.now(),
            index=105,
            volume=1200000,
            strength=0.7,
            confirmed=False
        )
        
        # Test methods
        self.assertTrue(high_point.is_swing_high())
        self.assertFalse(high_point.is_swing_low())
        self.assertFalse(low_point.is_swing_high())
        self.assertTrue(low_point.is_swing_low())
    
    def test_structure_break_methods(self):
        """Test StructureBreak class methods"""
        # Create test swing point
        test_swing = SwingPoint(
            type=StructureType.SWING_HIGH,
            price=50000.0,
            timestamp=datetime.now(),
            index=100,
            volume=1000000
        )
        
        # Create test structure breaks
        bos_break = StructureBreak(
            type=StructureType.BOS,
            direction=TrendDirection.BULLISH,
            break_price=50500.0,
            break_timestamp=datetime.now(),
            break_index=110,
            previous_structure=test_swing,
            volume_confirmation=1.5,
            strength=StructureStrength.STRONG,
            impulse_strength=0.7,
            retracement_depth=0.3,
            follow_through=True,
            institutional_signature=0.8
        )
        
        choch_break = StructureBreak(
            type=StructureType.CHOCH,
            direction=TrendDirection.BEARISH,
            break_price=49500.0,
            break_timestamp=datetime.now(),
            break_index=115,
            previous_structure=test_swing,
            volume_confirmation=2.0,
            strength=StructureStrength.VERY_STRONG,
            impulse_strength=0.9,
            retracement_depth=0.5,
            follow_through=False,
            institutional_signature=0.9
        )
        
        # Test methods
        self.assertTrue(bos_break.is_bos())
        self.assertFalse(bos_break.is_choch())
        self.assertFalse(choch_break.is_bos())
        self.assertTrue(choch_break.is_choch())
    
    def test_trend_direction_determination(self):
        """Test trend direction determination"""
        # Analyze structure first
        self.analyzer.analyze_market_structure()
        
        # Test trend determination with different indices
        for i in range(50, min(100, len(self.analyzer.ohlcv))):
            trend = self.analyzer._get_current_trend(i)
            self.assertIn(trend, [
                TrendDirection.BULLISH, TrendDirection.BEARISH, TrendDirection.NEUTRAL
            ])
    
    def test_volume_confirmation_calculation(self):
        """Test volume confirmation calculation"""
        # Test with different indices
        for i in range(20, min(50, len(self.analyzer.ohlcv))):
            volume_conf = self.analyzer._calculate_volume_confirmation(i)
            self.assertGreaterEqual(volume_conf, 0)
            self.assertIsInstance(volume_conf, float)
    
    def test_impulse_strength_calculation(self):
        """Test impulse strength calculation"""
        # Test with different indices and directions
        for i in range(10, min(30, len(self.analyzer.ohlcv))):
            bullish_impulse = self.analyzer._calculate_impulse_strength(i, TrendDirection.BULLISH)
            bearish_impulse = self.analyzer._calculate_impulse_strength(i, TrendDirection.BEARISH)
            
            self.assertGreaterEqual(bullish_impulse, 0)
            self.assertLessEqual(bullish_impulse, 1)
            self.assertGreaterEqual(bearish_impulse, 0)
            self.assertLessEqual(bearish_impulse, 1)
    
    def test_convenience_function(self):
        """Test convenience function for getting enhanced market structure"""
        enhanced_analysis = get_enhanced_market_structure(self.sample_data)
        
        # Should return dictionary with expected structure
        self.assertIsInstance(enhanced_analysis, dict)
        self.assertIn('market_structure', enhanced_analysis)
        self.assertIn('swing_points', enhanced_analysis)
        self.assertIn('structure_breaks', enhanced_analysis)
        self.assertIn('statistics', enhanced_analysis)
        
        # Check market structure section
        ms = enhanced_analysis['market_structure']
        self.assertIn('current_trend', ms)
        self.assertIn('trend_strength', ms)
        self.assertIn('structure_quality', ms)
        
        # Check swing points format
        swing_points = enhanced_analysis['swing_points']
        self.assertIsInstance(swing_points, list)
        
        if swing_points:
            sp = swing_points[0]
            expected_fields = ['type', 'price', 'timestamp', 'index', 'volume', 'strength', 'confirmed']
            for field in expected_fields:
                self.assertIn(field, sp)
        
        # Check structure breaks format
        structure_breaks = enhanced_analysis['structure_breaks']
        self.assertIsInstance(structure_breaks, list)
        
        if structure_breaks:
            sb = structure_breaks[0]
            expected_fields = [
                'type', 'direction', 'break_price', 'break_timestamp', 'break_index',
                'volume_confirmation', 'strength', 'impulse_strength', 'retracement_depth',
                'follow_through', 'institutional_signature'
            ]
            for field in expected_fields:
                self.assertIn(field, sb)


class TestSampleDataGeneration(unittest.TestCase):
    """Test cases for sample data generation"""
    
    def test_create_sample_market_structure_data(self):
        """Test sample market structure data creation"""
        data = create_sample_market_structure_data(200)
        
        # Should return DataFrame
        self.assertIsInstance(data, pd.DataFrame)
        
        # Should have correct number of rows
        self.assertGreater(len(data), 100)
        
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
        
        # Should have realistic price movements
        price_changes = data['close'].pct_change().dropna()
        self.assertLess(abs(price_changes.mean()), 0.1)  # Average change should be reasonable


if __name__ == '__main__':
    # Run the tests
    print("Running Market Structure Analysis System Tests...")
    print("=" * 60)
    
    # Run tests using unittest.main() for modern approach
    unittest.main(verbosity=2, exit=False)
