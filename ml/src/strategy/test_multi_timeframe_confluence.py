#!/usr/bin/env python3
"""
Test suite for the Multi-Timeframe Confluence System

This module contains comprehensive tests for the Multi-Timeframe Confluence System
that implements institutional-grade multi-timeframe analysis with HTF bias,
discount/premium zones, and confluence scoring.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_timeframe_confluence import (
    MultiTimeframeAnalyzer, TimeframeBias, DiscountPremiumZone, ConfluenceSignal,
    TimeframeType, BiasDirection, ZoneType, ConfluenceStrength,
    create_sample_multi_timeframe_data, get_enhanced_multi_timeframe_analysis
)


class TestMultiTimeframeConfluence(unittest.TestCase):
    """Test cases for Multi-Timeframe Confluence System"""
    
    def setUp(self):
        """Set up test data and analyzer instance"""
        # Create sample multi-timeframe data
        self.data_sources = create_sample_multi_timeframe_data()
        self.analyzer = MultiTimeframeAnalyzer(
            data_sources=self.data_sources,
            primary_timeframe=TimeframeType.M15,
            htf_timeframes=[TimeframeType.H4, TimeframeType.H1]
        )
    
    def test_analyzer_initialization(self):
        """Test MultiTimeframeAnalyzer initialization"""
        self.assertIsInstance(self.analyzer, MultiTimeframeAnalyzer)
        self.assertEqual(self.analyzer.primary_timeframe, TimeframeType.M15)
        self.assertIn(TimeframeType.H4, self.analyzer.htf_timeframes)
        self.assertIn(TimeframeType.H1, self.analyzer.htf_timeframes)
        self.assertIsInstance(self.analyzer.data_sources, dict)
        self.assertTrue(len(self.analyzer.data_sources) > 0)
    
    def test_data_validation(self):
        """Test data source validation"""
        # Test with valid data
        self.analyzer._validate_data_sources()
        
        # Test with invalid data (missing columns)
        invalid_data = {
            TimeframeType.M15: pd.DataFrame({'invalid': [1, 2, 3]})
        }
        
        with self.assertRaises(ValueError):
            MultiTimeframeAnalyzer(invalid_data)
    
    def test_htf_bias_establishment(self):
        """Test higher timeframe bias establishment"""
        biases = self.analyzer.establish_htf_bias()
        
        # Should return dictionary
        self.assertIsInstance(biases, dict)
        
        # Should have biases for HTF timeframes
        for timeframe in self.analyzer.htf_timeframes:
            if timeframe in self.data_sources:
                self.assertIn(timeframe, biases)
                bias = biases[timeframe]
                self.assertIsInstance(bias, TimeframeBias)
                self.assertIn(bias.direction, [BiasDirection.BULLISH, BiasDirection.BEARISH, BiasDirection.NEUTRAL])
                self.assertGreaterEqual(bias.strength, 0)
                self.assertLessEqual(bias.strength, 1)
                self.assertGreaterEqual(bias.confidence, 0)
                self.assertLessEqual(bias.confidence, 1)
                self.assertIsInstance(bias.key_levels, list)
    
    def test_discount_premium_zones(self):
        """Test discount/premium zone identification"""
        zones = self.analyzer.identify_discount_premium_zones()
        
        # Should return dictionary
        self.assertIsInstance(zones, dict)
        
        # Should have zones for all timeframes
        for timeframe in self.data_sources.keys():
            self.assertIn(timeframe, zones)
            zone = zones[timeframe]
            self.assertIsInstance(zone, DiscountPremiumZone)
            self.assertIn(zone.zone_type, [
                ZoneType.DISCOUNT, ZoneType.PREMIUM, ZoneType.EQUILIBRIUM,
                ZoneType.EXTREME_DISCOUNT, ZoneType.EXTREME_PREMIUM
            ])
            self.assertGreaterEqual(zone.current_position, 0)
            self.assertLessEqual(zone.current_position, 1)
            self.assertGreaterEqual(zone.strength, 0)
            self.assertLessEqual(zone.strength, 1)
            self.assertIsInstance(zone.key_levels, list)
    
    def test_confluence_score_calculation(self):
        """Test confluence score calculation"""
        # Test buy signal
        buy_signal = self.analyzer.calculate_confluence_score('buy')
        self.assertIsInstance(buy_signal, ConfluenceSignal)
        self.assertEqual(buy_signal.signal_type, 'buy')
        self.assertGreaterEqual(buy_signal.confluence_score, 0)
        self.assertLessEqual(buy_signal.confluence_score, 1)
        self.assertIn(buy_signal.strength, [
            ConfluenceStrength.WEAK, ConfluenceStrength.MODERATE,
            ConfluenceStrength.STRONG, ConfluenceStrength.VERY_STRONG,
            ConfluenceStrength.EXTREME
        ])
        self.assertIsInstance(buy_signal.timeframe_alignment, dict)
        self.assertIsInstance(buy_signal.key_levels, list)
        self.assertGreaterEqual(buy_signal.risk_reward_ratio, 0)
        
        # Test sell signal
        sell_signal = self.analyzer.calculate_confluence_score('sell')
        self.assertIsInstance(sell_signal, ConfluenceSignal)
        self.assertEqual(sell_signal.signal_type, 'sell')
        self.assertGreaterEqual(sell_signal.confluence_score, 0)
        self.assertLessEqual(sell_signal.confluence_score, 1)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive multi-timeframe analysis"""
        analysis = self.analyzer.get_comprehensive_analysis()
        
        # Should return dictionary with expected structure
        self.assertIsInstance(analysis, dict)
        self.assertIn('timestamp', analysis)
        self.assertIn('primary_timeframe', analysis)
        self.assertIn('htf_biases', analysis)
        self.assertIn('discount_premium_zones', analysis)
        self.assertIn('confluence_signals', analysis)
        self.assertIn('best_signal', analysis)
        self.assertIn('market_timing_score', analysis)
        self.assertIn('smc_available', analysis)
        
        # Check confluence signals structure
        signals = analysis['confluence_signals']
        self.assertIn('buy', signals)
        self.assertIn('sell', signals)
        
        for signal_type in ['buy', 'sell']:
            signal = signals[signal_type]
            self.assertIn('score', signal)
            self.assertIn('strength', signal)
            self.assertIn('timeframe_alignment', signal)
            self.assertIn('risk_reward_ratio', signal)
            self.assertIn('key_levels', signal)
        
        # Check best signal structure
        best_signal = analysis['best_signal']
        self.assertIn('type', best_signal)
        self.assertIn('score', best_signal)
        self.assertIn('strength', best_signal)
        self.assertIn('entry_zone', best_signal)
        self.assertIn('risk_reward_ratio', best_signal)
    
    def test_confluence_statistics(self):
        """Test confluence statistics calculation"""
        # First run analysis to populate data
        self.analyzer.get_comprehensive_analysis()
        
        stats = self.analyzer.get_confluence_statistics()
        
        # Should return dictionary with expected keys
        expected_keys = [
            'total_timeframes', 'htf_biases_established', 'zones_identified',
            'average_bias_strength', 'average_zone_strength', 'bullish_timeframes',
            'bearish_timeframes', 'neutral_timeframes', 'zone_distribution'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Values should be non-negative
        self.assertGreaterEqual(stats['total_timeframes'], 0)
        self.assertGreaterEqual(stats['htf_biases_established'], 0)
        self.assertGreaterEqual(stats['zones_identified'], 0)
        self.assertGreaterEqual(stats['average_bias_strength'], 0)
        self.assertLessEqual(stats['average_bias_strength'], 1)
        self.assertGreaterEqual(stats['average_zone_strength'], 0)
        self.assertLessEqual(stats['average_zone_strength'], 1)
    
    def test_timeframe_bias_methods(self):
        """Test TimeframeBias class methods"""
        bias = TimeframeBias(
            timeframe=TimeframeType.H4,
            direction=BiasDirection.BULLISH,
            strength=0.8,
            confidence=0.7,
            key_levels=[50000, 51000, 52000],
            trend_structure={},
            last_update=datetime.now()
        )
        
        self.assertTrue(bias.is_bullish())
        self.assertFalse(bias.is_bearish())
        
        bearish_bias = TimeframeBias(
            timeframe=TimeframeType.H4,
            direction=BiasDirection.BEARISH,
            strength=0.6,
            confidence=0.8,
            key_levels=[],
            trend_structure={},
            last_update=datetime.now()
        )
        
        self.assertFalse(bearish_bias.is_bullish())
        self.assertTrue(bearish_bias.is_bearish())
    
    def test_discount_premium_zone_methods(self):
        """Test DiscountPremiumZone class methods"""
        discount_zone = DiscountPremiumZone(
            zone_type=ZoneType.DISCOUNT,
            price_range=(49000, 51000),
            current_position=0.3,
            timeframe=TimeframeType.M15,
            strength=0.7,
            key_levels=[49500, 50000, 50500],
            formation_timestamp=datetime.now()
        )
        
        self.assertTrue(discount_zone.is_discount())
        self.assertFalse(discount_zone.is_premium())
        
        premium_zone = DiscountPremiumZone(
            zone_type=ZoneType.PREMIUM,
            price_range=(49000, 51000),
            current_position=0.8,
            timeframe=TimeframeType.M15,
            strength=0.6,
            key_levels=[],
            formation_timestamp=datetime.now()
        )
        
        self.assertFalse(premium_zone.is_discount())
        self.assertTrue(premium_zone.is_premium())
    
    def test_confluence_signal_methods(self):
        """Test ConfluenceSignal class methods"""
        test_zone = DiscountPremiumZone(
            zone_type=ZoneType.DISCOUNT,
            price_range=(49000, 51000),
            current_position=0.3,
            timeframe=TimeframeType.M15,
            strength=0.7,
            key_levels=[],
            formation_timestamp=datetime.now()
        )
        
        buy_signal = ConfluenceSignal(
            signal_type='buy',
            confluence_score=0.75,
            strength=ConfluenceStrength.STRONG,
            timeframe_alignment={},
            smc_confluence={},
            technical_confluence={},
            market_timing_score=0.8,
            entry_zone=test_zone,
            key_levels=[],
            risk_reward_ratio=2.5,
            timestamp=datetime.now()
        )
        
        self.assertTrue(buy_signal.is_buy_signal())
        self.assertFalse(buy_signal.is_sell_signal())
        
        sell_signal = ConfluenceSignal(
            signal_type='sell',
            confluence_score=0.65,
            strength=ConfluenceStrength.MODERATE,
            timeframe_alignment={},
            smc_confluence={},
            technical_confluence={},
            market_timing_score=0.7,
            entry_zone=test_zone,
            key_levels=[],
            risk_reward_ratio=1.8,
            timestamp=datetime.now()
        )
        
        self.assertFalse(sell_signal.is_buy_signal())
        self.assertTrue(sell_signal.is_sell_signal())
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        sample_prices = pd.Series([50000, 50100, 49900, 50200, 50050, 50300, 49800, 50150])
        
        # Test RSI calculation
        rsi = self.analyzer._calculate_rsi(sample_prices)
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(sample_prices))
        
        # Test MACD calculation
        macd, macd_signal = self.analyzer._calculate_macd(sample_prices)
        self.assertIsInstance(macd, pd.Series)
        self.assertIsInstance(macd_signal, pd.Series)
        self.assertEqual(len(macd), len(sample_prices))
        self.assertEqual(len(macd_signal), len(sample_prices))
        
        # Test Bollinger Bands calculation
        bb_upper, bb_lower = self.analyzer._calculate_bollinger_bands(sample_prices)
        self.assertIsInstance(bb_upper, pd.Series)
        self.assertIsInstance(bb_lower, pd.Series)
        self.assertEqual(len(bb_upper), len(sample_prices))
        self.assertEqual(len(bb_lower), len(sample_prices))
    
    def test_convenience_function(self):
        """Test convenience function for enhanced analysis"""
        enhanced_analysis = get_enhanced_multi_timeframe_analysis(self.data_sources)
        
        # Should return dictionary with expected structure
        self.assertIsInstance(enhanced_analysis, dict)
        self.assertIn('statistics', enhanced_analysis)
        self.assertIn('best_signal', enhanced_analysis)
        self.assertIn('confluence_signals', enhanced_analysis)
        self.assertIn('htf_biases', enhanced_analysis)
        self.assertIn('discount_premium_zones', enhanced_analysis)


class TestSampleDataGeneration(unittest.TestCase):
    """Test cases for sample data generation"""
    
    def test_create_sample_multi_timeframe_data(self):
        """Test sample multi-timeframe data creation"""
        data_sources = create_sample_multi_timeframe_data()
        
        # Should return dictionary
        self.assertIsInstance(data_sources, dict)
        
        # Should have multiple timeframes
        self.assertGreater(len(data_sources), 1)
        
        # Each timeframe should have valid DataFrame
        for timeframe, df in data_sources.items():
            self.assertIsInstance(timeframe, TimeframeType)
            self.assertIsInstance(df, pd.DataFrame)
            
            # Should have required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                self.assertIn(col, df.columns)
            
            # Should have reasonable amount of data
            self.assertGreater(len(df), 10)
            
            # OHLC relationships should be valid
            for _, row in df.iterrows():
                self.assertGreaterEqual(row['high'], row['open'])
                self.assertGreaterEqual(row['high'], row['close'])
                self.assertLessEqual(row['low'], row['open'])
                self.assertLessEqual(row['low'], row['close'])
                self.assertGreater(row['volume'], 0)


if __name__ == '__main__':
    # Run the tests
    print("Running Multi-Timeframe Confluence System Tests...")
    print("=" * 60)
    
    # Run tests using unittest.main() for modern approach
    unittest.main(verbosity=2, exit=False)
