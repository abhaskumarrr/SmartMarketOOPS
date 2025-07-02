import unittest
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_trading_strategy import CoreTradingStrategy

def create_test_data(rows, trend='none', volatility=1.0):
    start_price = 42000
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=rows, freq='H'))
    
    if trend == 'up':
        price_path = start_price + np.arange(rows) * 10
    elif trend == 'down':
        price_path = start_price - np.arange(rows) * 10
    else:
        price_path = start_price + np.random.randn(rows).cumsum()

    data = pd.DataFrame({
        'timestamp': dates,
        'open': price_path + np.random.uniform(-5, 5, size=rows) * volatility,
        'high': price_path + np.random.uniform(5, 10, size=rows) * volatility,
        'low': price_path + np.random.uniform(-10, -5, size=rows) * volatility,
        'close': price_path + np.random.uniform(-5, 5, size=rows) * volatility,
        'volume': np.random.uniform(100, 200, size=rows)
    })
    # Ensure OHLC integrity
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    return data

class TestCoreTradingStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = CoreTradingStrategy()

    def test_detect_regime(self):
        print("\\n🧪 Testing Market Regime Detection...")
        # Test uptrend
        up_data = create_test_data(100, trend='up')
        self.assertEqual(self.strategy.detect_regime(up_data), 'trend_up', "Failed to detect uptrend.")
        print("  ✅ Uptrend detected correctly.")
        
        # Test downtrend
        down_data = create_test_data(100, trend='down')
        self.assertEqual(self.strategy.detect_regime(down_data), 'trend_down', "Failed to detect downtrend.")
        print("  ✅ Downtrend detected correctly.")

        # Test range
        range_data = create_test_data(100, trend='none')
        self.assertEqual(self.strategy.detect_regime(range_data, adx_threshold=50), 'range', "Failed to detect range.")
        print("  ✅ Ranging market detected correctly.")

    def test_get_extreme_zone_details(self):
        print("\\n🧪 Testing Extreme Zone Detection...")
        data = create_test_data(100, trend='none', volatility=5.0)
        
        # Force a 'top' zone
        top_value = data['close'].iloc[-21:].mean() + (data['close'].iloc[-21:].std() * 3)
        data.loc[data.index[-1], 'close'] = top_value
        top_details = self.strategy.get_extreme_zone_details(data.copy())
        self.assertEqual(top_details['zone'], 'top', "Failed to detect top zone.")
        self.assertGreaterEqual(top_details['percentile'], 0.95, "Percentile for top zone is incorrect.")
        print("  ✅ Top extreme zone detected correctly.")

        # Force a 'bottom' zone
        bottom_value = data['close'].iloc[-21:].mean() - (data['close'].iloc[-21:].std() * 3)
        data.loc[data.index[-1], 'close'] = bottom_value
        bottom_details = self.strategy.get_extreme_zone_details(data.copy())
        self.assertEqual(bottom_details['zone'], 'bottom', "Failed to detect bottom zone.")
        self.assertLessEqual(bottom_details['percentile'], 0.05, "Percentile for bottom zone is incorrect.")
        print("  ✅ Bottom extreme zone detected correctly.")

    def test_verify_micro_structure(self):
        print("\\n🧪 Testing Micro-Structure Verification (BoS)...")
        # Test Bullish Break of Structure
        bullish_data = create_test_data(20, trend='up')
        swing_high = bullish_data['high'].max() + 10
        bullish_data.loc[bullish_data.index[-5], 'high'] = swing_high # Create a recent swing high
        bullish_data.loc[bullish_data.index[-1], 'close'] = swing_high + 5 # Break it
        self.assertTrue(self.strategy.verify_micro_structure(bullish_data, 'bullish'), "Failed to verify bullish BoS.")
        print("  ✅ Bullish Break of Structure verified.")

        # Test Bearish Break of Structure
        bearish_data = create_test_data(20, trend='down')
        swing_low = bearish_data['low'].min() - 10
        bearish_data.loc[bearish_data.index[-5], 'low'] = swing_low # Create a recent swing low
        bearish_data.loc[bearish_data.index[-1], 'close'] = swing_low - 5 # Break it
        self.assertTrue(self.strategy.verify_micro_structure(bearish_data, 'bearish'), "Failed to verify bearish BoS.")
        print("  ✅ Bearish Break of Structure verified.")

    def test_full_signal_generation(self):
        print("\\n🧪 Testing Full Signal Generation Logic...")
        # Scenario: Perfect bullish setup
        mock_data = {
            'daily': create_test_data(100, trend='up'),
            'h4': create_test_data(100, trend='up'),
            'm15': create_test_data(20, trend='up')
        }
        # Force H4 into bottom zone
        bottom_zone_value = mock_data['h4']['close'].iloc[-21:].mean() - (mock_data['h4']['close'].iloc[-21:].std() * 3)
        mock_data['h4'].loc[mock_data['h4'].index[-1], 'close'] = bottom_zone_value
        # Force 15m BoS
        swing_high_m15 = mock_data['m15']['high'].max() + 10 
        mock_data['m15'].loc[mock_data['m15'].index[-5], 'high'] = swing_high_m15
        mock_data['m15'].loc[mock_data['m15'].index[-1], 'close'] = swing_high_m15 + 5

        signal = self.strategy.generate_signal(mock_data)
        self.assertEqual(signal.get('signal'), 'buy', "Failed to generate BUY signal on perfect bullish setup.")
        self.assertIn('confidence', signal)
        self.assertIn('stop_loss', signal)
        self.assertIn('target_price', signal)
        self.assertGreater(signal['confidence'], 50, "Confidence for bullish setup is too low.")
        self.assertLess(signal['stop_loss'], mock_data['m15']['close'].iloc[-1])
        self.assertGreater(signal['target_price'], mock_data['m15']['close'].iloc[-1])
        print("  ✅ BUY signal generated correctly on bullish setup with risk parameters.")

        # Scenario: Perfect bearish setup
        mock_data = {
            'daily': create_test_data(100, trend='down'),
            'h4': create_test_data(100, trend='down'),
            'm15': create_test_data(20, trend='down')
        }
        # Force H4 into top zone
        top_zone_value = mock_data['h4']['close'].iloc[-21:].mean() + (mock_data['h4']['close'].iloc[-21:].std() * 3)
        mock_data['h4'].loc[mock_data['h4'].index[-1], 'close'] = top_zone_value
        # Force 15m BoS
        swing_low_m15 = mock_data['m15']['low'].min() - 10 
        mock_data['m15'].loc[mock_data['m15'].index[-5], 'low'] = swing_low_m15
        mock_data['m15'].loc[mock_data['m15'].index[-1], 'close'] = swing_low_m15 - 5

        signal = self.strategy.generate_signal(mock_data)
        self.assertEqual(signal.get('signal'), 'sell', "Failed to generate SELL signal on perfect bearish setup.")
        self.assertIn('confidence', signal)
        self.assertIn('stop_loss', signal)
        self.assertIn('target_price', signal)
        self.assertGreater(signal['confidence'], 50, "Confidence for bearish setup is too low.")
        self.assertGreater(signal['stop_loss'], mock_data['m15']['close'].iloc[-1])
        self.assertLess(signal['target_price'], mock_data['m15']['close'].iloc[-1])
        print("  ✅ SELL signal generated correctly on bearish setup with risk parameters.")

        # Scenario: No trend
        mock_data['daily'] = create_test_data(100, trend='none')
        signal = self.strategy.generate_signal(mock_data)
        self.assertEqual(signal.get('signal'), 'hold', "Generated a signal in a ranging market.")
        self.assertNotIn('stop_loss', signal)
        self.assertNotIn('target_price', signal)
        print("  ✅ HOLD signal generated correctly on ranging setup.")

if __name__ == '__main__':
    unittest.main() 