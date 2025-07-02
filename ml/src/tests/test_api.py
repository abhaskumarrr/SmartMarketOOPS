import unittest
import pandas as pd
from fastapi.testclient import TestClient
import sys
import os
import numpy as np

# Adjust path to import from the root 'ml' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api.app import app

def create_test_data(rows, trend='none', volatility=1.0):
    start_price = 42000
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=rows, freq='H'))
    
    if trend == 'up':
        price_path = start_price + np.arange(rows) * 10
    elif trend == 'down':
        price_path = start_price - np.arange(rows) * 10
    else:
        price_path = start_price + np.random.randn(rows).cumsum()

    df = pd.DataFrame({
        'timestamp': dates,
        'open': price_path + np.random.uniform(-5, 5, size=rows) * volatility,
        'high': price_path + np.random.uniform(5, 10, size=rows) * volatility,
        'low': price_path + np.random.uniform(-10, -5, size=rows) * volatility,
        'close': price_path + np.random.uniform(-5, 5, size=rows) * volatility,
        'volume': np.random.uniform(100, 200, size=rows)
    })
    # Ensure OHLC integrity
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    # The API expects dicts of lists, not dataframes
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    return df.to_dict(orient='list')

class TestCoreApi(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
        print("\\n🧪 Setting up API tests...")

    def test_predict_endpoint_buy_signal(self):
        """Test the /strategy/predict endpoint for a successful BUY signal."""
        print("   - Testing for successful BUY signal...")
        
        # Create a perfect bullish setup, similar to the unit test
        mock_data = {
            'daily': create_test_data(100, trend='up'),
            'h4': create_test_data(100, trend='up'),
            'm15': create_test_data(20, trend='up')
        }
        
        # Convert back to DataFrame to manipulate for the test scenario
        h4_df = pd.DataFrame.from_dict(mock_data['h4'])
        m15_df = pd.DataFrame.from_dict(mock_data['m15'])

        # Force H4 into bottom zone
        bottom_zone_value = h4_df['close'].iloc[-21:].mean() - (h4_df['close'].iloc[-21:].std() * 3)
        h4_df.loc[h4_df.index[-1], 'close'] = bottom_zone_value
        
        # Force 15m BoS
        swing_high_m15 = m15_df['high'].max() + 10 
        m15_df.loc[m15_df.index[-5], 'high'] = swing_high_m15
        m15_df.loc[m15_df.index[-1], 'close'] = swing_high_m15 + 5
        
        # Convert manipulated data back to dict for API
        mock_data['h4'] = h4_df.to_dict(orient='list')
        mock_data['m15'] = m15_df.to_dict(orient='list')

        response = self.client.post("/strategy/predict", json=mock_data)
        
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json['signal'], 'buy')
        self.assertIn('confidence', response_json)
        self.assertIn('stop_loss', response_json)
        self.assertIn('target_price', response_json)
        self.assertIsInstance(response_json['confidence'], float)
        self.assertIsInstance(response_json['stop_loss'], float)
        self.assertIsInstance(response_json['target_price'], float)
        print("     ✅ BUY signal success test passed.")

    def test_predict_endpoint_hold_signal(self):
        """Test the /strategy/predict endpoint for a HOLD signal."""
        print("   - Testing for HOLD signal...")
        
        # Create a ranging market scenario
        mock_data = {
            'daily': create_test_data(100, trend='none'),
            'h4': create_test_data(100, trend='none'),
            'm15': create_test_data(20, trend='none')
        }

        response = self.client.post("/strategy/predict", json=mock_data)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(response_json['signal'], 'hold')
        self.assertIn('confidence', response_json)
        # For HOLD signals, stop_loss and target_price should be None
        self.assertIsNone(response_json['stop_loss'])
        self.assertIsNone(response_json['target_price'])
        print("     ✅ HOLD signal success test passed.")


    def test_predict_endpoint_missing_data(self):
        """Test the /strategy/predict endpoint with incomplete data."""
        print("   - Testing for graceful failure on missing data...")
        test_data = {
            'daily': create_test_data(100),
            'h4': create_test_data(100)
            # m15 is missing
        }
        
        response = self.client.post("/strategy/predict", json=test_data)
        
        # FastAPI's Pydantic validation should catch this and return a 422
        self.assertEqual(response.status_code, 422)
        print("     ✅ Missing data test passed.")

    def test_root_endpoint(self):
        """Test the root endpoint for a health check."""
        print("   - Testing root health check endpoint...")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "message": "Core Trading Strategy API is running."})
        print("     ✅ Root endpoint test passed.")

if __name__ == '__main__':
    unittest.main() 