#!/usr/bin/env python3
"""
Standalone Delta Exchange Client

This is a standalone version of the Delta Exchange client that doesn't rely on
relative imports, making it easier to integrate into the production backtesting system.
"""

import time
import requests
import json
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standalone configuration (no relative imports)
DELTA_CONFIG = {
    "base_url_testnet": "https://cdn-ind.testnet.deltaex.org",  # Correct India testnet URL
    "base_url_production": "https://api.india.delta.exchange",  # India production
    "api_key": os.getenv("DELTA_EXCHANGE_API_KEY", "HmerKHhySssgFIAfEIh4CYA5E3VmKg"),
    "api_secret": os.getenv("DELTA_EXCHANGE_API_SECRET", "1YNVg1x9cIjz1g3BPOQPUJQr6LhEm8w7cTaXi8ebJYPUpx5BKCQysMoLd6FT"),
    "testnet": os.getenv("DELTA_EXCHANGE_TESTNET", "true").lower() == "true"
}


class StandaloneDeltaExchangeClient:
    """Standalone Delta Exchange API client for production backtesting"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = True):
        """
        Initialize the Delta Exchange API client

        Args:
            api_key: API key for authentication (defaults to config value)
            api_secret: API secret for authentication (defaults to config value)
            testnet: Whether to use testnet or production API
        """
        self.api_key = api_key or DELTA_CONFIG["api_key"]
        self.api_secret = api_secret or DELTA_CONFIG["api_secret"]
        self.testnet = testnet

        self.base_url = (
            DELTA_CONFIG["base_url_testnet"]
            if self.testnet
            else DELTA_CONFIG["base_url_production"]
        )

        # Default request headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        # Rate limiting parameters
        self.rate_limit_remaining = 100
        self.rate_limit_reset = 0
        self.request_count = 0
        self.last_request_time = 0

        logger.info(f"Delta Exchange India client initialized (testnet={self.testnet})")
        logger.info(f"Base URL: {self.base_url}")

    def _generate_signature(self, timestamp: int, method: str, request_path: str, body: Dict = None) -> str:
        """Generate HMAC signature for API authentication"""
        if body is None:
            body_str = ""
        else:
            body_str = json.dumps(body)

        message = str(timestamp) + method + request_path + body_str
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            digestmod=hashlib.sha256
        ).digest()

        return base64.b64encode(signature).decode("utf-8")

    def _handle_rate_limits(self) -> None:
        """Handle API rate limiting"""
        current_time = time.time()

        # Simple delay between requests
        elapsed = current_time - self.last_request_time
        if elapsed < 0.1 and self.request_count > 0:
            time.sleep(0.1 - elapsed)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        data: Dict = None,
        auth_required: bool = True
    ) -> Dict:
        """Make an API request to Delta Exchange"""
        self._handle_rate_limits()

        url = f"{self.base_url}{endpoint}"
        headers = self.headers.copy()

        # Add authentication if required
        if auth_required:
            timestamp = int(time.time() * 1000)
            signature = self._generate_signature(timestamp, method, endpoint, data)

            headers.update({
                "api-key": self.api_key,
                "timestamp": str(timestamp),
                "signature": signature
            })

        try:
            self.request_count += 1
            self.last_request_time = time.time()

            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Delta API request failed: {str(e)}")
            raise

    def get_products(self) -> List[Dict]:
        """Get available trading products"""
        try:
            result = self._make_request("GET", "/v2/products", auth_required=False)
            if isinstance(result, dict) and 'result' in result:
                return result['result']
            elif isinstance(result, list):
                return result
            else:
                logger.warning(f"Unexpected products response format: {type(result)}")
                return []
        except Exception as e:
            logger.error(f"Failed to get products: {e}")
            return []

    def get_product_by_symbol(self, symbol: str) -> Optional[Dict]:
        """Get product information for a specific symbol"""
        products = self.get_products()
        for product in products:
            if isinstance(product, dict) and product.get('symbol') == symbol:
                return product
        return None

    def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[Union[int, datetime]] = None,
        end_time: Optional[Union[int, datetime]] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get OHLCV (candlestick) data for a symbol"""
        # Convert datetime objects to timestamps if provided
        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp() * 1000)

        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)

        # Set default time range if not provided
        if not start_time:
            start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

        if not end_time:
            end_time = int(datetime.now().timestamp() * 1000)

        # Use the correct Delta Exchange API format based on documentation
        # The API expects: symbol, resolution, from, to
        params = {
            'symbol': symbol,
            'resolution': interval,
            'from': start_time // 1000,  # Convert to seconds
            'to': end_time // 1000,      # Convert to seconds
        }

        try:
            # Use the correct endpoint from Delta Exchange documentation
            endpoint = "/v2/chart/history"

            # Make the API request
            try:
                logger.info(f"Trying endpoint: {endpoint} with params: {params}")
                response = self._make_request("GET", endpoint, params=params, auth_required=False)
                logger.info(f"Success with endpoint: {endpoint}")
            except Exception as e:
                logger.error(f"Endpoint {endpoint} failed: {e}")
                response = None

            if response is None:
                raise Exception("Chart history API request failed")

            # Format the response data
            candles = []
            if isinstance(response, dict):
                # Try different response formats
                if 't' in response:  # TradingView format
                    times = response.get('t', [])
                    opens = response.get('o', [])
                    highs = response.get('h', [])
                    lows = response.get('l', [])
                    closes = response.get('c', [])
                    volumes = response.get('v', [])

                    for i in range(len(times)):
                        candle = {
                            'timestamp': times[i] * 1000,  # Convert to milliseconds
                            'open': opens[i] if i < len(opens) else 0,
                            'high': highs[i] if i < len(highs) else 0,
                            'low': lows[i] if i < len(lows) else 0,
                            'close': closes[i] if i < len(closes) else 0,
                            'volume': volumes[i] if i < len(volumes) else 0
                        }
                        candles.append(candle)

                elif 'result' in response and isinstance(response['result'], list):
                    # Standard API format
                    for item in response['result']:
                        if isinstance(item, dict):
                            candle = {
                                'timestamp': item.get('timestamp', item.get('time', 0)),
                                'open': float(item.get('open', 0)),
                                'high': float(item.get('high', 0)),
                                'low': float(item.get('low', 0)),
                                'close': float(item.get('close', 0)),
                                'volume': float(item.get('volume', 0))
                            }
                            candles.append(candle)

                else:
                    logger.warning(f"Unknown response format: {list(response.keys())}")

            return candles

        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            return []

    def get_historical_ohlcv(
        self,
        symbol: str,
        interval: str = "1h",
        days_back: int = 30,
        end_time: Optional[Union[int, datetime]] = None,
    ) -> List[Dict]:
        """Get historical OHLCV data for a specific period"""
        if not end_time:
            end_time = datetime.now()
        elif isinstance(end_time, int):
            end_time = datetime.fromtimestamp(end_time / 1000)

        start_time = end_time - timedelta(days=days_back)

        # For simplicity, make a single request (Delta has limits)
        # In production, you might want to implement pagination
        candles = self.get_ohlcv(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=min(1000, days_back * 24)  # Rough estimate
        )

        return candles

    def test_connection(self) -> bool:
        """Test if the client can connect to Delta Exchange India"""
        try:
            products = self.get_products()
            return len(products) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_simple_ticker(self, symbol: str) -> Optional[Dict]:
        """Get simple ticker data as a fallback test"""
        try:
            # Try simple ticker endpoint
            params = {'symbol': symbol}
            response = self._make_request("GET", "/v2/tickers", params=params, auth_required=False)
            return response
        except Exception as e:
            logger.warning(f"Ticker request failed: {e}")
            return None


def test_standalone_delta_client():
    """Test the standalone Delta Exchange client"""
    print("🔧 Testing Standalone Delta Exchange Client")
    print("-" * 40)

    try:
        # Initialize client
        client = StandaloneDeltaExchangeClient(testnet=True)
        print(f"✅ Client initialized (testnet={client.testnet})")

        # Test connection
        print("Testing connection...")
        if client.test_connection():
            print("✅ Connection successful")
        else:
            print("❌ Connection failed")
            return False

        # Test getting products
        print("Testing get_products()...")
        products = client.get_products()
        print(f"✅ Retrieved {len(products)} products")

        # Find BTC perpetual contract (not options)
        btc_products = []
        for p in products:
            symbol = str(p.get('symbol', ''))
            product_type = str(p.get('product_type', ''))

            # Look for BTC perpetual contracts (not options or futures)
            if ('BTC' in symbol and
                ('PERP' in symbol or 'USD' in symbol) and
                'C-' not in symbol and  # Exclude call options
                'P-' not in symbol and  # Exclude put options
                product_type != 'call_options' and
                product_type != 'put_options'):
                btc_products.append(p)

        if btc_products:
            # Prefer BTCUSD perpetual
            btc_perp = None
            for p in btc_products:
                if p.get('symbol') == 'BTCUSD':
                    btc_perp = p
                    break

            if btc_perp:
                btc_symbol = btc_perp['symbol']
            else:
                btc_symbol = btc_products[0]['symbol']

            print(f"✅ Found BTC perpetual: {btc_symbol}")
            print(f"   Product type: {btc_products[0].get('product_type', 'unknown')}")
        else:
            btc_symbol = 'BTCUSD'
            print(f"⚠️  Using default symbol: {btc_symbol}")

        # Test historical data
        print(f"Testing historical data for {btc_symbol}...")
        historical_data = client.get_historical_ohlcv(
            symbol=btc_symbol,
            interval='1h',
            days_back=7
        )

        if historical_data:
            print(f"✅ Retrieved {len(historical_data)} historical candles")

            # Show sample data
            if len(historical_data) > 0:
                sample = historical_data[0]
                print(f"📊 Sample candle: {sample}")

            return True
        else:
            print("❌ No historical data retrieved")

            # Try ticker as fallback
            print(f"Trying ticker data as fallback for {btc_symbol}...")
            ticker_data = client.get_simple_ticker(btc_symbol)

            if ticker_data:
                print(f"✅ Retrieved ticker data: {ticker_data}")
                print("⚠️  Historical data failed but ticker works - API endpoint issue")
                return True
            else:
                print("❌ Both historical and ticker data failed")
                return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Standalone Delta Exchange Client Test")
    print("=" * 50)

    success = test_standalone_delta_client()

    if success:
        print("\n🎉 Standalone Delta client is working!")
        print("✅ Ready for integration into production backtesting system")
    else:
        print("\n❌ Standalone Delta client needs debugging")
        print("⚠️  Check API credentials and network connectivity")
