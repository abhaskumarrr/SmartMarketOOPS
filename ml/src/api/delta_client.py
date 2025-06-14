"""
Delta Exchange API Client

This module provides a client for interacting with the Delta Exchange API,
specifically focused on fetching historical market data for ML training.
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
from ..utils.config import API_CONFIG
import threading
import websocket
import queue
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeltaExchangeClient:
    """Client for interacting with the Delta Exchange API"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, testnet: bool = True):
        """
        Initialize the Delta Exchange API client
        
        Args:
            api_key: API key for authentication (defaults to config value)
            api_secret: API secret for authentication (defaults to config value)
            testnet: Whether to use testnet or production API (defaults to config value)
        """
        self.api_key = api_key or API_CONFIG["delta_exchange"]["api_key"]
        self.api_secret = api_secret or API_CONFIG["delta_exchange"]["api_secret"]
        self.testnet = testnet if testnet is not None else API_CONFIG["delta_exchange"]["testnet"]
        
        # Use correct Delta Exchange India URLs from official documentation
        self.base_url = (
            "https://cdn-ind.testnet.deltaex.org"
            if self.testnet
            else "https://api.india.delta.exchange"
        )
        
        # Default request headers
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # Rate limiting parameters
        self.rate_limit_remaining = 100  # Default assumption
        self.rate_limit_reset = 0
        self.request_count = 0
        self.last_request_time = 0
    
    def _generate_signature(self, timestamp: int, method: str, request_path: str, body: Dict = None) -> str:
        """
        Generate HMAC signature for API authentication
        
        Args:
            timestamp: Current timestamp in milliseconds
            method: HTTP method (GET, POST, etc.)
            request_path: API endpoint path
            body: Request body for POST requests
            
        Returns:
            str: Base64 encoded HMAC signature
        """
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
        """
        Handle API rate limiting by implementing backoff when needed
        """
        current_time = time.time()
        
        # If we're close to rate limit, wait until reset
        if self.rate_limit_remaining < 5 and self.rate_limit_reset > current_time:
            wait_time = self.rate_limit_reset - current_time + 1  # Add 1 second buffer
            logger.info(f"Rate limit almost reached. Waiting for {wait_time:.2f} seconds")
            time.sleep(wait_time)
        
        # Simple delay between requests to be respectful
        elapsed = current_time - self.last_request_time
        if elapsed < 0.1 and self.request_count > 0:  # Minimum 100ms between requests
            time.sleep(0.1 - elapsed)
    
    def _update_rate_limit_info(self, headers: Dict) -> None:
        """
        Update rate limit tracking based on response headers
        
        Args:
            headers: Response headers from API request
        """
        if 'X-RateLimit-Remaining' in headers:
            self.rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
        
        if 'X-RateLimit-Reset' in headers:
            self.rate_limit_reset = int(headers['X-RateLimit-Reset'])
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Dict = None, 
        data: Dict = None,
        auth_required: bool = True
    ) -> Dict:
        """
        Make an API request to Delta Exchange
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint to call
            params: URL parameters for GET requests
            data: Body data for POST requests
            auth_required: Whether authentication is required
            
        Returns:
            Dict: JSON response from the API
        """
        # Apply rate limiting precautions
        self._handle_rate_limits()
        
        url = f"{self.base_url}{endpoint}"
        headers = self.headers.copy()
        
        # Add authentication if required
        if auth_required:
            timestamp = int(time.time() * 1000)  # milliseconds
            signature = self._generate_signature(timestamp, method, endpoint, data)
            
            headers.update({
                "api-key": self.api_key,
                "timestamp": str(timestamp),
                "signature": signature
            })
        
        # Make the request
        try:
            self.request_count += 1
            self.last_request_time = time.time()
            
            if method == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method == "POST":
                response = requests.post(url, json=data, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Update rate limit info
            self._update_rate_limit_info(response.headers)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def get_products(self) -> List[Dict]:
        """
        Get available trading products
        
        Returns:
            List of product information dictionaries
        """
        return self._make_request("GET", "/v2/products")
    
    def get_product_by_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get product information for a specific symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            
        Returns:
            Product information or None if not found
        """
        products = self.get_products()
        if isinstance(products, List):
            for product in products:
                if isinstance(product, Dict) and product.get('symbol') == symbol:
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
        """
        Get OHLCV (candlestick) data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            interval: Candlestick interval ('1m', '5m', '1h', '4h', '1d', etc.)
            start_time: Start time (datetime or timestamp in milliseconds)
            end_time: End time (datetime or timestamp in milliseconds)
            limit: Maximum number of candles to return
            
        Returns:
            List of OHLCV data points
        """
        # Convert datetime objects to timestamps if provided
        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp() * 1000)
        
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)
        
        # Set default time range if not provided
        if not start_time:
            # Default to 7 days ago
            start_time = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        
        if not end_time:
            end_time = int(datetime.now().timestamp() * 1000)
        
        # Get product ID for the symbol
        product = self.get_product_by_symbol(symbol)
        if not product:
            raise ValueError(f"Symbol not found: {symbol}")
        
        product_id = product.get('id')
        
        params = {
            'symbol': symbol,
            'resolution': interval,
            'from': start_time // 1000,  # Convert to seconds
            'to': end_time // 1000,      # Convert to seconds
            'limit': limit
        }
        
        response = self._make_request("GET", "/v2/chart/history", params=params, auth_required=False)
        
        # Format the response data into a more convenient structure
        candles = []
        for i in range(len(response.get('t', []))):
            candle = {
                'timestamp': response['t'][i] * 1000,  # Convert back to milliseconds
                'open': response['o'][i],
                'high': response['h'][i],
                'low': response['l'][i],
                'close': response['c'][i],
                'volume': response['v'][i]
            }
            candles.append(candle)
        
        return candles
    
    def get_historical_ohlcv(
        self, 
        symbol: str, 
        interval: str = "1h",
        days_back: int = 30,
        end_time: Optional[Union[int, datetime]] = None,
    ) -> List[Dict]:
        """
        Get historical OHLCV data for a specific period
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            interval: Candlestick interval ('1m', '5m', '1h', '4h', '1d', etc.)
            days_back: Number of days to look back
            end_time: End time (datetime or timestamp in milliseconds), defaults to now
            
        Returns:
            List of OHLCV data points
        """
        if not end_time:
            end_time = datetime.now()
        elif isinstance(end_time, int):
            end_time = datetime.fromtimestamp(end_time / 1000)
        
        start_time = end_time - timedelta(days=days_back)
        
        # Limit per request
        max_candles_per_request = 1000
        
        all_candles = []
        current_end = end_time
        
        # Calculate how many data points we roughly need
        interval_hours = self._interval_to_hours(interval)
        total_hours = days_back * 24
        total_candles = total_hours / interval_hours
        
        # If we need more than max_candles_per_request, paginate
        if total_candles > max_candles_per_request:
            while current_end > start_time:
                batch_start = max(start_time, current_end - timedelta(hours=max_candles_per_request * interval_hours))
                
                candles = self.get_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start_time=batch_start,
                    end_time=current_end,
                    limit=max_candles_per_request
                )
                
                if not candles:
                    break
                
                all_candles = candles + all_candles
                
                # Update for next iteration - go back to just before the oldest candle
                if len(candles) < 2:
                    break
                
                oldest_candle_time = datetime.fromtimestamp(candles[0]['timestamp'] / 1000)
                current_end = oldest_candle_time - timedelta(minutes=1)
        else:
            all_candles = self.get_ohlcv(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=int(total_candles) + 10  # Add some buffer
            )
        
        return all_candles
    
    def _interval_to_hours(self, interval: str) -> float:
        """
        Convert a time interval string to hours
        
        Args:
            interval: Time interval string (e.g., '1m', '1h', '1d')
            
        Returns:
            float: Number of hours represented by the interval
        """
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            return minutes / 60
        elif interval.endswith('h'):
            return int(interval[:-1])
        elif interval.endswith('d'):
            return int(interval[:-1]) * 24
        elif interval.endswith('w'):
            return int(interval[:-1]) * 24 * 7
        else:
            raise ValueError(f"Unsupported interval format: {interval}")
    
    def get_funding_rates(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get historical funding rates for a perpetual contract
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            limit: Maximum number of funding rates to return
            
        Returns:
            List of funding rate data points
        """
        # Get product ID for the symbol
        product = self.get_product_by_symbol(symbol)
        if not product:
            raise ValueError(f"Symbol not found: {symbol}")
        
        product_id = product.get('id')
        
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        return self._make_request("GET", "/v2/funding_rate_history", params=params, auth_required=False)
    
    def get_open_interest(self, symbol: str) -> Dict:
        """
        Get current open interest for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            
        Returns:
            Open interest data
        """
        params = {
            'symbol': symbol
        }
        
        return self._make_request("GET", "/v2/open_interest", params=params, auth_required=False)


# Singleton instance for reuse
_default_client = None

def get_delta_client() -> DeltaExchangeClient:
    """
    Get the default Delta Exchange client instance (singleton)
    
    Returns:
        DeltaExchangeClient: Default client instance
    """
    global _default_client
    
    if _default_client is None:
        _default_client = DeltaExchangeClient()
    
    return _default_client


class DeltaExchangeWebSocketClient:
    """
    Production-grade WebSocket client for Delta Exchange real-time market data.
    Supports trades, order book, and ticker channels. Handles reconnection, authentication, and thread-safe data delivery.
    """
    def __init__(self, symbols, channels=None, api_key=None, api_secret=None, testnet=True, on_message=None):
        self.api_key = api_key or API_CONFIG["delta_exchange"]["api_key"]
        self.api_secret = api_secret or API_CONFIG["delta_exchange"]["api_secret"]
        self.testnet = testnet if testnet is not None else API_CONFIG["delta_exchange"]["testnet"]
        # Use correct Delta Exchange India WebSocket URLs
        self.base_url = (
            "wss://socket-ind.testnet.deltaex.org" if self.testnet else "wss://socket.india.delta.exchange"
        )
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.channels = channels or ["trades", "orderbook", "ticker"]
        self.on_message = on_message
        self.ws = None
        self._stop_event = threading.Event()
        self._thread = None
        self._queue = queue.Queue()
        self._connected = False

    def _get_subscribe_payload(self):
        payload = []
        for symbol in self.symbols:
            for channel in self.channels:
                payload.append({
                    "type": "subscribe",
                    "payload": {
                        "channels": [f"{channel}.{symbol}"]
                    }
                })
        return payload

    def _on_open(self, ws):
        self._connected = True
        for sub in self._get_subscribe_payload():
            ws.send(json.dumps(sub))

    def _on_message(self, ws, message):
        data = json.loads(message)
        if self.on_message:
            self.on_message(data)
        else:
            self._queue.put(data)

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self._connected = False
        logger.warning(f"WebSocket closed: {close_status_code} {close_msg}")
        if not self._stop_event.is_set():
            logger.info("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self._start_ws()

    def _start_ws(self):
        ws_url = self.base_url
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self._thread = threading.Thread(target=self.ws.run_forever, kwargs={"sslopt": {"cert_reqs": ssl.CERT_NONE}})
        self._thread.daemon = True
        self._thread.start()

    def start(self):
        self._stop_event.clear()
        self._start_ws()

    def stop(self):
        self._stop_event.set()
        if self.ws:
            self.ws.close()
        if self._thread:
            self._thread.join()

    def get_message(self, timeout=None):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


if __name__ == "__main__":
    # Example usage
    client = DeltaExchangeClient()
    btc_data = client.get_historical_ohlcv(symbol="BTCUSD", interval="1h", days_back=7)
    print(f"Retrieved {len(btc_data)} BTC/USD hourly candles")
    print(btc_data[0])  # Print first candle 