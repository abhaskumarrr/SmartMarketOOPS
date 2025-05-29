import requests
import logging
import time

class RealTimeSignalPipeline:
    """
    Streaming pipeline for real-time signal generation and backend integration.
    Consumes normalized data, applies trading logic, and sends signals to backend.
    """
    def __init__(self, backend_url, symbol, signal_fn, api_key=None):
        self.backend_url = backend_url.rstrip('/')
        self.symbol = symbol
        self.signal_fn = signal_fn  # Callable: data -> signal dict or None
        self.api_key = api_key
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)

    def process(self, normalized_data):
        for item in normalized_data:
            try:
                signal = self.signal_fn(item)
                if signal:
                    self._send_signal(signal)
            except Exception as e:
                self.logger.error(f"Error in signal pipeline: {e}")

    def _send_signal(self, signal):
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        payload = {'symbol': self.symbol, 'data': signal}
        try:
            resp = self.session.post(f'{self.backend_url}/realtime/market-data', json=payload, headers=headers, timeout=2)
            if resp.status_code != 200:
                self.logger.warning(f"Failed to send signal: {resp.status_code} {resp.text}")
        except Exception as e:
            self.logger.error(f"Exception sending signal to backend: {e}")

def smc_signal_fn(data):
    """
    Example production-grade signal function for SMC/FVG/liquidity analysis.
    Returns a signal dict if a trading opportunity is detected, else None.
    """
    # Only process trades and orderbook updates
    if data.get('type') == 'trade':
        # Example: Large trade volume triggers a signal
        if data['volume'] > 10:  # Threshold can be tuned
            return {
                'type': 'trade_signal',
                'timestamp': data['timestamp'],
                'symbol': data['symbol'],
                'price': data['price'],
                'volume': data['volume'],
                'side': data.get('side'),
                'reason': 'Large trade volume'
            }
    elif data.get('type') == 'orderbook':
        # Example: Order book imbalance triggers a signal
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        if bids and asks:
            top_bid = float(bids[0][0]) if isinstance(bids[0], (list, tuple)) else 0
            top_ask = float(asks[0][0]) if isinstance(asks[0], (list, tuple)) else 0
            if top_bid > 0 and top_ask > 0 and (top_bid / top_ask) > 1.01:
                return {
                    'type': 'orderbook_signal',
                    'timestamp': data['timestamp'],
                    'symbol': data['symbol'],
                    'top_bid': top_bid,
                    'top_ask': top_ask,
                    'reason': 'Order book imbalance'
                }
    # Add more SMC/FVG/liquidity logic as needed
    return None

if __name__ == "__main__":
    import sys
    import signal
    from ml.src.api.delta_client import DeltaExchangeWebSocketClient
    from ml.src.data.data_loader import RealTimeDataNormalizer

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("streaming_pipeline")

    symbol = "BTCUSD"  # Can be parameterized
    backend_url = "http://localhost:3001"  # Adjust as needed

    normalizer = RealTimeDataNormalizer()
    pipeline = RealTimeSignalPipeline(backend_url, symbol, smc_signal_fn)

    def on_message(raw):
        normalized = normalizer.normalize(raw)
        if normalized:
            pipeline.process(normalized)

    ws_client = DeltaExchangeWebSocketClient(symbols=[symbol], on_message=on_message)

    def shutdown(signum, frame):
        logger.info("Shutting down WebSocket client...")
        ws_client.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(f"Starting real-time streaming pipeline for {symbol}")
    ws_client.start()
    while True:
        time.sleep(1) 