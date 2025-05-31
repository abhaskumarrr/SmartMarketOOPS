#!/usr/bin/env python3
"""
Real Market Data Service for Enhanced SmartMarketOOPS System
Integrates live market data feeds from cryptocurrency exchanges
"""

import asyncio
import websockets
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import aiohttp
import ccxt.async_support as ccxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Real-time market data point"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    enabled: bool
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    rate_limit: int = 10  # requests per second
    symbols: List[str] = None


class RealMarketDataService:
    """Real-time market data service with multiple exchange support"""
    
    def __init__(self):
        """Initialize the real market data service"""
        self.exchanges = {}
        self.websocket_connections = {}
        self.data_callbacks = []
        self.is_running = False
        self.data_buffer = {}
        
        # Exchange configurations
        self.exchange_configs = {
            'delta': ExchangeConfig(
                name='delta',
                enabled=True,
                testnet=True,
                symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
            ),
            'binance': ExchangeConfig(
                name='binance',
                enabled=True,
                testnet=True,
                symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
            ),
            'kucoin': ExchangeConfig(
                name='kucoin',
                enabled=False,  # Disabled by default
                testnet=True,
                symbols=['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'ADA-USDT']
            )
        }
        
        logger.info("Real Market Data Service initialized")
    
    async def initialize_exchanges(self):
        """Initialize exchange connections"""
        logger.info("Initializing exchange connections...")
        
        for exchange_name, config in self.exchange_configs.items():
            if not config.enabled:
                continue
                
            try:
                if exchange_name == 'delta':
                    await self._initialize_delta_exchange(config)
                elif exchange_name == 'binance':
                    await self._initialize_binance_exchange(config)
                elif exchange_name == 'kucoin':
                    await self._initialize_kucoin_exchange(config)
                    
                logger.info(f"✅ {exchange_name.upper()} exchange initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize {exchange_name}: {e}")
    
    async def _initialize_delta_exchange(self, config: ExchangeConfig):
        """Initialize Delta Exchange connection"""
        try:
            # Use existing Delta Exchange client
            from ..api.delta_client import DeltaExchangeClient
            
            delta_client = DeltaExchangeClient(testnet=config.testnet)
            self.exchanges['delta'] = delta_client
            
            # Initialize WebSocket for real-time data
            await self._setup_delta_websocket(config)
            
        except Exception as e:
            logger.error(f"Delta Exchange initialization failed: {e}")
            raise
    
    async def _initialize_binance_exchange(self, config: ExchangeConfig):
        """Initialize Binance exchange connection"""
        try:
            # Initialize Binance testnet
            exchange = ccxt.binance({
                'apiKey': config.api_key,
                'secret': config.api_secret,
                'sandbox': config.testnet,
                'enableRateLimit': True,
            })
            
            await exchange.load_markets()
            self.exchanges['binance'] = exchange
            
            # Setup WebSocket for real-time data
            await self._setup_binance_websocket(config)
            
        except Exception as e:
            logger.error(f"Binance initialization failed: {e}")
            raise
    
    async def _initialize_kucoin_exchange(self, config: ExchangeConfig):
        """Initialize KuCoin exchange connection"""
        try:
            exchange = ccxt.kucoin({
                'apiKey': config.api_key,
                'secret': config.api_secret,
                'sandbox': config.testnet,
                'enableRateLimit': True,
            })
            
            await exchange.load_markets()
            self.exchanges['kucoin'] = exchange
            
        except Exception as e:
            logger.error(f"KuCoin initialization failed: {e}")
            raise
    
    async def _setup_delta_websocket(self, config: ExchangeConfig):
        """Setup Delta Exchange WebSocket connection"""
        try:
            # Delta WebSocket URL
            ws_url = "wss://testnet-ws.delta.exchange" if config.testnet else "wss://ws.delta.exchange"
            
            # Create WebSocket connection
            websocket = await websockets.connect(ws_url)
            self.websocket_connections['delta'] = websocket
            
            # Subscribe to market data channels
            for symbol in config.symbols:
                subscribe_msg = {
                    "type": "subscribe",
                    "payload": {
                        "channels": [
                            {"name": "ticker", "symbol": symbol},
                            {"name": "orderbook", "symbol": symbol}
                        ]
                    }
                }
                await websocket.send(json.dumps(subscribe_msg))
            
            # Start listening for messages
            asyncio.create_task(self._listen_delta_websocket())
            
        except Exception as e:
            logger.error(f"Delta WebSocket setup failed: {e}")
            raise
    
    async def _setup_binance_websocket(self, config: ExchangeConfig):
        """Setup Binance WebSocket connection"""
        try:
            # Binance WebSocket URL for testnet
            ws_url = "wss://testnet.binance.vision/ws-api/v3"
            
            websocket = await websockets.connect(ws_url)
            self.websocket_connections['binance'] = websocket
            
            # Subscribe to ticker streams
            streams = []
            for symbol in config.symbols:
                symbol_lower = symbol.lower()
                streams.extend([
                    f"{symbol_lower}@ticker",
                    f"{symbol_lower}@depth5@100ms"
                ])
            
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            
            # Start listening for messages
            asyncio.create_task(self._listen_binance_websocket())
            
        except Exception as e:
            logger.error(f"Binance WebSocket setup failed: {e}")
            raise
    
    async def _listen_delta_websocket(self):
        """Listen for Delta Exchange WebSocket messages"""
        websocket = self.websocket_connections.get('delta')
        if not websocket:
            return
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self._process_delta_message(data)
                
        except Exception as e:
            logger.error(f"Delta WebSocket listening error: {e}")
            # Attempt reconnection
            await self._reconnect_websocket('delta')
    
    async def _listen_binance_websocket(self):
        """Listen for Binance WebSocket messages"""
        websocket = self.websocket_connections.get('binance')
        if not websocket:
            return
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self._process_binance_message(data)
                
        except Exception as e:
            logger.error(f"Binance WebSocket listening error: {e}")
            # Attempt reconnection
            await self._reconnect_websocket('binance')
    
    async def _process_delta_message(self, data: Dict[str, Any]):
        """Process Delta Exchange WebSocket message"""
        try:
            if data.get('type') == 'ticker':
                symbol = data.get('symbol')
                ticker_data = data.get('data', {})
                
                market_data = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=float(ticker_data.get('open', 0)),
                    high=float(ticker_data.get('high', 0)),
                    low=float(ticker_data.get('low', 0)),
                    close=float(ticker_data.get('close', 0)),
                    volume=float(ticker_data.get('volume', 0)),
                    bid=float(ticker_data.get('bid', 0)),
                    ask=float(ticker_data.get('ask', 0))
                )
                
                # Calculate spread
                if market_data.bid and market_data.ask:
                    market_data.spread = market_data.ask - market_data.bid
                
                # Store and broadcast data
                await self._store_and_broadcast_data('delta', market_data)
                
        except Exception as e:
            logger.error(f"Error processing Delta message: {e}")
    
    async def _process_binance_message(self, data: Dict[str, Any]):
        """Process Binance WebSocket message"""
        try:
            if 'stream' in data and '@ticker' in data['stream']:
                ticker_data = data.get('data', {})
                symbol = ticker_data.get('s')
                
                market_data = MarketDataPoint(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(ticker_data.get('E', 0) / 1000),
                    open=float(ticker_data.get('o', 0)),
                    high=float(ticker_data.get('h', 0)),
                    low=float(ticker_data.get('l', 0)),
                    close=float(ticker_data.get('c', 0)),
                    volume=float(ticker_data.get('v', 0)),
                    bid=float(ticker_data.get('b', 0)),
                    ask=float(ticker_data.get('a', 0))
                )
                
                # Calculate spread
                if market_data.bid and market_data.ask:
                    market_data.spread = market_data.ask - market_data.bid
                
                # Store and broadcast data
                await self._store_and_broadcast_data('binance', market_data)
                
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")
    
    async def _store_and_broadcast_data(self, exchange: str, data: MarketDataPoint):
        """Store market data and broadcast to callbacks"""
        # Store in buffer
        key = f"{exchange}_{data.symbol}"
        self.data_buffer[key] = data
        
        # Broadcast to callbacks
        for callback in self.data_callbacks:
            try:
                await callback(exchange, data)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    async def _reconnect_websocket(self, exchange: str):
        """Reconnect WebSocket for an exchange"""
        logger.info(f"Attempting to reconnect {exchange} WebSocket...")
        
        try:
            # Close existing connection
            if exchange in self.websocket_connections:
                await self.websocket_connections[exchange].close()
                del self.websocket_connections[exchange]
            
            # Wait before reconnecting
            await asyncio.sleep(5)
            
            # Reinitialize connection
            config = self.exchange_configs[exchange]
            if exchange == 'delta':
                await self._setup_delta_websocket(config)
            elif exchange == 'binance':
                await self._setup_binance_websocket(config)
            
            logger.info(f"✅ {exchange} WebSocket reconnected")
            
        except Exception as e:
            logger.error(f"❌ Failed to reconnect {exchange}: {e}")
    
    def add_data_callback(self, callback: Callable):
        """Add a callback for real-time data updates"""
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable):
        """Remove a data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    async def get_latest_data(self, symbol: str, exchange: Optional[str] = None) -> Optional[MarketDataPoint]:
        """Get latest market data for a symbol"""
        if exchange:
            key = f"{exchange}_{symbol}"
            return self.data_buffer.get(key)
        else:
            # Return data from any available exchange
            for ex in self.exchanges.keys():
                key = f"{ex}_{symbol}"
                if key in self.data_buffer:
                    return self.data_buffer[key]
        return None
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                                limit: int = 100, exchange: str = 'binance') -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            if exchange in self.exchanges:
                ex = self.exchanges[exchange]
                ohlcv = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
            else:
                raise ValueError(f"Exchange {exchange} not available")
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    async def start(self):
        """Start the real market data service"""
        logger.info("Starting Real Market Data Service...")
        
        await self.initialize_exchanges()
        self.is_running = True
        
        logger.info("✅ Real Market Data Service started successfully")
    
    async def stop(self):
        """Stop the real market data service"""
        logger.info("Stopping Real Market Data Service...")
        
        self.is_running = False
        
        # Close WebSocket connections
        for exchange, websocket in self.websocket_connections.items():
            try:
                await websocket.close()
                logger.info(f"Closed {exchange} WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing {exchange} WebSocket: {e}")
        
        # Close exchange connections
        for exchange, ex in self.exchanges.items():
            try:
                if hasattr(ex, 'close'):
                    await ex.close()
                logger.info(f"Closed {exchange} exchange connection")
            except Exception as e:
                logger.error(f"Error closing {exchange} exchange: {e}")
        
        logger.info("✅ Real Market Data Service stopped")


# Global instance
_market_data_service = None

async def get_market_data_service() -> RealMarketDataService:
    """Get the global market data service instance"""
    global _market_data_service
    
    if _market_data_service is None:
        _market_data_service = RealMarketDataService()
        await _market_data_service.start()
    
    return _market_data_service


async def main():
    """Test the real market data service"""
    service = await get_market_data_service()
    
    # Add a test callback
    async def test_callback(exchange: str, data: MarketDataPoint):
        logger.info(f"📊 {exchange.upper()}: {data.symbol} = ${data.close:.2f} "
                   f"(Vol: {data.volume:.0f}, Spread: ${data.spread:.4f})")
    
    service.add_data_callback(test_callback)
    
    # Run for 60 seconds
    await asyncio.sleep(60)
    
    await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
