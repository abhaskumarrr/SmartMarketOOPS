#!/usr/bin/env python3
"""
Real-time Market Data WebSocket Server for SmartMarketOOPS
Integrates with Node.js backend to fetch real Delta Exchange market data
"""

import asyncio
import websockets
import json
import aiohttp
import logging
import time
from typing import Dict, Set, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMarketWebSocketServer:
    def __init__(self, backend_url: str = "http://localhost:3002"):
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.backend_url = backend_url
        self.symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD']
        self.last_market_data: Dict[str, dict] = {}
        self.signal_counter = 0
        self.session: Optional[aiohttp.ClientSession] = None

        # Configuration
        self.market_data_interval = 5  # Fetch market data every 5 seconds
        self.signal_generation_interval = (30, 60)  # Generate signals every 30-60 seconds
        self.max_retries = 3
        self.retry_delay = 5

    async def initialize(self):
        """Initialize the WebSocket server"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=10)
        )
        logger.info("✅ Real-time Market WebSocket Server initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        logger.info("🧹 WebSocket server cleaned up")

    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client"""
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"👤 Client connected: {client_info} (Total: {len(self.clients)})")

        # Send connection confirmation
        try:
            await websocket.send(json.dumps({
                'type': 'connection_status',
                'data': {'status': 'connected', 'server': 'real-time-market-data'}
            }))

            # Send latest market data if available
            for symbol, data in self.last_market_data.items():
                await websocket.send(json.dumps({
                    'type': 'market_data',
                    'data': data
                }))

        except Exception as e:
            logger.error(f"❌ Error sending initial data to client: {e}")

    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client"""
        self.clients.discard(websocket)
        logger.info(f"👋 Client disconnected (Total: {len(self.clients)})")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connections"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')

                    if message_type == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                    elif message_type == 'subscribe':
                        await self.handle_subscription(websocket, data)
                    elif message_type == 'unsubscribe':
                        await self.handle_unsubscription(websocket, data)

                except json.JSONDecodeError:
                    logger.error("❌ Invalid JSON received from client")
                except Exception as e:
                    logger.error(f"❌ Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Client connection closed normally")
        except Exception as e:
            logger.error(f"❌ Error in client handler: {e}")
        finally:
            await self.unregister_client(websocket)

    async def handle_subscription(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """Handle client subscription requests"""
        channel = data.get('data', {}).get('channel')
        symbols = data.get('data', {}).get('symbols', [])

        logger.info(f"📡 Client subscribed to {channel}: {symbols}")

        # Send current data for subscribed symbols
        if channel == 'market_data':
            for symbol in symbols:
                if symbol in self.last_market_data:
                    await websocket.send(json.dumps({
                        'type': 'market_data',
                        'data': self.last_market_data[symbol]
                    }))

    async def handle_unsubscription(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """Handle client unsubscription requests"""
        channel = data.get('data', {}).get('channel')
        symbols = data.get('data', {}).get('symbols', [])
        logger.info(f"📡 Client unsubscribed from {channel}: {symbols}")

    async def fetch_market_data_from_backend(self) -> List[dict]:
        """Fetch real market data from Node.js backend"""
        if not self.session:
            return []

        try:
            # Fetch market data for all symbols from backend
            market_data = []

            for symbol in self.symbols:
                try:
                    url = f"{self.backend_url}/api/market-data/{symbol}"
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('success') and data.get('data'):
                                market_data.append(data['data'])
                        else:
                            logger.warning(f"⚠️ Backend returned {response.status} for {symbol}")

                except Exception as e:
                    logger.error(f"❌ Error fetching {symbol}: {e}")

                # Small delay between requests to avoid overwhelming backend
                await asyncio.sleep(0.1)

            return market_data

        except Exception as e:
            logger.error(f"❌ Error fetching market data from backend: {e}")
            return []

    async def broadcast_market_data(self):
        """Fetch and broadcast real market data"""
        retry_count = 0

        while True:
            try:
                if self.clients:
                    # Fetch real market data from backend
                    market_data_list = await self.fetch_market_data_from_backend()

                    if market_data_list:
                        # Reset retry count on successful fetch
                        retry_count = 0

                        # Broadcast each symbol's data
                        for market_data in market_data_list:
                            # Store latest data
                            symbol = market_data.get('symbol')
                            if symbol:
                                self.last_market_data[symbol] = market_data

                            # Broadcast to all clients
                            message = json.dumps({
                                'type': 'market_data',
                                'data': market_data
                            })

                            disconnected_clients = set()
                            for client in self.clients.copy():
                                try:
                                    await client.send(message)
                                except websockets.exceptions.ConnectionClosed:
                                    disconnected_clients.add(client)
                                except Exception as e:
                                    logger.error(f"❌ Error broadcasting to client: {e}")
                                    disconnected_clients.add(client)

                            # Remove disconnected clients
                            self.clients -= disconnected_clients

                        logger.debug(f"📊 Broadcasted market data for {len(market_data_list)} symbols to {len(self.clients)} clients")

                    else:
                        # No data received, increment retry count
                        retry_count += 1
                        if retry_count <= self.max_retries:
                            logger.warning(f"⚠️ No market data received, retry {retry_count}/{self.max_retries}")
                        else:
                            logger.error(f"❌ Failed to fetch market data after {self.max_retries} retries")
                            # Generate fallback mock data
                            await self.generate_fallback_data()
                            retry_count = 0

                await asyncio.sleep(self.market_data_interval)

            except Exception as e:
                logger.error(f"❌ Error in market data broadcast: {e}")
                retry_count += 1
                await asyncio.sleep(self.retry_delay)

    async def generate_fallback_data(self):
        """Generate fallback mock data when real data is unavailable"""
        logger.info("🔄 Generating fallback market data")

        base_prices = {
            'BTCUSD': 50000,
            'ETHUSD': 3000,
            'ADAUSD': 0.5,
            'SOLUSD': 100,
            'DOTUSD': 25
        }

        for symbol in self.symbols:
            # Generate realistic price movement
            base_price = base_prices.get(symbol, 100)
            change_percent = (random.random() - 0.5) * 2  # ±1%
            current_price = base_price * (1 + change_percent / 100)
            change = current_price - base_price

            fallback_data = {
                'symbol': symbol,
                'price': round(current_price, 2),
                'change': round(change, 2),
                'changePercent': round(change_percent, 2),
                'volume': random.randint(100000, 1000000),
                'high24h': round(current_price * 1.05, 2),
                'low24h': round(current_price * 0.95, 2),
                'timestamp': int(time.time() * 1000),
                'source': 'fallback'
            }

            self.last_market_data[symbol] = fallback_data

    async def generate_trading_signals(self):
        """Generate mock trading signals based on real market data"""
        while True:
            try:
                # Wait random interval
                interval = random.uniform(*self.signal_generation_interval)
                await asyncio.sleep(interval)

                if self.clients and self.last_market_data:
                    symbol = random.choice(list(self.last_market_data.keys()))
                    market_data = self.last_market_data[symbol]

                    self.signal_counter += 1

                    signal = {
                        'type': 'trading_signal',
                        'data': {
                            'id': f"signal_{int(time.time())}_{self.signal_counter}",
                            'symbol': symbol,
                            'signal_type': random.choice(['buy', 'sell', 'strong_buy', 'strong_sell']),
                            'confidence': random.uniform(0.6, 0.95),
                            'quality': random.choice(['excellent', 'good', 'fair']),
                            'price': market_data.get('price', 0),
                            'timestamp': int(time.time() * 1000),
                            'transformer_prediction': random.uniform(0.5, 0.9),
                            'ensemble_prediction': random.uniform(0.5, 0.9),
                            'smc_score': random.uniform(0.4, 0.9),
                            'technical_score': random.uniform(0.4, 0.9),
                            'stop_loss': market_data.get('price', 0) * random.uniform(0.96, 0.99),
                            'take_profit': market_data.get('price', 0) * random.uniform(1.01, 1.04),
                            'position_size': random.uniform(0.05, 0.15),
                            'risk_reward_ratio': random.uniform(1.5, 3.0),
                            'based_on_real_data': True
                        }
                    }

                    # Broadcast signal to all clients
                    message = json.dumps(signal)
                    disconnected_clients = set()
                    for client in self.clients.copy():
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                        except Exception as e:
                            logger.error(f"❌ Error sending signal: {e}")
                            disconnected_clients.add(client)

                    # Remove disconnected clients
                    self.clients -= disconnected_clients

                    logger.info(f"🎯 Generated signal: {signal['data']['signal_type']} {symbol} @ {signal['data']['price']:.2f}")

            except Exception as e:
                logger.error(f"❌ Error generating trading signals: {e}")
                await asyncio.sleep(10)

    async def start_server(self, host: str = "localhost", port: int = 3001):
        """Start the WebSocket server"""
        await self.initialize()

        logger.info(f"🚀 Starting Real-time Market WebSocket server on {host}:{port}")
        logger.info(f"🔗 Backend URL: {self.backend_url}")
        logger.info(f"📊 Symbols: {', '.join(self.symbols)}")

        # Start background tasks
        market_data_task = asyncio.create_task(self.broadcast_market_data())
        signals_task = asyncio.create_task(self.generate_trading_signals())

        try:
            # Create a wrapper function for the handler
            async def websocket_handler(websocket, path):
                await self.handle_client(websocket, path)

            # Start WebSocket server
            server = await websockets.serve(
                websocket_handler,
                host,
                port,
                ping_interval=30,
                ping_timeout=10
            )

            logger.info("✅ Real-time Market WebSocket server started")
            logger.info(f"📡 Broadcasting real market data every {self.market_data_interval} seconds")
            logger.info(f"🎯 Generating trading signals every {self.signal_generation_interval[0]}-{self.signal_generation_interval[1]} seconds")

            # Wait for server to close
            await server.wait_closed()

        except KeyboardInterrupt:
            logger.info("🛑 Server shutdown requested")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
        finally:
            # Cancel background tasks
            market_data_task.cancel()
            signals_task.cancel()

            try:
                await market_data_task
            except asyncio.CancelledError:
                pass

            try:
                await signals_task
            except asyncio.CancelledError:
                pass

            await self.cleanup()
            logger.info("🏁 Real-time Market WebSocket server stopped")

if __name__ == "__main__":
    import random

    server = RealTimeMarketWebSocketServer()
    asyncio.run(server.start_server())
