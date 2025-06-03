"""
Mock WebSocket Server for Real-Time Trading Dashboard
Task #30: Real-Time Trading Dashboard
Development server for testing real-time functionality
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any
import websockets
try:
    import jwt
except ImportError:
    print("Warning: PyJWT not installed. Authentication will be simplified.")
    jwt = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockWebSocketServer:
    """Mock WebSocket server for real-time trading data"""

    def __init__(self, host: str = "localhost", port: int = 3001):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.subscriptions: Dict[websockets.WebSocketServerProtocol, Dict[str, List[str]]] = {}

        # Mock data
        self.symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD']
        self.market_data = self._initialize_market_data()
        self.portfolio_data = self._initialize_portfolio_data()

        # Data generation tasks
        self.data_tasks: List[asyncio.Task] = []

    def _initialize_market_data(self) -> Dict[str, Dict]:
        """Initialize mock market data"""
        base_prices = {
            'BTCUSD': 50000,
            'ETHUSD': 3000,
            'ADAUSD': 0.5,
            'SOLUSD': 100,
            'DOTUSD': 25
        }

        market_data = {}
        for symbol, base_price in base_prices.items():
            market_data[symbol] = {
                'symbol': symbol,
                'price': base_price,
                'change': 0,
                'changePercent': 0,
                'volume': random.uniform(100000, 1000000),
                'high24h': base_price * 1.05,
                'low24h': base_price * 0.95,
                'timestamp': int(time.time() * 1000)
            }

        return market_data

    def _initialize_portfolio_data(self) -> Dict:
        """Initialize mock portfolio data"""
        return {
            'totalValue': 10000,
            'totalPnL': 0,
            'totalPnLPercent': 0,
            'positions': {
                'BTCUSD': {
                    'symbol': 'BTCUSD',
                    'amount': 0.1,
                    'averagePrice': 48000,
                    'currentPrice': 50000,
                    'pnl': 200,
                    'pnlPercent': 4.17
                }
            }
        }

    async def authenticate_client(self, websocket: websockets.WebSocketServerProtocol, token: str) -> bool:
        """Authenticate client using JWT token"""
        try:
            # For demo purposes, accept any non-empty token
            if token and len(token) > 10:
                logger.info(f"Client authenticated: {websocket.remote_address}")
                return True
            return False
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client"""
        self.clients.add(websocket)
        self.subscriptions[websocket] = {}
        logger.info(f"Client registered: {websocket.remote_address}")

        # Send connection confirmation
        await self.send_message(websocket, {
            'type': 'connection_status',
            'data': {'status': 'connected'}
        })

    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client"""
        self.clients.discard(websocket)
        self.subscriptions.pop(websocket, None)
        logger.info(f"Client unregistered: {websocket.remote_address}")

    async def send_message(self, websocket: websockets.WebSocketServerProtocol, message: Dict):
        """Send message to a specific client"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast_message(self, message: Dict, channel: str = None):
        """Broadcast message to subscribed clients"""
        if not self.clients:
            return

        for websocket in self.clients.copy():
            try:
                # Check if client is subscribed to this channel
                if channel and channel not in self.subscriptions.get(websocket, {}):
                    continue

                await self.send_message(websocket, message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                await self.unregister_client(websocket)

    async def handle_subscription(self, websocket: websockets.WebSocketServerProtocol, data: Dict):
        """Handle client subscription requests"""
        channel = data.get('channel')
        symbols = data.get('symbols', [])

        if channel not in self.subscriptions[websocket]:
            self.subscriptions[websocket][channel] = []

        if symbols:
            self.subscriptions[websocket][channel].extend(symbols)

        logger.info(f"Client {websocket.remote_address} subscribed to {channel}: {symbols}")

        # Send initial data
        if channel == 'market_data':
            for symbol in symbols:
                if symbol in self.market_data:
                    await self.send_message(websocket, {
                        'type': 'market_data',
                        'data': self.market_data[symbol]
                    })
        elif channel == 'portfolio':
            await self.send_message(websocket, {
                'type': 'portfolio_update',
                'data': self.portfolio_data
            })

    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming client messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')

            if message_type == 'subscribe':
                await self.handle_subscription(websocket, data.get('data', {}))
            elif message_type == 'ping':
                await self.send_message(websocket, {'type': 'pong'})
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def generate_market_data(self):
        """Generate realistic market data updates"""
        while True:
            try:
                for symbol in self.symbols:
                    if symbol in self.market_data:
                        # Generate price movement
                        current_price = self.market_data[symbol]['price']
                        change_percent = random.uniform(-0.05, 0.05)  # ±5% max change
                        new_price = current_price * (1 + change_percent)

                        # Update market data
                        old_price = self.market_data[symbol]['price']
                        self.market_data[symbol].update({
                            'price': new_price,
                            'change': new_price - old_price,
                            'changePercent': ((new_price - old_price) / old_price) * 100,
                            'volume': random.uniform(100000, 1000000),
                            'timestamp': int(time.time() * 1000)
                        })

                        # Broadcast update
                        await self.broadcast_message({
                            'type': 'market_data',
                            'data': self.market_data[symbol]
                        }, 'market_data')

                await asyncio.sleep(2)  # Update every 2 seconds

            except Exception as e:
                logger.error(f"Error generating market data: {e}")
                await asyncio.sleep(5)

    async def generate_trading_signals(self):
        """Generate mock trading signals"""
        while True:
            try:
                await asyncio.sleep(random.uniform(10, 30))  # Random interval 10-30 seconds

                symbol = random.choice(self.symbols)
                signal_types = ['buy', 'sell', 'strong_buy', 'strong_sell']
                qualities = ['excellent', 'good', 'fair', 'poor']

                signal = {
                    'id': f"signal_{int(time.time())}_{random.randint(1000, 9999)}",
                    'symbol': symbol,
                    'signal_type': random.choice(signal_types),
                    'confidence': random.uniform(0.5, 0.95),
                    'quality': random.choice(qualities),
                    'price': self.market_data[symbol]['price'],
                    'timestamp': int(time.time() * 1000),
                    'transformer_prediction': random.uniform(0.4, 0.9),
                    'ensemble_prediction': random.uniform(0.4, 0.9),
                    'smc_score': random.uniform(0.3, 0.9),
                    'technical_score': random.uniform(0.3, 0.9),
                    'stop_loss': self.market_data[symbol]['price'] * random.uniform(0.95, 0.98),
                    'take_profit': self.market_data[symbol]['price'] * random.uniform(1.02, 1.05),
                    'position_size': random.uniform(0.05, 0.15),
                    'risk_reward_ratio': random.uniform(1.5, 3.0)
                }

                await self.broadcast_message({
                    'type': 'trading_signal',
                    'data': signal
                }, 'trading_signals')

                logger.info(f"Generated signal: {signal['signal_type']} {signal['symbol']} @ {signal['price']:.2f}")

            except Exception as e:
                logger.error(f"Error generating trading signals: {e}")

    async def update_portfolio(self):
        """Update portfolio data periodically"""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds

                # Update portfolio based on current prices
                total_value = 10000  # Base value
                total_pnl = 0

                for symbol, position in self.portfolio_data['positions'].items():
                    if symbol in self.market_data:
                        current_price = self.market_data[symbol]['price']
                        position['currentPrice'] = current_price
                        position['pnl'] = (current_price - position['averagePrice']) * position['amount']
                        position['pnlPercent'] = ((current_price - position['averagePrice']) / position['averagePrice']) * 100

                        total_pnl += position['pnl']

                self.portfolio_data['totalValue'] = total_value + total_pnl
                self.portfolio_data['totalPnL'] = total_pnl
                self.portfolio_data['totalPnLPercent'] = (total_pnl / total_value) * 100

                await self.broadcast_message({
                    'type': 'portfolio_update',
                    'data': self.portfolio_data
                }, 'portfolio')

            except Exception as e:
                logger.error(f"Error updating portfolio: {e}")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle individual client connections"""
        try:
            # For demo purposes, skip token authentication
            token = 'demo-token'

            # Authenticate client
            if not await self.authenticate_client(websocket, token):
                await websocket.close(code=4001, reason="Authentication failed")
                return

            # Register client
            await self.register_client(websocket)

            # Handle messages
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.unregister_client(websocket)

    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        # Start data generation tasks
        self.data_tasks = [
            asyncio.create_task(self.generate_market_data()),
            asyncio.create_task(self.generate_trading_signals()),
            asyncio.create_task(self.update_portfolio())
        ]

        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )

        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        finally:
            # Cancel data generation tasks
            for task in self.data_tasks:
                task.cancel()

            server.close()
            await server.wait_closed()
            logger.info("WebSocket server stopped")


async def main():
    """Main function to run the mock WebSocket server"""
    server = MockWebSocketServer()
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())
