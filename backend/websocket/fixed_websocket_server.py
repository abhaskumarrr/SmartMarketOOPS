#!/usr/bin/env python3
"""
Fixed WebSocket Server for SmartMarketOOPS
Properly handles websocket connections with correct method signature
"""

import asyncio
import websockets
import json
import random
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedWebSocketServer:
    def __init__(self):
        self.clients = set()
        self.symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD']
        self.prices = {
            'BTCUSD': 50000,
            'ETHUSD': 3000,
            'ADAUSD': 0.5,
            'SOLUSD': 100,
            'DOTUSD': 25
        }
        self.signal_counter = 0

    async def register_client(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

        # Send connection confirmation
        try:
            await websocket.send(json.dumps({
                'type': 'connection_status',
                'data': {'status': 'connected'}
            }))
        except Exception as e:
            logger.error(f"Error sending connection status: {e}")

    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        logger.info("Client disconnected")

    async def handle_client(self, websocket, path):
        """Handle individual client connections with correct signature"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')

                    if message_type == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                    elif message_type == 'subscribe':
                        channel = data.get('data', {}).get('channel')
                        symbols = data.get('data', {}).get('symbols', [])
                        logger.info(f"Client subscribed to {channel}: {symbols}")

                        # Send initial data for subscribed symbols
                        if channel == 'market_data':
                            for symbol in symbols:
                                if symbol in self.prices:
                                    await self.send_market_data(websocket, symbol)

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from client")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            await self.unregister_client(websocket)

    async def send_market_data(self, websocket, symbol):
        """Send market data for a specific symbol"""
        try:
            market_data = {
                'type': 'market_data',
                'data': {
                    'symbol': symbol,
                    'price': round(self.prices[symbol], 2),
                    'change': round(random.uniform(-100, 100), 2),
                    'changePercent': round(random.uniform(-2, 2), 2),
                    'volume': random.randint(100000, 1000000),
                    'high24h': round(self.prices[symbol] * 1.05, 2),
                    'low24h': round(self.prices[symbol] * 0.95, 2),
                    'timestamp': int(time.time() * 1000)
                }
            }
            await websocket.send(json.dumps(market_data))
        except Exception as e:
            logger.error(f"Error sending market data: {e}")

    async def broadcast_market_data(self):
        """Broadcast market data to all connected clients"""
        while True:
            try:
                if self.clients:
                    for symbol in self.symbols:
                        # Generate realistic price movement
                        change_percent = random.uniform(-0.02, 0.02)  # ±2% max change
                        self.prices[symbol] *= (1 + change_percent)

                        market_data = {
                            'type': 'market_data',
                            'data': {
                                'symbol': symbol,
                                'price': round(self.prices[symbol], 2),
                                'change': round(self.prices[symbol] * change_percent, 2),
                                'changePercent': round(change_percent * 100, 2),
                                'volume': random.randint(100000, 1000000),
                                'high24h': round(self.prices[symbol] * 1.05, 2),
                                'low24h': round(self.prices[symbol] * 0.95, 2),
                                'timestamp': int(time.time() * 1000)
                            }
                        }

                        # Broadcast to all clients
                        disconnected_clients = set()
                        for client in self.clients.copy():
                            try:
                                await client.send(json.dumps(market_data))
                            except websockets.exceptions.ConnectionClosed:
                                disconnected_clients.add(client)
                            except Exception as e:
                                logger.error(f"Error broadcasting to client: {e}")
                                disconnected_clients.add(client)

                        # Remove disconnected clients
                        self.clients -= disconnected_clients

                await asyncio.sleep(2)  # Update every 2 seconds

            except Exception as e:
                logger.error(f"Error in market data broadcast: {e}")
                await asyncio.sleep(5)

    async def generate_trading_signals(self):
        """Generate mock trading signals"""
        while True:
            try:
                await asyncio.sleep(random.uniform(15, 45))  # Random interval 15-45 seconds

                if self.clients:
                    symbol = random.choice(self.symbols)
                    signal_types = ['buy', 'sell', 'strong_buy', 'strong_sell']
                    qualities = ['excellent', 'good', 'fair', 'poor']

                    self.signal_counter += 1

                    signal = {
                        'type': 'trading_signal',
                        'data': {
                            'id': f"signal_{int(time.time())}_{self.signal_counter}",
                            'symbol': symbol,
                            'signal_type': random.choice(signal_types),
                            'confidence': random.uniform(0.6, 0.95),
                            'quality': random.choice(qualities),
                            'price': self.prices[symbol],
                            'timestamp': int(time.time() * 1000),
                            'transformer_prediction': random.uniform(0.5, 0.9),
                            'ensemble_prediction': random.uniform(0.5, 0.9),
                            'smc_score': random.uniform(0.4, 0.9),
                            'technical_score': random.uniform(0.4, 0.9),
                            'stop_loss': self.prices[symbol] * random.uniform(0.96, 0.99),
                            'take_profit': self.prices[symbol] * random.uniform(1.01, 1.04),
                            'position_size': random.uniform(0.05, 0.15),
                            'risk_reward_ratio': random.uniform(1.5, 3.0)
                        }
                    }

                    # Broadcast signal to all clients
                    disconnected_clients = set()
                    for client in self.clients.copy():
                        try:
                            await client.send(json.dumps(signal))
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                        except Exception as e:
                            logger.error(f"Error sending signal: {e}")
                            disconnected_clients.add(client)

                    # Remove disconnected clients
                    self.clients -= disconnected_clients

                    logger.info(f"Generated signal: {signal['data']['signal_type']} {symbol} @ {signal['data']['price']:.2f}")

            except Exception as e:
                logger.error(f"Error generating trading signals: {e}")
                await asyncio.sleep(10)

    async def start_server(self):
        """Start the WebSocket server"""
        logger.info("Starting fixed WebSocket server on localhost:3001")

        # Start background tasks
        market_data_task = asyncio.create_task(self.broadcast_market_data())
        signals_task = asyncio.create_task(self.generate_trading_signals())

        try:
            # Start WebSocket server
            server = await websockets.serve(
                self.handle_client,
                "localhost",
                3001,
                ping_interval=30,
                ping_timeout=10
            )

            logger.info("✅ WebSocket server started on ws://localhost:3001")
            logger.info("📊 Broadcasting market data every 2 seconds")
            logger.info("🎯 Generating trading signals every 15-45 seconds")

            # Wait for server to close
            await server.wait_closed()

        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        except Exception as e:
            logger.error(f"Server error: {e}")
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

            logger.info("WebSocket server stopped")

if __name__ == "__main__":
    server = FixedWebSocketServer()
    asyncio.run(server.start_server())
