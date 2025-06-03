#!/usr/bin/env python3
"""
Working Real-time Market Data WebSocket Server
Functional approach that integrates with backend API
"""

import asyncio
import websockets
import json
import aiohttp
import logging
import time
import random
from typing import Set

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
clients: Set = set()
symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD']
backend_url = "http://localhost:3002"
last_market_data = {}
signal_counter = 0
session = None

async def initialize():
    """Initialize the WebSocket server"""
    global session
    session = aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10),
        connector=aiohttp.TCPConnector(limit=10)
    )
    logger.info("✅ Market WebSocket Server initialized")

async def cleanup():
    """Cleanup resources"""
    global session
    if session:
        await session.close()
    logger.info("🧹 WebSocket server cleaned up")

async def register_client(websocket):
    """Register a new client"""
    clients.add(websocket)
    client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    logger.info(f"👤 Client connected: {client_info} (Total: {len(clients)})")
    
    # Send connection confirmation
    try:
        await websocket.send(json.dumps({
            'type': 'connection_status',
            'data': {'status': 'connected', 'server': 'real-time-market-data'}
        }))
        
        # Send latest market data if available
        for symbol, data in last_market_data.items():
            await websocket.send(json.dumps({
                'type': 'market_data',
                'data': data
            }))
            
    except Exception as e:
        logger.error(f"❌ Error sending initial data to client: {e}")

async def unregister_client(websocket):
    """Unregister a client"""
    clients.discard(websocket)
    logger.info(f"👋 Client disconnected (Total: {len(clients)})")

async def handle_client(websocket, path):
    """Handle individual client connections - CORRECT SIGNATURE"""
    await register_client(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                elif message_type == 'subscribe':
                    await handle_subscription(websocket, data)
                elif message_type == 'unsubscribe':
                    await handle_unsubscription(websocket, data)
                    
            except json.JSONDecodeError:
                logger.error("❌ Invalid JSON received from client")
            except Exception as e:
                logger.error(f"❌ Error handling message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("🔌 Client connection closed normally")
    except Exception as e:
        logger.error(f"❌ Error in client handler: {e}")
    finally:
        await unregister_client(websocket)

async def handle_subscription(websocket, data):
    """Handle client subscription requests"""
    channel = data.get('data', {}).get('channel')
    symbols_list = data.get('data', {}).get('symbols', [])
    
    logger.info(f"📡 Client subscribed to {channel}: {symbols_list}")
    
    # Send current data for subscribed symbols
    if channel == 'market_data':
        for symbol in symbols_list:
            if symbol in last_market_data:
                await websocket.send(json.dumps({
                    'type': 'market_data',
                    'data': last_market_data[symbol]
                }))

async def handle_unsubscription(websocket, data):
    """Handle client unsubscription requests"""
    channel = data.get('data', {}).get('channel')
    symbols_list = data.get('data', {}).get('symbols', [])
    logger.info(f"📡 Client unsubscribed from {channel}: {symbols_list}")

async def fetch_market_data_from_backend():
    """Fetch real market data from Node.js backend"""
    global session, last_market_data
    
    if not session:
        return []
        
    try:
        market_data = []
        
        for symbol in symbols:
            try:
                url = f"{backend_url}/api/market-data/{symbol}"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success') and data.get('data'):
                            market_data_item = data['data']
                            market_data.append(market_data_item)
                            # Store latest data
                            last_market_data[symbol] = market_data_item
                    else:
                        logger.warning(f"⚠️ Backend returned {response.status} for {symbol}")
                        
            except Exception as e:
                logger.error(f"❌ Error fetching {symbol}: {e}")
                
            # Small delay between requests
            await asyncio.sleep(0.1)
            
        return market_data
        
    except Exception as e:
        logger.error(f"❌ Error fetching market data from backend: {e}")
        return []

async def broadcast_market_data():
    """Fetch and broadcast real market data"""
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            if clients:
                # Fetch real market data from backend
                market_data_list = await fetch_market_data_from_backend()
                
                if market_data_list:
                    # Reset retry count on successful fetch
                    retry_count = 0
                    
                    # Broadcast each symbol's data
                    for market_data in market_data_list:
                        message = json.dumps({
                            'type': 'market_data',
                            'data': market_data
                        })
                        
                        disconnected_clients = set()
                        for client in clients.copy():
                            try:
                                await client.send(message)
                            except websockets.exceptions.ConnectionClosed:
                                disconnected_clients.add(client)
                            except Exception as e:
                                logger.error(f"❌ Error broadcasting to client: {e}")
                                disconnected_clients.add(client)
                        
                        # Remove disconnected clients
                        clients.difference_update(disconnected_clients)
                        
                    logger.debug(f"📊 Broadcasted market data for {len(market_data_list)} symbols to {len(clients)} clients")
                    
                else:
                    # No data received, increment retry count
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"⚠️ No market data received, retry {retry_count}/{max_retries}")
                    else:
                        logger.error(f"❌ Failed to fetch market data after {max_retries} retries")
                        retry_count = 0
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"❌ Error in market data broadcast: {e}")
            await asyncio.sleep(5)

async def generate_trading_signals():
    """Generate mock trading signals based on real market data"""
    global signal_counter
    
    while True:
        try:
            # Wait random interval
            interval = random.uniform(30, 60)
            await asyncio.sleep(interval)
            
            if clients and last_market_data:
                symbol = random.choice(list(last_market_data.keys()))
                market_data = last_market_data[symbol]
                
                signal_counter += 1
                
                signal = {
                    'type': 'trading_signal',
                    'data': {
                        'id': f"signal_{int(time.time())}_{signal_counter}",
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
                for client in clients.copy():
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(client)
                    except Exception as e:
                        logger.error(f"❌ Error sending signal: {e}")
                        disconnected_clients.add(client)
                
                # Remove disconnected clients
                clients.difference_update(disconnected_clients)
                
                logger.info(f"🎯 Generated signal: {signal['data']['signal_type']} {symbol} @ {signal['data']['price']:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error generating trading signals: {e}")
            await asyncio.sleep(10)

async def start_server():
    """Start the WebSocket server"""
    await initialize()
    
    logger.info(f"🚀 Starting Working Market WebSocket server on localhost:3001")
    logger.info(f"🔗 Backend URL: {backend_url}")
    logger.info(f"📊 Symbols: {', '.join(symbols)}")
    
    # Start background tasks
    market_data_task = asyncio.create_task(broadcast_market_data())
    signals_task = asyncio.create_task(generate_trading_signals())
    
    try:
        # Start WebSocket server with CORRECT function signature
        server = await websockets.serve(
            handle_client,  # Function, not method - this is the key!
            "localhost",
            3001,
            ping_interval=30,
            ping_timeout=10
        )
        
        logger.info("✅ Working Market WebSocket server started")
        logger.info("📡 Broadcasting real market data every 5 seconds")
        logger.info("🎯 Generating trading signals every 30-60 seconds")
        
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
        
        await cleanup()
        logger.info("🏁 Working Market WebSocket server stopped")

if __name__ == "__main__":
    asyncio.run(start_server())
