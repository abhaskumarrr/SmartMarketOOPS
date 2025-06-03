#!/bin/bash

# SmartMarketOOPS Port Conflict Resolution Script
# Resolves port conflicts and configures alternative ports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "🔧 Resolving SmartMarketOOPS Port Conflicts..."

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to find available port
find_available_port() {
    local start_port=$1
    local port=$start_port
    
    while check_port $port; do
        port=$((port + 1))
        if [ $port -gt $((start_port + 100)) ]; then
            print_error "Could not find available port in range $start_port-$((start_port + 100))"
            return 1
        fi
    done
    
    echo $port
}

# Function to kill process on port
kill_port_process() {
    local port=$1
    local pid=$(lsof -ti:$port)
    
    if [ ! -z "$pid" ]; then
        print_warning "Killing process $pid on port $port"
        kill -9 $pid 2>/dev/null || true
        sleep 2
        
        # Verify process is killed
        if check_port $port; then
            print_error "Failed to kill process on port $port"
            return 1
        else
            print_success "Process on port $port terminated"
            return 0
        fi
    fi
}

# Check and resolve WebSocket port (3001)
print_status "Checking WebSocket port 3001..."

if check_port 3001; then
    print_warning "Port 3001 is in use"
    
    # Show what's using the port
    print_status "Process using port 3001:"
    lsof -i :3001 || true
    
    # Ask user preference (auto-kill for script automation)
    print_status "Attempting to free port 3001..."
    
    if kill_port_process 3001; then
        WEBSOCKET_PORT=3001
        print_success "Port 3001 is now available"
    else
        # Find alternative port
        print_status "Finding alternative port for WebSocket server..."
        WEBSOCKET_PORT=$(find_available_port 3002)
        if [ $? -eq 0 ]; then
            print_success "Using alternative WebSocket port: $WEBSOCKET_PORT"
        else
            print_error "Could not find available port for WebSocket server"
            exit 1
        fi
    fi
else
    WEBSOCKET_PORT=3001
    print_success "Port 3001 is available"
fi

# Check and resolve Frontend port (3000)
print_status "Checking Frontend port 3000..."

if check_port 3000; then
    print_warning "Port 3000 is in use"
    
    # Show what's using the port
    print_status "Process using port 3000:"
    lsof -i :3000 || true
    
    if kill_port_process 3000; then
        FRONTEND_PORT=3000
        print_success "Port 3000 is now available"
    else
        # Find alternative port
        print_status "Finding alternative port for Frontend server..."
        FRONTEND_PORT=$(find_available_port 3001)
        if [ $? -eq 0 ]; then
            print_success "Using alternative Frontend port: $FRONTEND_PORT"
        else
            print_error "Could not find available port for Frontend server"
            exit 1
        fi
    fi
else
    FRONTEND_PORT=3000
    print_success "Port 3000 is available"
fi

# Update environment configuration
print_status "Updating environment configuration..."

# Update frontend environment
cat > frontend/.env.local << EOF
NEXT_PUBLIC_WS_URL=ws://localhost:$WEBSOCKET_PORT
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:$FRONTEND_PORT
NODE_ENV=development
WEBSOCKET_PORT=$WEBSOCKET_PORT
FRONTEND_PORT=$FRONTEND_PORT
EOF

# Update backend environment
mkdir -p backend
cat > backend/.env << EOF
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=$WEBSOCKET_PORT
ML_API_HOST=localhost
ML_API_PORT=8000
JWT_SECRET=your-secret-key-here
ENVIRONMENT=development
FRONTEND_PORT=$FRONTEND_PORT
EOF

print_success "Environment configuration updated"

# Create port-aware WebSocket server
print_status "Creating port-aware WebSocket server..."

cat > backend/websocket/port_aware_websocket_server.py << 'EOF'
#!/usr/bin/env python3
"""
Port-Aware WebSocket Server for SmartMarketOOPS
Automatically handles port conflicts and uses configured ports
"""

import asyncio
import websockets
import json
import random
import time
import os
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortAwareWebSocketServer:
    def __init__(self, port=None):
        # Get port from environment or use default
        self.port = port or int(os.getenv('WEBSOCKET_PORT', 3001))
        self.host = os.getenv('WEBSOCKET_HOST', 'localhost')
        
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
    
    async def find_available_port(self, start_port):
        """Find an available port starting from start_port"""
        port = start_port
        while port < start_port + 100:
            try:
                # Try to bind to the port
                server = await websockets.serve(
                    lambda ws, path: None,
                    self.host,
                    port,
                    ping_interval=None
                )
                server.close()
                await server.wait_closed()
                return port
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    port += 1
                    continue
                else:
                    raise
        
        raise Exception(f"Could not find available port in range {start_port}-{start_port + 100}")
    
    async def register_client(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send connection confirmation
        try:
            await websocket.send(json.dumps({
                'type': 'connection_status',
                'data': {'status': 'connected', 'port': self.port}
            }))
        except Exception as e:
            logger.error(f"Error sending connection status: {e}")
    
    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected")
    
    async def handle_client(self, websocket, path):
        """Handle individual client connections"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'ping':
                        await websocket.send(json.dumps({'type': 'pong', 'port': self.port}))
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
        """Start the WebSocket server with port conflict resolution"""
        logger.info(f"Starting port-aware WebSocket server...")
        
        # Try to use configured port, find alternative if needed
        try:
            server_port = self.port
            
            # Start background tasks
            market_data_task = asyncio.create_task(self.broadcast_market_data())
            signals_task = asyncio.create_task(self.generate_trading_signals())
            
            try:
                # Start WebSocket server
                server = await websockets.serve(
                    self.handle_client,
                    self.host,
                    server_port,
                    ping_interval=30,
                    ping_timeout=10
                )
                
                logger.info(f"✅ WebSocket server started on ws://{self.host}:{server_port}")
                logger.info(f"📊 Broadcasting market data every 2 seconds")
                logger.info(f"🎯 Generating trading signals every 15-45 seconds")
                
                # Update environment with actual port used
                if server_port != self.port:
                    logger.info(f"Updated WebSocket port from {self.port} to {server_port}")
                    self.port = server_port
                
                # Wait for server to close
                await server.wait_closed()
                
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    logger.warning(f"Port {server_port} is in use, finding alternative...")
                    server_port = await self.find_available_port(server_port + 1)
                    logger.info(f"Using alternative port: {server_port}")
                    
                    # Retry with new port
                    server = await websockets.serve(
                        self.handle_client,
                        self.host,
                        server_port,
                        ping_interval=30,
                        ping_timeout=10
                    )
                    
                    logger.info(f"✅ WebSocket server started on ws://{self.host}:{server_port}")
                    self.port = server_port
                    
                    await server.wait_closed()
                else:
                    raise
                
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
    # Get port from command line argument or environment
    port = None
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error("Invalid port number provided")
            sys.exit(1)
    
    server = PortAwareWebSocketServer(port)
    asyncio.run(server.start_server())
EOF

print_success "Port-aware WebSocket server created"

# Save port configuration for other scripts
cat > scripts/port_config.sh << EOF
#!/bin/bash
# Port configuration for SmartMarketOOPS
export WEBSOCKET_PORT=$WEBSOCKET_PORT
export FRONTEND_PORT=$FRONTEND_PORT
export WS_URL="ws://localhost:$WEBSOCKET_PORT"
export FRONTEND_URL="http://localhost:$FRONTEND_PORT"
EOF

chmod +x scripts/port_config.sh

print_success "✅ Port conflicts resolved!"
print_status "📋 Configuration:"
print_status "   WebSocket Server: ws://localhost:$WEBSOCKET_PORT"
print_status "   Frontend Server: http://localhost:$FRONTEND_PORT"
print_status "   Dashboard URL: http://localhost:$FRONTEND_PORT/dashboard"
print_status ""
print_status "🔧 Use the port-aware server: python backend/websocket/port_aware_websocket_server.py"
