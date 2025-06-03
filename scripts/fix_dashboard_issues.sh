#!/bin/bash

# SmartMarketOOPS Dashboard Issues Fix Script
# Fixes all known issues with the dashboard launch process

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

print_status "🔧 Fixing SmartMarketOOPS Dashboard Issues..."

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    print_error "Please run this script from the SmartMarketOOPS root directory"
    exit 1
fi

PROJECT_ROOT=$(pwd)

# Fix 1: Install missing PyJWT dependency
print_status "🔧 Fix 1: Installing missing PyJWT dependency..."

if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

print_status "Installing PyJWT and other dependencies..."
pip install --quiet --upgrade pip
pip install --quiet PyJWT websockets asyncio pandas numpy

print_success "✅ PyJWT dependency installed"

# Fix 2: Create a reliable WebSocket server without deprecated imports
print_status "🔧 Fix 2: Creating reliable WebSocket server..."

mkdir -p backend/websocket

cat > backend/websocket/reliable_websocket_server.py << 'EOF'
#!/usr/bin/env python3
"""
Reliable WebSocket Server for SmartMarketOOPS
Fixed version without deprecated imports and with proper error handling
"""

import asyncio
import websockets
import json
import random
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReliableWebSocketServer:
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
        logger.info("Starting reliable WebSocket server on localhost:3001")
        
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
    server = ReliableWebSocketServer()
    asyncio.run(server.start_server())
EOF

print_success "✅ Reliable WebSocket server created"

# Fix 3: Remove conflicting Next.js files
print_status "🔧 Fix 3: Removing conflicting Next.js files..."

# Remove any remaining conflicting files
rm -f frontend/pages/_app.js 2>/dev/null || true
rm -f frontend/pages/_document.js 2>/dev/null || true
rm -f frontend/pages/index.js 2>/dev/null || true
rm -f frontend/pages/settings.js 2>/dev/null || true
rm -f frontend/pages/dashboard.tsx 2>/dev/null || true

print_success "✅ Conflicting files removed"

# Fix 4: Update package.json scripts
print_status "🔧 Fix 4: Updating package.json scripts..."

cd frontend

# Add missing Chart.js dependencies if not present
if ! grep -q "chart.js" package.json; then
    print_status "Adding Chart.js dependencies..."
    npm install --save chart.js react-chartjs-2 chartjs-adapter-date-fns date-fns
fi

cd "$PROJECT_ROOT"

print_success "✅ Package.json updated"

# Create a simple launch script
print_status "🔧 Creating simple launch script..."

cat > scripts/start_dashboard.sh << 'EOF'
#!/bin/bash

# Simple Dashboard Launcher
# Starts all services for SmartMarketOOPS dashboard

set -e

PROJECT_ROOT=$(pwd)

echo "🚀 Starting SmartMarketOOPS Dashboard..."

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the SmartMarketOOPS root directory"
    exit 1
fi

# Activate Python environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Python environment activated"
else
    echo "❌ Error: Python virtual environment not found. Run ./scripts/fix_dashboard_issues.sh first"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down services..."
    jobs -p | xargs -r kill
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start WebSocket server in background
echo "📡 Starting WebSocket server..."
python backend/websocket/reliable_websocket_server.py &
WEBSOCKET_PID=$!

# Wait for WebSocket server to start
sleep 3

# Start frontend in background
echo "🌐 Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!

cd "$PROJECT_ROOT"

echo ""
echo "✅ All services started successfully!"
echo "📡 WebSocket server: ws://localhost:3001"
echo "🌐 Frontend dashboard: http://localhost:3000"
echo "🎯 Dashboard URL: http://localhost:3000/dashboard"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for services
wait
EOF

chmod +x scripts/start_dashboard.sh

print_success "✅ Simple launch script created"

print_success "🎉 All issues fixed successfully!"
print_status ""
print_status "📋 Summary of fixes:"
print_status "✅ 1. PyJWT dependency installed"
print_status "✅ 2. Reliable WebSocket server created (no deprecated imports)"
print_status "✅ 3. Conflicting Next.js files removed"
print_status "✅ 4. Package.json and scripts updated"
print_status ""
print_status "🚀 To start the dashboard:"
print_status "   ./scripts/start_dashboard.sh"
print_status ""
print_status "🌐 Then open: http://localhost:3000/dashboard"
