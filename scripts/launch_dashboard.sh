#!/bin/bash

# SmartMarketOOPS Dashboard Launch Script
# Launches all services needed for the Real-Time Trading Dashboard

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if we're in the right directory
if [ ! -f "package.json" ] && [ ! -d "frontend" ]; then
    print_error "Please run this script from the SmartMarketOOPS root directory"
    exit 1
fi

# Get the project root directory
PROJECT_ROOT=$(pwd)

print_status "Starting SmartMarketOOPS Real-Time Trading Dashboard..."
print_status "Project root: $PROJECT_ROOT"

# Check prerequisites
print_status "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    print_error "Node.js version 18+ is required. Current version: $(node --version)"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

print_success "Prerequisites check passed"

# Setup Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating Python virtual environment..."
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet asyncio websockets pandas numpy torch scikit-learn PyJWT python-jose[cryptography] python-multipart fastapi uvicorn aiofiles python-dotenv

# Setup frontend dependencies
print_status "Setting up frontend dependencies..."
cd frontend

# Install npm dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    print_status "Installing npm dependencies..."
    npm install --silent
else
    print_status "npm dependencies already installed"
fi

# Go back to project root
cd "$PROJECT_ROOT"

# Create necessary directories
mkdir -p backend/websocket
mkdir -p ml/src/intelligence
mkdir -p logs

print_success "Setup completed successfully!"

# Function to start WebSocket server
start_websocket_server() {
    print_status "Starting WebSocket server..."
    cd "$PROJECT_ROOT"
    source venv/bin/activate

    # Always use the simplified WebSocket server for reliability
    print_status "Creating reliable WebSocket server..."
    cat > backend/websocket/simple_websocket_server.py << 'EOF'
import asyncio
import websockets
import json
import random
import time
from datetime import datetime

class SimpleWebSocketServer:
    def __init__(self):
        self.clients = set()
        self.symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD']
        self.prices = {'BTCUSD': 50000, 'ETHUSD': 3000, 'ADAUSD': 0.5, 'SOLUSD': 100}

    async def register(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        await websocket.send(json.dumps({
            'type': 'connection_status',
            'data': {'status': 'connected'}
        }))

    async def unregister(self, websocket):
        self.clients.discard(websocket)
        print(f"Client disconnected: {websocket.remote_address}")

    async def handle_client(self, websocket, path):
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                elif data.get('type') == 'subscribe':
                    print(f"Client subscribed to: {data.get('data', {}).get('channel')}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def broadcast_market_data(self):
        while True:
            if self.clients:
                for symbol in self.symbols:
                    # Generate realistic price movement
                    change = random.uniform(-0.02, 0.02)
                    self.prices[symbol] *= (1 + change)

                    market_data = {
                        'type': 'market_data',
                        'data': {
                            'symbol': symbol,
                            'price': round(self.prices[symbol], 2),
                            'change': round(self.prices[symbol] * change, 2),
                            'changePercent': round(change * 100, 2),
                            'volume': random.randint(100000, 1000000),
                            'high24h': round(self.prices[symbol] * 1.05, 2),
                            'low24h': round(self.prices[symbol] * 0.95, 2),
                            'timestamp': int(time.time() * 1000)
                        }
                    }

                    # Broadcast to all clients
                    disconnected = set()
                    for client in self.clients:
                        try:
                            await client.send(json.dumps(market_data))
                        except websockets.exceptions.ConnectionClosed:
                            disconnected.add(client)

                    # Remove disconnected clients
                    self.clients -= disconnected

            await asyncio.sleep(2)  # Update every 2 seconds

    async def start_server(self):
        print("Starting WebSocket server on localhost:3001")

        # Start market data broadcasting
        asyncio.create_task(self.broadcast_market_data())

        # Start WebSocket server
        server = await websockets.serve(
            self.handle_client,
            "localhost",
            3001,
            ping_interval=30,
            ping_timeout=10
        )

        print("WebSocket server started on ws://localhost:3001")
        await server.wait_closed()

if __name__ == "__main__":
    server = SimpleWebSocketServer()
    asyncio.run(server.start_server())
EOF
        python backend/websocket/simple_websocket_server.py
}

# Function to start frontend
start_frontend() {
    print_status "Starting frontend development server..."
    cd "$PROJECT_ROOT/frontend"
    npm run dev
}

# Check if we should start services
if [ "$1" = "--start" ]; then
    print_status "Starting all services..."

    # Start WebSocket server in background
    start_websocket_server &
    WEBSOCKET_PID=$!

    # Wait a moment for WebSocket server to start
    sleep 3

    # Start frontend
    start_frontend &
    FRONTEND_PID=$!

    # Function to cleanup on exit
    cleanup() {
        print_status "Shutting down services..."
        kill $WEBSOCKET_PID 2>/dev/null || true
        kill $FRONTEND_PID 2>/dev/null || true
        exit 0
    }

    # Set trap to cleanup on script exit
    trap cleanup SIGINT SIGTERM

    print_success "All services started!"
    print_status "WebSocket server: ws://localhost:3001"
    print_status "Frontend dashboard: http://localhost:3000"
    print_status "Press Ctrl+C to stop all services"

    # Wait for services
    wait
else
    print_success "Setup completed! To start the dashboard, run:"
    print_status "./scripts/launch_dashboard.sh --start"
    print_status ""
    print_status "Or start services manually:"
    print_status "1. WebSocket server: python backend/websocket/simple_websocket_server.py"
    print_status "2. Frontend: cd frontend && npm run dev"
    print_status "3. Open browser: http://localhost:3000/dashboard"
fi
