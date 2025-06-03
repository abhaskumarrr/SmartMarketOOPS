#!/bin/bash

# Reliable SmartMarketOOPS Startup Script
# Handles all startup scenarios and error recovery

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Load port configuration
if [ -f "frontend/.env.local" ]; then
    source frontend/.env.local
    WEBSOCKET_PORT=${WEBSOCKET_PORT:-3001}
    FRONTEND_PORT=${FRONTEND_PORT:-3000}
else
    WEBSOCKET_PORT=3001
    FRONTEND_PORT=3000
fi

print_status "🚀 Starting SmartMarketOOPS Real-Time Trading Dashboard..."
print_status "📡 WebSocket Port: $WEBSOCKET_PORT"
print_status "🌐 Frontend Port: $FRONTEND_PORT"

# Check prerequisites
if [ ! -d "venv" ]; then
    print_error "Python virtual environment not found. Run setup first."
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    print_warning "Node modules not found. Installing..."
    cd frontend
    npm install
    cd ..
fi

# Activate Python environment
source venv/bin/activate

# Function to cleanup on exit
cleanup() {
    print_status "🛑 Shutting down services..."
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start WebSocket server
print_status "📡 Starting WebSocket server on port $WEBSOCKET_PORT..."

if [ -f "backend/websocket/port_aware_websocket_server.py" ]; then
    python backend/websocket/port_aware_websocket_server.py $WEBSOCKET_PORT &
    WEBSOCKET_PID=$!
elif [ -f "backend/websocket/reliable_websocket_server.py" ]; then
    python backend/websocket/reliable_websocket_server.py &
    WEBSOCKET_PID=$!
else
    print_error "No WebSocket server found. Run fix scripts first."
    exit 1
fi

# Wait for WebSocket server to start
sleep 3

# Check if WebSocket server started successfully
if ! kill -0 $WEBSOCKET_PID 2>/dev/null; then
    print_error "WebSocket server failed to start"
    exit 1
fi

print_success "✅ WebSocket server started"

# Start frontend server
print_status "🌐 Starting frontend server on port $FRONTEND_PORT..."

cd frontend

# Set the port for Next.js
export PORT=$FRONTEND_PORT

npm run dev &
FRONTEND_PID=$!

cd ..

# Wait for frontend to start
sleep 5

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    print_error "Frontend server failed to start"
    kill $WEBSOCKET_PID 2>/dev/null || true
    exit 1
fi

print_success "✅ Frontend server started"

print_success "🎉 All services started successfully!"
print_status ""
print_status "📊 Dashboard URLs:"
print_status "   🌐 Main Dashboard: http://localhost:$FRONTEND_PORT"
print_status "   🎯 Trading Dashboard: http://localhost:$FRONTEND_PORT/dashboard"
print_status "   📡 WebSocket: ws://localhost:$WEBSOCKET_PORT"
print_status ""
print_status "🔍 Expected Features:"
print_status "   ✅ Real-time price charts (2-second updates)"
print_status "   ✅ Trading signals (15-45 second intervals)"
print_status "   ✅ ML Intelligence dashboard (4 tabs)"
print_status "   ✅ Portfolio monitoring"
print_status "   ✅ WebSocket connectivity status"
print_status ""
print_status "Press Ctrl+C to stop all services"

# Wait for services
wait
