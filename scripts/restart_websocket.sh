#!/bin/bash

# Restart WebSocket Server with Fixed Handler
# Fixes the handler signature issue

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

print_status "🔧 Restarting WebSocket server with fixed handler..."

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    print_error "Please run this script from the SmartMarketOOPS root directory"
    exit 1
fi

# Kill any existing WebSocket processes
print_status "🛑 Stopping existing WebSocket servers..."
pkill -f "python.*websocket" 2>/dev/null || true
pkill -f "reliable_websocket_server" 2>/dev/null || true
pkill -f "mock_websocket_server" 2>/dev/null || true

# Wait for processes to stop
sleep 2

# Check if Python environment exists
if [ ! -d "venv" ]; then
    print_error "Python virtual environment not found"
    exit 1
fi

# Activate Python environment
source venv/bin/activate

# Start the fixed WebSocket server
print_status "🚀 Starting fixed WebSocket server..."

python backend/websocket/reliable_websocket_server.py &
WEBSOCKET_PID=$!

# Wait for server to start
sleep 3

# Check if server started successfully
if kill -0 $WEBSOCKET_PID 2>/dev/null; then
    print_success "✅ WebSocket server started successfully (PID: $WEBSOCKET_PID)"
    print_status "📡 WebSocket server: ws://localhost:3001"
    print_status "🔄 The handler signature issue has been fixed"
    print_status ""
    print_status "🌐 You can now refresh your browser at:"
    print_status "   http://localhost:3000/dashboard"
    print_status ""
    print_status "Expected behavior:"
    print_status "   ✅ No more 'missing 1 required positional argument' errors"
    print_status "   ✅ WebSocket connections should work properly"
    print_status "   ✅ Real-time data should start flowing"
    print_status ""
    print_status "Press Ctrl+C to stop the WebSocket server"
    
    # Wait for the server process
    wait $WEBSOCKET_PID
else
    print_error "❌ WebSocket server failed to start"
    exit 1
fi
