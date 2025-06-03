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
