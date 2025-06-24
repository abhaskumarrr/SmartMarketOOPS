#!/bin/bash

# SmartMarketOOPS Simple Stop Script

echo "🛑 Stopping SmartMarketOOPS..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Stop frontend
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    print_status "Stopping frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null || true
    rm -f logs/frontend.pid
fi

# Stop backend
if [ -f "logs/backend.pid" ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    print_status "Stopping backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    rm -f logs/backend.pid
fi

# Stop ML service
if [ -f "logs/ml-service.pid" ]; then
    ML_PID=$(cat logs/ml-service.pid)
    print_status "Stopping ML service (PID: $ML_PID)..."
    kill $ML_PID 2>/dev/null || true
    rm -f logs/ml-service.pid
fi

# Stop Docker services
print_status "Stopping infrastructure services..."
docker-compose -f docker-compose.simple.yml down

print_success "✅ SmartMarketOOPS stopped successfully!"
echo ""
echo "💾 Your data has been preserved"
echo "🔄 To restart: ./start-simple.sh"