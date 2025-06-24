#!/bin/bash

# SmartMarketOOPS Quick Start Script
# Assumes infrastructure is already running

set -e

echo "🚀 Quick Starting SmartMarketOOPS..."

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

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    print_error ".env.local not found. Please create it with your Delta Exchange credentials."
    exit 1
fi

# Load environment variables
export $(cat .env.local | grep -v '^#' | xargs)

# Create necessary directories
mkdir -p logs backups backend/logs ml/logs

# Check if database is accessible
print_status "Checking database connection..."
if docker exec smartmarket_postgres pg_isready -U smartmarket > /dev/null 2>&1; then
    print_success "Database is ready!"
else
    print_error "Database not accessible. Starting infrastructure..."
    docker-compose -f docker-compose.simple.yml up -d
    sleep 10
fi

# Install backend dependencies if needed
if [ ! -d "backend/node_modules" ]; then
    print_status "Installing backend dependencies..."
    cd backend && npm install && cd ..
fi

# Run database migrations
print_status "Running database migrations..."
cd backend
export DATABASE_URL="postgresql://smartmarket:secure_local_password_2024@localhost:5432/smartmarket"
npm run prisma:db:push > /dev/null 2>&1 || true
npm run prisma:generate > /dev/null 2>&1 || true
cd ..

# Install ML dependencies if needed
if [ ! -f "ml/.venv/bin/activate" ]; then
    print_status "Setting up ML environment..."
    cd ml
    python3 -m venv .venv
    source .venv/bin/activate
    pip install flask numpy > /dev/null 2>&1 || true
    cd ..
fi

# Kill any existing processes
print_status "Stopping any existing services..."
pkill -f "simple_app.py" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "ts-node" 2>/dev/null || true

# Start ML service
print_status "Starting ML service..."
cd ml
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH=$(pwd)
nohup python simple_app.py > ../logs/ml-service.log 2>&1 &
ML_PID=$!
echo $ML_PID > ../logs/ml-service.pid
cd ..
print_success "ML service started (PID: $ML_PID)"

# Wait for ML service
sleep 3

# Start backend service
print_status "Starting backend API..."
cd backend
export NODE_ENV=development
export DATABASE_URL="postgresql://smartmarket:secure_local_password_2024@localhost:5432/smartmarket"
export REDIS_URL="redis://localhost:6379"
export DELTA_API_KEY="VuBmLRHofoTVFSAMvzOrjJKMU3x1Xt"
export DELTA_API_SECRET="YW6KCAIuoON1vBciRGzn5v0YYg7aKlzXOkYamZUMoUpknMT0PMh6ewVXd2DY"
export DELTA_BASE_URL="https://testnet-api.delta.exchange"
export DELTA_TESTNET="true"

nohup npx ts-node simple-server.ts > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid
cd ..
print_success "Backend API started (PID: $BACKEND_PID)"

# Wait for services to start
print_status "Waiting for services to initialize..."
sleep 10

# Check service health
print_status "Checking service health..."

# Check ML service
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "✅ ML service is healthy"
else
    print_warning "⚠️  ML service may not be ready yet"
    echo "ML service log:"
    tail -5 logs/ml-service.log
fi

# Check backend service
if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
    print_success "✅ Backend API is healthy"
else
    print_warning "⚠️  Backend API may not be ready yet"
    echo "Backend log:"
    tail -5 logs/backend.log
fi

# Test Delta Exchange connection
print_status "Testing Delta Exchange connection..."
if python3 test-delta-simple.py > /dev/null 2>&1; then
    print_success "✅ Delta Exchange connection working"
else
    print_warning "⚠️  Delta Exchange connection may have issues"
fi

echo ""
print_success "🎉 SmartMarketOOPS is running!"
echo ""
echo "📊 Access your trading system:"
echo "   Backend API: http://localhost:3001"
echo "   ML Service:  http://localhost:8000"
echo "   Grafana:     http://localhost:3002 (admin/admin123)"
echo "   Prometheus:  http://localhost:9090"
echo ""
echo "💰 Trading Configuration:"
echo "   Exchange:     Delta Exchange India Testnet"
echo "   API Key:      ${DELTA_API_KEY:0:10}..."
echo "   Max Position: \$50"
echo "   Risk per Trade: 1.5%"
echo ""
echo "📋 Management Commands:"
echo "   Stop system:  ./stop-simple.sh"
echo "   View logs:    tail -f logs/*.log"
echo "   Test ML:      curl http://localhost:8000/health"
echo "   Test API:     curl http://localhost:3001/api/health"
echo ""
print_warning "🔥 TESTNET MODE: No real money will be used"
print_warning "📊 Monitor the system closely"
echo ""
echo "🚀 Your AI trading bot is now running! 💰"