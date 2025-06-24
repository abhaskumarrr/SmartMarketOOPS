#!/bin/bash

# SmartMarketOOPS Simple Startup Script
# Runs infrastructure in Docker, services locally

set -e

echo "🚀 Starting SmartMarketOOPS (Simple Mode)..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    print_error ".env.local not found. Please create it with your Delta Exchange credentials."
    exit 1
fi

# Load environment variables
export $(cat .env.local | grep -v '^#' | xargs)

# Check for required environment variables
if [ -z "$DELTA_API_KEY" ] || [ "$DELTA_API_KEY" = "your_delta_api_key_here" ]; then
    print_error "Please set your Delta Exchange API key in .env.local"
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p backups
mkdir -p backend/logs
mkdir -p ml/logs

# Start infrastructure services with Docker
print_status "Starting infrastructure services (Docker)..."
docker-compose -f docker-compose.simple.yml down --remove-orphans > /dev/null 2>&1 || true
docker-compose -f docker-compose.simple.yml up -d

# Wait for database to be ready
print_status "Waiting for database to be ready..."
TIMEOUT=60
COUNTER=0
until docker-compose -f docker-compose.simple.yml exec -T postgres pg_isready -U smartmarket > /dev/null 2>&1; do
    echo -n "."
    sleep 2
    COUNTER=$((COUNTER + 2))
    if [ $COUNTER -ge $TIMEOUT ]; then
        echo ""
        print_error "Database failed to start within $TIMEOUT seconds"
        print_status "Checking database status..."
        docker-compose -f docker-compose.simple.yml ps postgres
        docker-compose -f docker-compose.simple.yml logs postgres --tail 10
        exit 1
    fi
done
echo ""
print_success "Database is ready!"

# Install backend dependencies if needed
if [ ! -d "backend/node_modules" ]; then
    print_status "Installing backend dependencies..."
    cd backend
    npm install
    cd ..
fi

# Run database migrations
print_status "Running database migrations..."
cd backend
npm run prisma:migrate:deploy > /dev/null 2>&1 || npm run prisma:db:push > /dev/null 2>&1 || true
npm run prisma:generate > /dev/null 2>&1 || true
cd ..

# Install ML dependencies if needed
if [ ! -f "ml/.venv/bin/activate" ]; then
    print_status "Setting up ML environment..."
    cd ml
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt > /dev/null 2>&1 || true
    cd ..
fi

# Start ML service in background
print_status "Starting ML service..."
cd ml
source .venv/bin/activate
export PYTHONPATH=$(pwd)
nohup python simple_app.py > ../logs/ml-service.log 2>&1 &
ML_PID=$!
echo $ML_PID > ../logs/ml-service.pid
cd ..
print_success "ML service started (PID: $ML_PID)"

# Wait a bit for ML service to start
sleep 5

# Start backend service in background
print_status "Starting backend API..."
cd backend
export NODE_ENV=development
export DATABASE_URL="postgresql://smartmarket:secure_local_password_2024@localhost:5432/smartmarket"
export REDIS_URL="redis://localhost:6379"
nohup npm run dev > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../logs/backend.pid
cd ..
print_success "Backend API started (PID: $BACKEND_PID)"

# Wait a bit for backend to start
sleep 10

# Start frontend in background
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    print_status "Starting frontend..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        npm install > /dev/null 2>&1
    fi
    export NEXT_PUBLIC_API_URL="http://localhost:3001/api"
    nohup npm run dev > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../logs/frontend.pid
    cd ..
    print_success "Frontend started (PID: $FRONTEND_PID)"
else
    print_warning "Frontend not found, skipping..."
fi

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 15

# Check service health
print_status "Checking service health..."

# Check ML service
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "ML service is healthy"
else
    print_warning "ML service may not be ready yet"
fi

# Check backend service
if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
    print_success "Backend API is healthy"
else
    print_warning "Backend API may not be ready yet"
fi

# Check frontend service
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend is healthy"
else
    print_warning "Frontend may not be ready yet"
fi

print_success "🎉 SmartMarketOOPS is running!"
echo ""
echo "📊 Access your trading system:"
echo "   Frontend:    http://localhost:3000"
echo "   Backend API: http://localhost:3001"
echo "   ML Service:  http://localhost:8000"
echo "   Grafana:     http://localhost:3002 (admin/admin123)"
echo "   Prometheus:  http://localhost:9090"
echo ""
echo "💰 Trading Configuration:"
echo "   Exchange:     Delta Exchange India Testnet"
echo "   Initial Capital: \$${INITIAL_CAPITAL:-1000}"
echo "   Max Position:    \$${MAX_POSITION_SIZE:-50}"
echo "   Risk per Trade:  ${RISK_PER_TRADE:-1.5}%"
echo ""
echo "📋 Management Commands:"
echo "   Stop system:  ./stop-simple.sh"
echo "   View logs:    tail -f logs/*.log"
echo "   Restart:      ./restart-simple.sh"
echo ""
print_warning "🔥 TESTNET MODE: No real money will be used"
print_warning "📊 Monitor the system closely for the first few hours"
echo ""
echo "🚀 Your AI trading bot is now running! Happy trading! 💰"