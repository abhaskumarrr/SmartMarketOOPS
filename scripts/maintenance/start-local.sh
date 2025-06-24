#!/bin/bash

# SmartMarketOOPS Local Startup Script
# Optimized for MacBook Air M2 8GB RAM

set -e

echo "🚀 Starting SmartMarketOOPS Personal Trading Bot on MacBook Air M2..."

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

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check available memory
AVAILABLE_MEMORY=$(sysctl hw.memsize | awk '{print $2/1024/1024/1024}')
print_status "Available system memory: ${AVAILABLE_MEMORY}GB"

if (( $(echo "$AVAILABLE_MEMORY < 8" | bc -l) )); then
    print_warning "Less than 8GB RAM detected. Performance may be affected."
fi

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    print_warning ".env.local not found. Creating from template..."
    cp .env.local.template .env.local
    print_warning "Please edit .env.local with your Delta Exchange API credentials before continuing."
    read -p "Press Enter to continue after editing .env.local..."
fi

# Load environment variables
if [ -f ".env.local" ]; then
    export $(cat .env.local | grep -v '^#' | xargs)
fi

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
mkdir -p monitoring/data

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose -f docker-compose.local.yml down --remove-orphans > /dev/null 2>&1 || true

# Pull latest images
print_status "Pulling latest base images..."
docker-compose -f docker-compose.local.yml pull --quiet

# Build custom images
print_status "Building SmartMarketOOPS images for ARM64..."
docker-compose -f docker-compose.local.yml build --parallel

# Start the database first
print_status "Starting PostgreSQL database..."
docker-compose -f docker-compose.local.yml up -d postgres
sleep 10

# Wait for database to be ready
print_status "Waiting for database to be ready..."
until docker-compose -f docker-compose.local.yml exec -T postgres pg_isready -U smartmarket > /dev/null 2>&1; do
    echo -n "."
    sleep 2
done
echo ""
print_success "Database is ready!"

# Run database migrations
print_status "Running database migrations..."
cd backend
npm run prisma:migrate:deploy
npm run prisma:generate
cd ..

# Start Redis
print_status "Starting Redis cache..."
docker-compose -f docker-compose.local.yml up -d redis
sleep 5

# Start ML service
print_status "Starting ML service..."
docker-compose -f docker-compose.local.yml up -d ml-service
sleep 10

# Start backend API
print_status "Starting backend API..."
docker-compose -f docker-compose.local.yml up -d backend
sleep 10

# Start frontend
print_status "Starting frontend dashboard..."
docker-compose -f docker-compose.local.yml up -d frontend
sleep 5

# Start monitoring
print_status "Starting monitoring stack..."
docker-compose -f docker-compose.local.yml up -d monitoring grafana

# Wait for all services to be healthy
print_status "Waiting for all services to be healthy..."
sleep 30

# Check service health
print_status "Checking service health..."

services=("postgres" "redis" "backend" "ml-service" "frontend" "monitoring" "grafana")
all_healthy=true

for service in "${services[@]}"; do
    if docker-compose -f docker-compose.local.yml ps | grep -q "$service.*healthy\|$service.*Up"; then
        print_success "$service is healthy"
    else
        print_error "$service is not healthy"
        all_healthy=false
    fi
done

if [ "$all_healthy" = true ]; then
    print_success "🎉 SmartMarketOOPS is running successfully!"
    echo ""
    echo "📊 Access your trading dashboard:"
    echo "   Frontend:    http://localhost:3000"
    echo "   API:         http://localhost:3001"
    echo "   Grafana:     http://localhost:3002 (admin/admin123)"
    echo "   Prometheus:  http://localhost:9090"
    echo ""
    echo "💰 Trading Configuration:"
    echo "   Initial Capital: \$${INITIAL_CAPITAL:-1000}"
    echo "   Max Position:    \$${MAX_POSITION_SIZE:-100}"
    echo "   Risk per Trade:  ${RISK_PER_TRADE:-2}%"
    echo "   Max Positions:   ${MAX_POSITIONS:-3}"
    echo ""
    echo "🔧 To stop the system: ./stop-local.sh"
    echo "📊 To view logs: docker-compose -f docker-compose.local.yml logs -f"
    echo "🔄 To restart: ./restart-local.sh"
    echo ""
    print_warning "Remember: This is connected to your REAL Delta Exchange account!"
    print_warning "Start with paper trading mode to test the system first."
else
    print_error "Some services are not healthy. Check the logs:"
    echo "docker-compose -f docker-compose.local.yml logs"
    exit 1
fi