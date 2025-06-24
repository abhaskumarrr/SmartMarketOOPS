#!/bin/bash

# SmartMarketOOPS Local Stop Script
# Safely stops all services and preserves data

set -e

echo "🛑 Stopping SmartMarketOOPS Personal Trading Bot..."

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

# Check if any positions are open
print_status "Checking for open positions..."

# Try to get position count from API
if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
    OPEN_POSITIONS=$(curl -s http://localhost:3001/api/positions/count 2>/dev/null || echo "unknown")
    if [ "$OPEN_POSITIONS" != "0" ] && [ "$OPEN_POSITIONS" != "unknown" ]; then
        print_warning "⚠️  You have $OPEN_POSITIONS open position(s)!"
        print_warning "Stopping the system will not close your positions on Delta Exchange."
        read -p "Do you want to continue? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled. Your trading bot is still running."
            exit 1
        fi
    fi
fi

# Graceful shutdown sequence
print_status "Stopping frontend..."
docker-compose -f docker-compose.local.yml stop frontend

print_status "Stopping backend API (allowing current trades to complete)..."
docker-compose -f docker-compose.local.yml stop backend

print_status "Stopping ML service..."
docker-compose -f docker-compose.local.yml stop ml-service

print_status "Stopping monitoring services..."
docker-compose -f docker-compose.local.yml stop monitoring grafana

print_status "Stopping Redis cache..."
docker-compose -f docker-compose.local.yml stop redis

print_status "Stopping PostgreSQL database..."
docker-compose -f docker-compose.local.yml stop postgres

# Remove containers but keep volumes (preserve data)
print_status "Removing containers (keeping data)..."
docker-compose -f docker-compose.local.yml down --remove-orphans

print_success "✅ SmartMarketOOPS has been stopped successfully!"
echo ""
echo "💾 Your data has been preserved:"
echo "   - Database data in Docker volume"
echo "   - Logs in ./logs directory"
echo "   - Backups in ./backups directory"
echo ""
echo "🔄 To restart: ./start-local.sh"
echo "🧹 To clean everything: ./clean-local.sh"
echo ""
print_warning "Note: Your positions on Delta Exchange remain active!"
print_warning "Monitor them manually or restart the bot to continue automated trading."