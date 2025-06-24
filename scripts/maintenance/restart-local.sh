#!/bin/bash

# SmartMarketOOPS Local Restart Script
# Quickly restarts the system while preserving data

set -e

echo "🔄 Restarting SmartMarketOOPS Personal Trading Bot..."

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

# Quick restart without full rebuild
print_status "Stopping services..."
docker-compose -f docker-compose.local.yml down

print_status "Starting services..."
docker-compose -f docker-compose.local.yml up -d

print_status "Waiting for services to be ready..."
sleep 30

# Quick health check
if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
    print_success "✅ SmartMarketOOPS restarted successfully!"
    echo ""
    echo "📊 Dashboard: http://localhost:3000"
    echo "📈 Monitoring: http://localhost:3002"
else
    echo "❌ Services may not be fully ready yet. Check logs:"
    echo "docker-compose -f docker-compose.local.yml logs -f"
fi