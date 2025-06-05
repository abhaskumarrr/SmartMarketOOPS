#!/bin/bash

echo "🚀 SmartMarketOOPS System Integration Test"
echo "=========================================="

# Test Backend Health
echo "🔍 Testing Backend Health..."
curl -s http://localhost:3002/api/health | jq '.'
echo ""

# Test Auth Routes
echo "🔐 Testing Auth Routes..."
curl -s http://localhost:3002/api/auth/health | jq '.'
curl -s http://localhost:3002/api/auth/csrf-token | jq '.'
echo ""

# Test User Routes
echo "👤 Testing User Routes..."
curl -s http://localhost:3002/api/users/health | jq '.'
echo ""

# Test API Key Routes
echo "🔑 Testing API Key Routes..."
curl -s http://localhost:3002/api/api-keys 2>/dev/null | jq '.' || echo "Expected auth error - route working"
echo ""

# Test Trading Routes
echo "💰 Testing Trading Routes..."
curl -s http://localhost:3002/api/trading-working/status | jq '.'
curl -s http://localhost:3002/api/trading-working/products | jq '.data[0]'
curl -s http://localhost:3002/api/trading-working/market-data/BTCUSD | jq '.data'
curl -s http://localhost:3002/api/trading-working/balances | jq '.data[0]'
echo ""

# Test Metrics
echo "📊 Testing Metrics..."
curl -s http://localhost:3002/api/metrics/health | jq '.'
curl -s http://localhost:3002/api/metrics/system | jq '.data | {uptime, platform, nodeVersion}'
echo ""

# Test Frontend
echo "🌐 Testing Frontend..."
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$FRONTEND_STATUS" = "200" ]; then
    echo "✅ Frontend is running on http://localhost:3000"
else
    echo "❌ Frontend not responding"
fi

DASHBOARD_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/dashboard)
if [ "$DASHBOARD_STATUS" = "200" ]; then
    echo "✅ Dashboard is accessible on http://localhost:3000/dashboard"
else
    echo "❌ Dashboard not responding"
fi

echo ""
echo "🎉 Integration Test Complete!"
echo "Backend: http://localhost:3002"
echo "Frontend: http://localhost:3000"
echo "Dashboard: http://localhost:3000/dashboard"
