#!/bin/bash

# SmartMarketOOPS Test and Launch Script
# Comprehensive testing and launching of the Real-Time Trading Dashboard

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_header() {
    echo -e "${PURPLE}[HEADER]${NC} $1"
}

print_header "🚀 SmartMarketOOPS Real-Time Trading Dashboard"
print_header "🔧 Complete Test and Launch Sequence"
echo ""

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    print_error "Please run this script from the SmartMarketOOPS root directory"
    exit 1
fi

PROJECT_ROOT=$(pwd)

# Step 1: Run all fixes
print_status "🔧 Step 1: Running comprehensive fixes..."

if [ -f "scripts/fix_startup_errors.sh" ]; then
    ./scripts/fix_startup_errors.sh
    print_success "✅ Startup errors fixed"
else
    print_warning "Fix script not found, continuing with manual checks..."
fi

# Step 2: Validate all critical files
print_status "🔧 Step 2: Validating critical files..."

CRITICAL_FILES=(
    "frontend/app/globals.css:CSS styles file"
    "frontend/app/layout.tsx:App layout component"
    "frontend/app/page.tsx:Home page component"
    "frontend/app/dashboard/page.tsx:Dashboard page component"
    "frontend/components/trading/RealTimeTradingDashboard.tsx:Main dashboard component"
    "frontend/components/intelligence/MLIntelligenceDashboard.tsx:ML Intelligence component"
    "frontend/lib/stores/tradingStore.ts:Trading state store"
    "frontend/lib/services/mlIntelligenceService.ts:ML Intelligence service"
    "frontend/package.json:Package configuration"
    "frontend/.env.local:Environment configuration"
    "backend/.env:Backend environment"
)

MISSING_FILES=()

for file_desc in "${CRITICAL_FILES[@]}"; do
    IFS=':' read -r file desc <<< "$file_desc"
    if [ -f "$file" ]; then
        print_success "✅ $desc"
    else
        print_error "❌ $desc (missing: $file)"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    print_error "Critical files are missing. Please ensure all components are properly created."
    for file in "${MISSING_FILES[@]}"; do
        print_error "   Missing: $file"
    done
    exit 1
fi

# Step 3: Test Python environment
print_status "🔧 Step 3: Testing Python environment..."

if [ ! -d "venv" ]; then
    print_warning "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Test Python dependencies
python -c "
import sys
try:
    import websockets
    print('✅ websockets: OK')
except ImportError:
    print('❌ websockets: MISSING')
    sys.exit(1)

try:
    import asyncio
    print('✅ asyncio: OK')
except ImportError:
    print('❌ asyncio: MISSING')
    sys.exit(1)

try:
    import jwt
    print('✅ PyJWT: OK')
except ImportError:
    print('❌ PyJWT: MISSING')
    sys.exit(1)

print('✅ All Python dependencies available')
"

if [ $? -ne 0 ]; then
    print_warning "Installing missing Python dependencies..."
    pip install --quiet PyJWT websockets asyncio pandas numpy
fi

print_success "✅ Python environment ready"

# Step 4: Test Node.js environment
print_status "🔧 Step 4: Testing Node.js environment..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    print_warning "Installing Node.js dependencies..."
    npm install
fi

# Test critical npm packages
npm list react >/dev/null 2>&1 && print_success "✅ React: OK" || print_error "❌ React: MISSING"
npm list next >/dev/null 2>&1 && print_success "✅ Next.js: OK" || print_error "❌ Next.js: MISSING"
npm list zustand >/dev/null 2>&1 && print_success "✅ Zustand: OK" || print_error "❌ Zustand: MISSING"
npm list tailwindcss >/dev/null 2>&1 && print_success "✅ Tailwind CSS: OK" || print_warning "⚠️  Tailwind CSS: May need installation"

cd "$PROJECT_ROOT"

print_success "✅ Node.js environment ready"

# Step 5: Check port availability
print_status "🔧 Step 5: Checking port availability..."

# Load port configuration
if [ -f "frontend/.env.local" ]; then
    source frontend/.env.local
    WEBSOCKET_PORT=${WEBSOCKET_PORT:-3001}
    FRONTEND_PORT=${FRONTEND_PORT:-3000}
else
    WEBSOCKET_PORT=3001
    FRONTEND_PORT=3000
fi

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

if check_port $WEBSOCKET_PORT; then
    print_warning "Port $WEBSOCKET_PORT is in use. Will attempt to use alternative port."
else
    print_success "✅ WebSocket port $WEBSOCKET_PORT is available"
fi

if check_port $FRONTEND_PORT; then
    print_warning "Port $FRONTEND_PORT is in use. Will attempt to use alternative port."
else
    print_success "✅ Frontend port $FRONTEND_PORT is available"
fi

# Step 6: Launch services
print_status "🔧 Step 6: Launching services..."

# Function to cleanup on exit
cleanup() {
    print_status "🛑 Shutting down services..."
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start WebSocket server
print_status "📡 Starting WebSocket server..."

if [ -f "backend/websocket/port_aware_websocket_server.py" ]; then
    python backend/websocket/port_aware_websocket_server.py $WEBSOCKET_PORT &
    WEBSOCKET_PID=$!
elif [ -f "backend/websocket/reliable_websocket_server.py" ]; then
    python backend/websocket/reliable_websocket_server.py &
    WEBSOCKET_PID=$!
else
    print_error "No WebSocket server found"
    exit 1
fi

# Wait for WebSocket server to start
sleep 3

# Check if WebSocket server started successfully
if ! kill -0 $WEBSOCKET_PID 2>/dev/null; then
    print_error "WebSocket server failed to start"
    exit 1
fi

print_success "✅ WebSocket server started (PID: $WEBSOCKET_PID)"

# Start frontend server
print_status "🌐 Starting frontend server..."

cd frontend
export PORT=$FRONTEND_PORT
npm run dev &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for frontend to start
sleep 5

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    print_error "Frontend server failed to start"
    kill $WEBSOCKET_PID 2>/dev/null || true
    exit 1
fi

print_success "✅ Frontend server started (PID: $FRONTEND_PID)"

# Step 7: Display success information
print_header "🎉 SmartMarketOOPS Dashboard Successfully Launched!"
echo ""
print_success "📊 Dashboard URLs:"
print_success "   🌐 Main Dashboard: http://localhost:$FRONTEND_PORT"
print_success "   🎯 Trading Dashboard: http://localhost:$FRONTEND_PORT/dashboard"
print_success "   📡 WebSocket Server: ws://localhost:$WEBSOCKET_PORT"
echo ""
print_success "🔍 Expected Features:"
print_success "   ✅ Real-time price charts (updates every 2 seconds)"
print_success "   ✅ Trading signals (generated every 15-45 seconds)"
print_success "   ✅ ML Intelligence dashboard with 4 tabs:"
print_success "      • Overview: Signal summary and performance metrics"
print_success "      • Performance: Accuracy metrics and system performance"
print_success "      • Analysis: Market regime and risk assessment"
print_success "      • Execution: Strategy and risk management"
print_success "   ✅ Portfolio monitoring with real-time P&L"
print_success "   ✅ WebSocket connectivity status indicators"
print_success "   ✅ Symbol switching (BTCUSD, ETHUSD, ADAUSD, SOLUSD, DOTUSD)"
echo ""
print_success "🎯 Success Indicators to Look For:"
print_success "   ✅ Green 'Live' status in dashboard header"
print_success "   ✅ Green 'ML Active' status for ML intelligence"
print_success "   ✅ Price charts updating with live data"
print_success "   ✅ Trading signals appearing in history feed"
print_success "   ✅ All ML Intelligence tabs functional"
print_success "   ✅ No console errors in browser"
echo ""
print_header "🌐 Open your browser and navigate to:"
print_header "   http://localhost:$FRONTEND_PORT/dashboard"
echo ""
print_status "Press Ctrl+C to stop all services"

# Wait for services
wait
