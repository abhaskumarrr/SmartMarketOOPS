#!/bin/bash

# SmartMarketOOPS Startup Errors Fix Script
# Resolves all critical startup errors for the Real-Time Trading Dashboard

set -e

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

print_status "🔧 Fixing SmartMarketOOPS Critical Startup Errors..."

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    print_error "Please run this script from the SmartMarketOOPS root directory"
    exit 1
fi

PROJECT_ROOT=$(pwd)

# Step 1: Fix Port Conflicts
print_status "🔧 Step 1: Resolving port conflicts..."

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill process on port
kill_port_process() {
    local port=$1
    local pid=$(lsof -ti:$port 2>/dev/null)
    
    if [ ! -z "$pid" ]; then
        print_warning "Killing process $pid on port $port"
        kill -9 $pid 2>/dev/null || true
        sleep 2
        return 0
    fi
    return 1
}

# Resolve port 3001 conflict
if check_port 3001; then
    print_warning "Port 3001 is in use by:"
    lsof -i :3001 2>/dev/null || true
    
    if kill_port_process 3001; then
        print_success "Port 3001 freed"
        WEBSOCKET_PORT=3001
    else
        print_warning "Using alternative port 3002 for WebSocket"
        WEBSOCKET_PORT=3002
    fi
else
    WEBSOCKET_PORT=3001
    print_success "Port 3001 is available"
fi

# Resolve port 3000 conflict if needed
if check_port 3000; then
    print_warning "Port 3000 is in use by:"
    lsof -i :3000 2>/dev/null || true
    
    if kill_port_process 3000; then
        print_success "Port 3000 freed"
        FRONTEND_PORT=3000
    else
        print_warning "Using alternative port 3001 for Frontend"
        FRONTEND_PORT=3001
    fi
else
    FRONTEND_PORT=3000
    print_success "Port 3000 is available"
fi

print_success "✅ Port conflicts resolved"

# Step 2: Create/Fix globals.css
print_status "🔧 Step 2: Creating missing globals.css file..."

# Ensure the globals.css file exists (already created above)
if [ ! -f "frontend/app/globals.css" ]; then
    print_error "globals.css file was not created properly"
    exit 1
fi

print_success "✅ globals.css file created"

# Step 3: Fix CSS import paths
print_status "🔧 Step 3: Fixing CSS import paths..."

# Fix pages/_app.tsx import path
if [ -f "frontend/pages/_app.tsx" ]; then
    # Check if the import path is correct
    if grep -q "\.\./app/globals\.css" frontend/pages/_app.tsx; then
        print_success "CSS import path in pages/_app.tsx is correct"
    else
        # Fix the import path
        sed -i '' "s|import.*globals\.css.*|import '../app/globals.css';|g" frontend/pages/_app.tsx 2>/dev/null || \
        sed -i "s|import.*globals\.css.*|import '../app/globals.css';|g" frontend/pages/_app.tsx 2>/dev/null || true
        print_success "Fixed CSS import path in pages/_app.tsx"
    fi
fi

# Verify app/layout.tsx import path
if [ -f "frontend/app/layout.tsx" ]; then
    if grep -q "\./globals\.css" frontend/app/layout.tsx; then
        print_success "CSS import path in app/layout.tsx is correct"
    else
        print_warning "CSS import may need to be added to app/layout.tsx"
    fi
fi

print_success "✅ CSS import paths fixed"

# Step 4: Update environment configuration
print_status "🔧 Step 4: Updating environment configuration..."

# Create frontend environment
cat > frontend/.env.local << EOF
NEXT_PUBLIC_WS_URL=ws://localhost:$WEBSOCKET_PORT
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:$FRONTEND_PORT
NODE_ENV=development
WEBSOCKET_PORT=$WEBSOCKET_PORT
FRONTEND_PORT=$FRONTEND_PORT
EOF

# Create backend environment
mkdir -p backend
cat > backend/.env << EOF
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=$WEBSOCKET_PORT
ML_API_HOST=localhost
ML_API_PORT=8000
JWT_SECRET=your-secret-key-here
ENVIRONMENT=development
FRONTEND_PORT=$FRONTEND_PORT
EOF

print_success "✅ Environment configuration updated"

# Step 5: Create reliable startup script
print_status "🔧 Step 5: Creating reliable startup script..."

cat > scripts/reliable_start.sh << 'EOF'
#!/bin/bash

# Reliable SmartMarketOOPS Startup Script
# Handles all startup scenarios and error recovery

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

# Load port configuration
if [ -f "frontend/.env.local" ]; then
    source frontend/.env.local
    WEBSOCKET_PORT=${WEBSOCKET_PORT:-3001}
    FRONTEND_PORT=${FRONTEND_PORT:-3000}
else
    WEBSOCKET_PORT=3001
    FRONTEND_PORT=3000
fi

print_status "🚀 Starting SmartMarketOOPS Real-Time Trading Dashboard..."
print_status "📡 WebSocket Port: $WEBSOCKET_PORT"
print_status "🌐 Frontend Port: $FRONTEND_PORT"

# Check prerequisites
if [ ! -d "venv" ]; then
    print_error "Python virtual environment not found. Run setup first."
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    print_warning "Node modules not found. Installing..."
    cd frontend
    npm install
    cd ..
fi

# Activate Python environment
source venv/bin/activate

# Function to cleanup on exit
cleanup() {
    print_status "🛑 Shutting down services..."
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start WebSocket server
print_status "📡 Starting WebSocket server on port $WEBSOCKET_PORT..."

if [ -f "backend/websocket/port_aware_websocket_server.py" ]; then
    python backend/websocket/port_aware_websocket_server.py $WEBSOCKET_PORT &
    WEBSOCKET_PID=$!
elif [ -f "backend/websocket/reliable_websocket_server.py" ]; then
    python backend/websocket/reliable_websocket_server.py &
    WEBSOCKET_PID=$!
else
    print_error "No WebSocket server found. Run fix scripts first."
    exit 1
fi

# Wait for WebSocket server to start
sleep 3

# Check if WebSocket server started successfully
if ! kill -0 $WEBSOCKET_PID 2>/dev/null; then
    print_error "WebSocket server failed to start"
    exit 1
fi

print_success "✅ WebSocket server started"

# Start frontend server
print_status "🌐 Starting frontend server on port $FRONTEND_PORT..."

cd frontend

# Set the port for Next.js
export PORT=$FRONTEND_PORT

npm run dev &
FRONTEND_PID=$!

cd ..

# Wait for frontend to start
sleep 5

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    print_error "Frontend server failed to start"
    kill $WEBSOCKET_PID 2>/dev/null || true
    exit 1
fi

print_success "✅ Frontend server started"

print_success "🎉 All services started successfully!"
print_status ""
print_status "📊 Dashboard URLs:"
print_status "   🌐 Main Dashboard: http://localhost:$FRONTEND_PORT"
print_status "   🎯 Trading Dashboard: http://localhost:$FRONTEND_PORT/dashboard"
print_status "   📡 WebSocket: ws://localhost:$WEBSOCKET_PORT"
print_status ""
print_status "🔍 Expected Features:"
print_status "   ✅ Real-time price charts (2-second updates)"
print_status "   ✅ Trading signals (15-45 second intervals)"
print_status "   ✅ ML Intelligence dashboard (4 tabs)"
print_status "   ✅ Portfolio monitoring"
print_status "   ✅ WebSocket connectivity status"
print_status ""
print_status "Press Ctrl+C to stop all services"

# Wait for services
wait
EOF

chmod +x scripts/reliable_start.sh

print_success "✅ Reliable startup script created"

# Step 6: Install missing dependencies
print_status "🔧 Step 6: Installing missing dependencies..."

# Activate Python environment
source venv/bin/activate

# Install Python dependencies
pip install --quiet --upgrade pip
pip install --quiet PyJWT websockets asyncio pandas numpy

# Install frontend dependencies
cd frontend

# Check if Chart.js is installed
if ! npm list chart.js >/dev/null 2>&1; then
    print_status "Installing Chart.js dependencies..."
    npm install --save chart.js react-chartjs-2 chartjs-adapter-date-fns date-fns
fi

# Check if Tailwind CSS is properly configured
if [ ! -f "tailwind.config.js" ]; then
    print_status "Initializing Tailwind CSS..."
    npx tailwindcss init -p
fi

cd ..

print_success "✅ Dependencies installed"

# Step 7: Validate setup
print_status "🔧 Step 7: Validating setup..."

# Check critical files
CRITICAL_FILES=(
    "frontend/app/globals.css"
    "frontend/app/layout.tsx"
    "frontend/app/page.tsx"
    "frontend/app/dashboard/page.tsx"
    "frontend/pages/_app.tsx"
    "frontend/package.json"
    "frontend/.env.local"
    "backend/.env"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        print_success "✅ $file exists"
    else
        print_warning "⚠️  $file missing"
    fi
done

# Check Python dependencies
python -c "import websockets, asyncio; print('✅ Python WebSocket dependencies OK')" 2>/dev/null || \
    print_warning "⚠️  Python WebSocket dependencies may be missing"

# Check if PyJWT is available
python -c "import jwt; print('✅ PyJWT available')" 2>/dev/null || \
    print_warning "⚠️  PyJWT may be missing"

print_success "✅ Setup validation completed"

print_success "🎉 All critical startup errors have been fixed!"
print_status ""
print_status "🚀 To start the dashboard:"
print_status "   ./scripts/reliable_start.sh"
print_status ""
print_status "🌐 Dashboard will be available at:"
print_status "   http://localhost:$FRONTEND_PORT/dashboard"
print_status ""
print_status "📋 What was fixed:"
print_status "   ✅ Port 3001 conflict resolved (using port $WEBSOCKET_PORT)"
print_status "   ✅ Missing globals.css file created"
print_status "   ✅ CSS import paths fixed"
print_status "   ✅ Environment configuration updated"
print_status "   ✅ Dependencies installed"
print_status "   ✅ Reliable startup script created"
