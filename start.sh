#!/bin/bash

# SmartMarketOOPS Quick Start Script
# Comprehensive system startup with compatibility checks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    log "🔍 Checking system dependencies..."
    
    # Check Python
    if command_exists python3; then
        success "✅ Python3 available"
    else
        error "❌ Python3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check Node.js
    if command_exists node; then
        success "✅ Node.js available"
    else
        warning "⚠️ Node.js not found. Frontend may not work properly."
    fi
    
    # Check npm
    if command_exists npm; then
        success "✅ npm available"
    else
        warning "⚠️ npm not found. Cannot install Node.js dependencies."
    fi
    
    # Check Docker
    if command_exists docker; then
        success "✅ Docker available"
    else
        warning "⚠️ Docker not found. Will run in local mode."
    fi
    
    # Check Docker Compose
    if command_exists docker-compose; then
        success "✅ Docker Compose available"
    else
        warning "⚠️ Docker Compose not found. Cannot start infrastructure services."
    fi
}

# Setup environment
setup_environment() {
    log "🔧 Setting up environment..."
    
    # Create necessary directories
    mkdir -p logs data models config temp
    success "✅ Created necessary directories"
    
    # Check .env file
    if [ ! -f .env ]; then
        warning "⚠️ .env file not found. Creating basic configuration..."
        cat > .env << EOF
# SmartMarketOOPS Environment Configuration
NODE_ENV=development
PORT=8001
FRONTEND_PORT=3000
BACKEND_PORT=3002
DATABASE_URL=postgresql://postgres:password@localhost:5432/smartmarket
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
DELTA_EXCHANGE_TESTNET=true
EOF
        success "✅ Created basic .env file"
    fi
}

# Install Python dependencies
install_python_deps() {
    log "📦 Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    success "✅ Python dependencies installed"
}

# Install Node.js dependencies
install_node_deps() {
    log "📦 Installing Node.js dependencies..."
    
    # Install root dependencies
    if [ -f package.json ]; then
        npm install
        success "✅ Root Node.js dependencies installed"
    fi
    
    # Install frontend dependencies
    if [ -f frontend/package.json ]; then
        cd frontend
        npm install
        cd ..
        success "✅ Frontend dependencies installed"
    fi
    
    # Install backend dependencies
    if [ -f backend/package.json ]; then
        cd backend
        npm install
        cd ..
        success "✅ Backend dependencies installed"
    fi
}

# Start infrastructure services
start_infrastructure() {
    log "🚀 Starting infrastructure services..."
    
    if command_exists docker-compose; then
        docker-compose up -d postgres redis
        success "✅ Infrastructure services started"
        
        # Wait for services to be ready
        log "⏳ Waiting for services to be ready..."
        sleep 10
    else
        warning "⚠️ Docker Compose not available. Skipping infrastructure startup."
    fi
}

# Start the complete system
start_system() {
    log "🚀 Starting SmartMarketOOPS Complete System..."
    
    check_dependencies
    setup_environment
    install_python_deps
    install_node_deps
    start_infrastructure
    
    # Start the Python ML system
    log "🤖 Starting ML system..."
    source venv/bin/activate
    python3 main.py &
    ML_PID=$!

    # Start backend
    log "🔧 Starting backend..."
    cd backend
    npm run start:ts &
    BACKEND_PID=$!
    cd ..

    # Start frontend
    log "🎨 Starting frontend..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..

    success "🎉 SmartMarketOOPS system startup completed!"
    log "📊 Access points:"
    log "   - ML System: http://localhost:3002"
    log "   - Backend API: http://localhost:3001"
    log "   - Frontend: http://localhost:3000"
    log "   - API Docs: http://localhost:3002/docs"
    log ""
    log "Press Ctrl+C to stop the system"

    # Wait for all processes
    wait $ML_PID $BACKEND_PID $FRONTEND_PID
}

# Stop the system
stop_system() {
    log "🛑 Stopping SmartMarketOOPS system..."

    # Kill all related processes
    pkill -f "main.py" || true
    pkill -f "npm run dev" || true
    pkill -f "npm run start" || true
    pkill -f "ts-node" || true
    pkill -f "next dev" || true

    # Stop Docker services
    if command_exists docker-compose; then
        docker-compose down
        success "✅ Infrastructure services stopped"
    fi

    success "✅ System stopped"
}

# Handle script arguments
case "${1:-start}" in
    start)
        start_system
        ;;
    stop)
        stop_system
        ;;
    restart)
        stop_system
        sleep 2
        start_system
        ;;
    deps)
        check_dependencies
        ;;
    setup)
        setup_environment
        install_python_deps
        install_node_deps
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|deps|setup}"
        echo "  start   - Start the complete system (default)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart the system"
        echo "  deps    - Check dependencies only"
        echo "  setup   - Setup environment and install dependencies only"
        exit 1
        ;;
esac
