#!/bin/bash

# SmartMarketOOPS Quick Setup Script
# Sets up the complete development environment

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "🚀 SmartMarketOOPS Quick Setup"
print_status "Setting up the complete development environment..."

# Check if we're in the right directory
if [ ! -d "frontend" ]; then
    print_error "Please run this script from the SmartMarketOOPS root directory"
    exit 1
fi

# Install frontend dependencies
print_status "📦 Installing frontend dependencies..."
cd frontend
npm install

# Go back to root
cd ..

# Create Python virtual environment
print_status "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
print_status "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install asyncio websockets pandas numpy torch scikit-learn PyJWT python-jose[cryptography] python-multipart fastapi uvicorn aiofiles python-dotenv

# Create necessary directories
print_status "📁 Creating project directories..."
mkdir -p backend/websocket
mkdir -p ml/src/intelligence
mkdir -p logs

# Create environment files
print_status "⚙️ Creating environment configuration..."

# Frontend environment
cat > frontend/.env.local << 'EOF'
NEXT_PUBLIC_WS_URL=ws://localhost:3001
NEXT_PUBLIC_ML_API_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:3000
NODE_ENV=development
EOF

# Backend environment
cat > backend/.env << 'EOF'
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=3001
ML_API_HOST=localhost
ML_API_PORT=8000
JWT_SECRET=your-secret-key-here
ENVIRONMENT=development
EOF

print_success "✅ Setup completed successfully!"
print_status ""
print_status "🎯 Next steps:"
print_status "1. Start the dashboard: ./scripts/launch_dashboard.sh --start"
print_status "2. Open browser: http://localhost:3000/dashboard"
print_status ""
print_status "📊 The dashboard will include:"
print_status "• Real-time price charts with live updates"
print_status "• ML Intelligence with regime analysis"
print_status "• Signal quality indicators"
print_status "• Portfolio monitoring"
print_status "• WebSocket connectivity status"
