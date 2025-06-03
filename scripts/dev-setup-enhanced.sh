#!/bin/bash
# Enhanced Development Setup for SmartMarketOOPS
# Optimized for M2 MacBook Air 8GB with memory-efficient patterns

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="SmartMarketOOPS"
NODE_VERSION="20.15.1"
PYTHON_VERSION="3.11"
MEMORY_LIMIT="1024" # MB per Node.js process

echo -e "${BLUE}🚀 Setting up ${PROJECT_NAME} development environment...${NC}"

# Function to check system resources
check_system_resources() {
    echo -e "${BLUE}📊 Checking system resources...${NC}"
    
    # Check available memory
    if [[ "$OSTYPE" == "darwin"* ]]; then
        TOTAL_MEM=$(sysctl -n hw.memsize)
        TOTAL_MEM_GB=$((TOTAL_MEM / 1024 / 1024 / 1024))
        echo -e "${GREEN}✅ Total Memory: ${TOTAL_MEM_GB}GB${NC}"
        
        if [ $TOTAL_MEM_GB -lt 8 ]; then
            echo -e "${YELLOW}⚠️  Warning: Less than 8GB RAM detected. Consider using cloud development.${NC}"
        fi
    fi
    
    # Check Node.js version
    if command -v node &> /dev/null; then
        NODE_CURRENT=$(node --version)
        echo -e "${GREEN}✅ Node.js: ${NODE_CURRENT}${NC}"
    else
        echo -e "${RED}❌ Node.js not found. Please install Node.js ${NODE_VERSION}${NC}"
        exit 1
    fi
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_CURRENT=$(python3 --version)
        echo -e "${GREEN}✅ Python: ${PYTHON_CURRENT}${NC}"
    else
        echo -e "${RED}❌ Python3 not found. Please install Python ${PYTHON_VERSION}${NC}"
        exit 1
    fi
}

# Function to setup environment variables
setup_environment() {
    echo -e "${BLUE}🔧 Setting up environment variables...${NC}"
    
    # Create .env files if they don't exist
    if [ ! -f .env ]; then
        echo -e "${YELLOW}📝 Creating root .env file...${NC}"
        cat > .env << EOF
# SmartMarketOOPS Environment Configuration
NODE_ENV=development
LOG_LEVEL=info

# Memory optimization for M2 MacBook Air 8GB
NODE_OPTIONS=--max-old-space-size=${MEMORY_LIMIT}
UV_THREADPOOL_SIZE=4

# Development URLs
CLIENT_URL=http://localhost:3000
API_URL=http://localhost:3001
ML_SERVICE_URL=http://localhost:3002

# Database
DATABASE_URL="postgresql://smoops:smoops_dev@localhost:5432/smoops_dev"

# JWT Configuration (Enhanced Security)
JWT_SECRET=$(openssl rand -base64 32)
JWT_REFRESH_SECRET=$(openssl rand -base64 32)
JWT_EXPIRY=15m
JWT_REFRESH_EXPIRY=7d

# Encryption
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Email Configuration (Development)
EMAIL_FROM=noreply@smartmarketoops.dev
EMAIL_HOST=localhost
EMAIL_PORT=1025
EMAIL_USER=
EMAIL_PASS=

# Delta Exchange (Testnet)
DELTA_API_KEY=your_testnet_api_key
DELTA_API_SECRET=your_testnet_api_secret
DELTA_BASE_URL=https://testnet-api.delta.exchange

# Redis (Optional for production)
REDIS_URL=redis://localhost:6379

# Development flags
ENABLE_CORS=true
ENABLE_RATE_LIMITING=true
ENABLE_CSRF=true
ENABLE_HELMET=true
EOF
        echo -e "${GREEN}✅ Root .env file created${NC}"
    fi
    
    # Backend .env
    if [ ! -f backend/.env ]; then
        echo -e "${YELLOW}📝 Creating backend .env file...${NC}"
        cp .env backend/.env
        echo -e "${GREEN}✅ Backend .env file created${NC}"
    fi
    
    # Frontend .env.local
    if [ ! -f frontend/.env.local ]; then
        echo -e "${YELLOW}📝 Creating frontend .env.local file...${NC}"
        cat > frontend/.env.local << EOF
# Frontend Environment Variables
NEXT_PUBLIC_API_URL=http://localhost:3001
NEXT_PUBLIC_WS_URL=ws://localhost:3001
NEXT_PUBLIC_ML_SERVICE_URL=http://localhost:3002
NEXT_PUBLIC_APP_NAME=SmartMarketOOPS
NEXT_PUBLIC_APP_VERSION=1.0.0

# Development flags
NEXT_PUBLIC_ENABLE_DEVTOOLS=true
NEXT_PUBLIC_LOG_LEVEL=debug

# Memory optimization
NODE_OPTIONS=--max-old-space-size=${MEMORY_LIMIT}
EOF
        echo -e "${GREEN}✅ Frontend .env.local file created${NC}"
    fi
    
    # ML service .env
    if [ ! -f ml/.env ]; then
        echo -e "${YELLOW}📝 Creating ML service .env file...${NC}"
        cat > ml/.env << EOF
# ML Service Environment Variables
ENVIRONMENT=development
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=3002

# Model Configuration
MODEL_PATH=./models
ENABLE_GPU=false
BATCH_SIZE=32
MAX_SEQUENCE_LENGTH=100

# Memory optimization for M2 MacBook Air
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Database
DATABASE_URL=postgresql://smoops:smoops_dev@localhost:5432/smoops_dev

# External APIs
DELTA_API_URL=https://testnet-api.delta.exchange
EOF
        echo -e "${GREEN}✅ ML service .env file created${NC}"
    fi
}

# Function to install dependencies with memory optimization
install_dependencies() {
    echo -e "${BLUE}📦 Installing dependencies with memory optimization...${NC}"
    
    # Root dependencies
    echo -e "${YELLOW}Installing root dependencies...${NC}"
    NODE_OPTIONS="--max-old-space-size=${MEMORY_LIMIT}" npm install
    
    # Backend dependencies
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    cd backend
    NODE_OPTIONS="--max-old-space-size=${MEMORY_LIMIT}" npm install
    cd ..
    
    # Frontend dependencies
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd frontend
    NODE_OPTIONS="--max-old-space-size=${MEMORY_LIMIT}" npm install
    cd ..
    
    # ML service dependencies
    echo -e "${YELLOW}Installing ML service dependencies...${NC}"
    cd ml
    if [ -f requirements.txt ]; then
        python3 -m pip install -r requirements.txt
    fi
    cd ..
    
    echo -e "${GREEN}✅ All dependencies installed${NC}"
}

# Function to setup database
setup_database() {
    echo -e "${BLUE}🗄️  Setting up database...${NC}"
    
    # Check if PostgreSQL is running
    if ! pgrep -x "postgres" > /dev/null; then
        echo -e "${YELLOW}⚠️  PostgreSQL not running. Please start PostgreSQL first.${NC}"
        echo -e "${BLUE}💡 Tip: brew services start postgresql@14${NC}"
        return 1
    fi
    
    # Create database if it doesn't exist
    createdb smoops_dev 2>/dev/null || echo -e "${YELLOW}Database smoops_dev already exists${NC}"
    
    # Run Prisma migrations
    cd backend
    npx prisma generate
    npx prisma migrate dev --name init
    cd ..
    
    echo -e "${GREEN}✅ Database setup complete${NC}"
}

# Function to create development scripts
create_dev_scripts() {
    echo -e "${BLUE}📝 Creating development scripts...${NC}"
    
    # Memory-efficient development script
    cat > scripts/dev-memory-efficient.sh << 'EOF'
#!/bin/bash
# Memory-efficient development server

export NODE_OPTIONS="--max-old-space-size=1024"
export UV_THREADPOOL_SIZE=4

echo "🚀 Starting memory-efficient development environment..."

# Start services with memory limits
concurrently \
  --names "BACKEND,FRONTEND,ML" \
  --prefix-colors "blue,green,yellow" \
  "cd backend && NODE_OPTIONS='--max-old-space-size=512' npm run dev" \
  "cd frontend && NODE_OPTIONS='--max-old-space-size=512' npm run dev" \
  "cd ml && python -m src.api.app"
EOF
    
    chmod +x scripts/dev-memory-efficient.sh
    
    # Quick test script
    cat > scripts/quick-test.sh << 'EOF'
#!/bin/bash
# Quick test runner for development

echo "🧪 Running quick tests..."

# Backend tests
echo "Testing backend..."
cd backend && npm test -- --maxWorkers=2 --forceExit
cd ..

# Frontend tests
echo "Testing frontend..."
cd frontend && npm test -- --watchAll=false --maxWorkers=2
cd ..

echo "✅ Quick tests complete"
EOF
    
    chmod +x scripts/quick-test.sh
    
    echo -e "${GREEN}✅ Development scripts created${NC}"
}

# Function to setup Git hooks
setup_git_hooks() {
    echo -e "${BLUE}🔗 Setting up Git hooks...${NC}"
    
    # Pre-commit hook for code quality
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for SmartMarketOOPS

echo "🔍 Running pre-commit checks..."

# Check for memory-efficient patterns
if git diff --cached --name-only | grep -E '\.(js|ts|tsx)$' | xargs grep -l 'new Array\|new Object' 2>/dev/null; then
    echo "⚠️  Warning: Consider using literal syntax instead of new Array() or new Object()"
fi

# Run linting on staged files
npm run lint:fix

echo "✅ Pre-commit checks complete"
EOF
    
    chmod +x .git/hooks/pre-commit
    
    echo -e "${GREEN}✅ Git hooks setup complete${NC}"
}

# Function to display development tips
show_dev_tips() {
    echo -e "${BLUE}💡 Development Tips for M2 MacBook Air 8GB:${NC}"
    echo -e "${GREEN}1. Use memory-efficient development server: ./scripts/dev-memory-efficient.sh${NC}"
    echo -e "${GREEN}2. Monitor memory usage: Activity Monitor or htop${NC}"
    echo -e "${GREEN}3. Use local development server: ./scripts/local_dev_server.sh start${NC}"
    echo -e "${GREEN}4. Run quick tests: ./scripts/quick-test.sh${NC}"
    echo -e "${GREEN}5. For heavy ML training, consider cloud platforms${NC}"
    echo -e "${GREEN}6. Use Docker for production-like testing: npm run docker:up${NC}"
    echo ""
    echo -e "${BLUE}🚀 Ready to start development!${NC}"
    echo -e "${YELLOW}Run: npm run dev (or ./scripts/dev-memory-efficient.sh for optimized)${NC}"
}

# Main execution
main() {
    check_system_resources
    setup_environment
    install_dependencies
    setup_database
    create_dev_scripts
    setup_git_hooks
    show_dev_tips
    
    echo -e "${GREEN}🎉 Development environment setup complete!${NC}"
}

# Run main function
main "$@"
