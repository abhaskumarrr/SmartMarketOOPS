#!/bin/bash

# Consolidated Setup Script for SmartMarketOOPS
# This script combines functionality from:
# - setup.sh
# - setup-infrastructure.sh
# - validate-docker-setup.sh
# - docker-dev.sh

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    local message=$2
    local color=""
    
    case $level in
        "INFO") color=$GREEN ;;
        "WARN") color=$YELLOW ;;
        "ERROR") color=$RED ;;
    esac
    
    echo -e "${color}[${level}] ${message}${NC}"
}

# Check system requirements
check_requirements() {
    log "INFO" "Checking system requirements..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    if [[ $(echo $PYTHON_VERSION | cut -d. -f1,2) < "3.10" ]]; then
        log "ERROR" "Python 3.10+ required. Found $PYTHON_VERSION"
        exit 1
    fi
    
    # Check Node.js version
    if ! command -v node &> /dev/null; then
        log "ERROR" "Node.js is not installed"
        exit 1
    fi
    NODE_VERSION=$(node --version | cut -c2-)
    if [[ $(echo $NODE_VERSION | cut -d. -f1) < "18" ]]; then
        log "ERROR" "Node.js 18+ required. Found $NODE_VERSION"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log "ERROR" "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "ERROR" "Docker Compose is not installed"
        exit 1
    fi
}

# Setup Python environment
setup_python() {
    log "INFO" "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
}

# Setup Node.js environment
setup_node() {
    log "INFO" "Setting up Node.js environment..."
    
    # Install dependencies
    cd frontend
    npm install
    cd ..
}

# Setup Docker environment
setup_docker() {
    log "INFO" "Setting up Docker environment..."
    
    # Build development containers
    docker-compose -f docker-compose.dev.yml build
    
    # Check if containers built successfully
    if [ $? -ne 0 ]; then
        log "ERROR" "Failed to build Docker containers"
        exit 1
    fi
}

# Setup infrastructure
setup_infrastructure() {
    log "INFO" "Setting up infrastructure..."
    
    # Create necessary directories
    mkdir -p data/{raw,processed,backtesting,validation}
    mkdir -p models/registry
    mkdir -p logs
    
    # Setup monitoring
    if [ -d "monitoring" ]; then
        cd monitoring
        docker-compose up -d
        cd ..
    fi
    
    # Initialize database
    if [ -f "backend/prisma/schema.prisma" ]; then
        cd backend
        npx prisma generate
        npx prisma migrate deploy
        cd ..
    fi
}

# Validate setup
validate_setup() {
    log "INFO" "Validating setup..."
    
    # Run validation script
    python scripts/validate_system.py
    
    if [ $? -ne 0 ]; then
        log "ERROR" "Setup validation failed"
        exit 1
    fi
}

# Main setup process
main() {
    log "INFO" "Starting setup process..."
    
    check_requirements
    setup_python
    setup_node
    setup_docker
    setup_infrastructure
    validate_setup
    
    log "INFO" "Setup completed successfully!"
    log "INFO" "You can now start the development server with: ./scripts/start.sh"
}

# Run main function
main