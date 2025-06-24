#!/bin/bash

# SmartMarketOOPS Startup Script
# Orchestrates the startup of all services in the correct order
# Usage: ./start.sh [dev|prod]

# Error handling
set -e
trap 'echo "Error on line $LINENO. Exiting..."; exit 1' ERR

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV=${1:-dev}
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/startup_$(date +%Y%m%d_%H%M%S).log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Logging function
log() {
  local level=$1
  local message=$2
  local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
  
  case $level in
    "INFO")
      echo -e "${GREEN}[INFO]${NC} $timestamp - $message"
      ;;
    "WARN")
      echo -e "${YELLOW}[WARN]${NC} $timestamp - $message"
      ;;
    "ERROR")
      echo -e "${RED}[ERROR]${NC} $timestamp - $message"
      ;;
    *)
      echo -e "$timestamp - $message"
      ;;
  esac
  
  echo "[$level] $timestamp - $message" >> "$LOG_FILE"
}

# Environment validation
validate_environment() {
  log "INFO" "Validating environment..."
  
  # Check if .env file exists
  if [ ! -f .env ]; then
    log "ERROR" ".env file not found. Please create one based on example.env"
    exit 1
  fi
  
  # Check for required tools
  commands=("python" "pip" "node" "npm" "docker" "docker-compose")
  for cmd in "${commands[@]}"; do
    if ! command -v $cmd &> /dev/null; then
      log "ERROR" "$cmd is not installed. Please install it before continuing."
      exit 1
    fi
  done
  
  # Check Python version
  PYTHON_VERSION=$(python --version | sed 's/Python //')
  if [[ $(echo "$PYTHON_VERSION" | cut -d. -f1) -lt 3 || ($(echo "$PYTHON_VERSION" | cut -d. -f1) -eq 3 && $(echo "$PYTHON_VERSION" | cut -d. -f2) -lt 10) ]]; then
    log "ERROR" "Python 3.10+ required. Found $PYTHON_VERSION"
    exit 1
  fi
  
  # Check Node.js version
  NODE_VERSION=$(node --version | sed 's/v//')
  if [[ $(echo "$NODE_VERSION" | cut -d. -f1) -lt 18 ]]; then
    log "ERROR" "Node.js 18+ required. Found $NODE_VERSION"
    exit 1
  fi
  
  # Verify critical environment variables
  source .env
  critical_vars=("POSTGRES_USER" "POSTGRES_PASSWORD" "DELTA_EXCHANGE_API_KEY" "DELTA_EXCHANGE_SECRET")
  for var in "${critical_vars[@]}"; do
    if [ -z "${!var}" ]; then
      log "ERROR" "Critical environment variable $var is not set in .env file"
      exit 1
    fi
  done
  
  log "INFO" "Environment validation completed successfully!"
}

# Setup Python virtual environment
setup_python_env() {
  log "INFO" "Setting up Python environment..."
  
  if [ ! -d "venv" ]; then
    log "INFO" "Creating virtual environment..."
    python -m venv venv
  fi
  
  # Activate virtual environment
  source venv/bin/activate
  
  # Install requirements
  log "INFO" "Installing Python dependencies..."
  if [ "$ENV" == "prod" ]; then
    pip install -r requirements.txt
  else
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  fi
  
  log "INFO" "Python environment setup completed!"
}

# Setup Node.js environment
setup_node_env() {
  log "INFO" "Setting up Node.js environment..."
  
  # Install backend dependencies
  log "INFO" "Installing backend dependencies..."
  cd backend
  npm install
  cd ..
  
  # Install frontend dependencies
  log "INFO" "Installing frontend dependencies..."
  cd frontend
  npm install
  cd ..
  
  log "INFO" "Node.js environment setup completed!"
}

# Start services using Docker Compose
start_docker_services() {
  log "INFO" "Starting services with Docker Compose..."
  
  if [ "$ENV" == "prod" ]; then
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
  else
    docker-compose up -d
  fi
  
  log "INFO" "Waiting for services to be healthy..."
  attempt=1
  max_attempts=10
  
  while [ $attempt -le $max_attempts ]; do
    log "INFO" "Health check attempt $attempt/$max_attempts..."
    
    # Check backend health
    if curl -s http://localhost:3006/health &> /dev/null; then
      log "INFO" "Backend is healthy!"
      break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
      log "ERROR" "Services failed to start properly. Check logs for details."
      log "INFO" "You can manually check the status with: docker-compose ps"
      exit 1
    fi
    
    attempt=$((attempt+1))
    sleep 5
  done
  
  log "INFO" "All services started successfully!"
}

# Start the ML system
start_ml_system() {
  log "INFO" "Starting ML system..."
  
  # Activate virtual environment if not already activated
  if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
  fi
  
  # Start the ML system in the appropriate mode
  if [ "$ENV" == "prod" ]; then
    python main.py --mode production &
  else
    python main.py --mode development &
  fi
  
  ML_PID=$!
  echo $ML_PID > "$LOG_DIR/ml_system.pid"
  log "INFO" "ML system started with PID: $ML_PID"
}

# Main function
main() {
  log "INFO" "Starting SmartMarketOOPS in $ENV mode..."
  
  # Run validation and setup
  validate_environment
  setup_python_env
  setup_node_env
  
  # Start services
  start_docker_services
  start_ml_system
  
  log "INFO" "SmartMarketOOPS started successfully in $ENV mode!"
  log "INFO" "Frontend: http://localhost:3000"
  log "INFO" "Backend: http://localhost:3006"
  log "INFO" "ML API: http://localhost:8000"
  log "INFO" "Grafana: http://localhost:3001 (admin/admin)"
  
  log "INFO" "To stop all services: ./stop.sh"
}

# Execute main function
main
