#!/bin/bash

# Consolidated Start Script for SmartMarketOOPS
# This script combines functionality from:
# - start_week2.sh
# - reliable_start.sh
# - local_dev_server.sh
# - test_and_launch.sh

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

# Environment variables
export NODE_ENV=${NODE_ENV:-"development"}
export PYTHON_ENV=${PYTHON_ENV:-"development"}

# Process management
declare -A pids
trap 'cleanup' EXIT

cleanup() {
    log "INFO" "Cleaning up processes..."
    for pid in "${pids[@]}"; do
        if ps -p $pid > /dev/null; then
            kill $pid 2>/dev/null || true
        fi
    done
}

# Check if services are ready
wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    while ! nc -z localhost $port && [ $attempt -le $max_attempts ]; do
        log "INFO" "Waiting for $service to be ready (attempt $attempt/$max_attempts)..."
        sleep 2
        ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log "ERROR" "$service failed to start"
        return 1
    fi
    
    log "INFO" "$service is ready"
    return 0
}

# Start backend services
start_backend() {
    log "INFO" "Starting backend services..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Start main backend service
    python main.py &
    pids["backend"]=$!
    wait_for_service "Backend" 8000
    
    # Start ML service
    cd ml/src
    python service.py &
    pids["ml"]=$!
    wait_for_service "ML Service" 8001
    cd ../..
    
    # Start analysis bridge
    cd analysis_execution_bridge
    python main.py &
    pids["bridge"]=$!
    wait_for_service "Analysis Bridge" 8002
    cd ..
}

# Start frontend development server
start_frontend() {
    log "INFO" "Starting frontend development server..."
    
    cd frontend
    if [ "$NODE_ENV" = "production" ]; then
        npm run build
        npm run start &
    else
        npm run dev &
    fi
    pids["frontend"]=$!
    wait_for_service "Frontend" 3000
    cd ..
}

# Start monitoring services
start_monitoring() {
    log "INFO" "Starting monitoring services..."
    
    if [ -d "monitoring" ]; then
        cd monitoring
        docker-compose up -d
        cd ..
        wait_for_service "Monitoring" 9090
    fi
}

# Run pre-launch tests
run_tests() {
    log "INFO" "Running pre-launch tests..."
    
    # Run backend tests
    cd backend
    python -m pytest tests/
    cd ..
    
    # Run frontend tests
    cd frontend
    npm run test
    cd ..
}

# Main function
main() {
    local skip_tests=false
    
    # Parse command line arguments
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --skip-tests) skip_tests=true ;;
            --prod) 
                export NODE_ENV="production"
                export PYTHON_ENV="production"
                ;;
            *) log "ERROR" "Unknown parameter: $1"; exit 1 ;;
        esac
        shift
    done
    
    # Run tests unless skipped
    if [ "$skip_tests" = false ]; then
        run_tests
    fi
    
    # Start all services
    start_monitoring
    start_backend
    start_frontend
    
    log "INFO" "All services started successfully!"
    log "INFO" "Frontend: http://localhost:3000"
    log "INFO" "Backend: http://localhost:8000"
    log "INFO" "ML Service: http://localhost:8001"
    log "INFO" "Analysis Bridge: http://localhost:8002"
    log "INFO" "Monitoring: http://localhost:9090"
    
    # Wait for any process to exit
    wait -n
    
    # If any process exits, terminate all
    log "ERROR" "A service has stopped unexpectedly"
    exit 1
}

# Run main function with all arguments
main "$@" 