#!/bin/bash

# SmartMarketOOPS Infrastructure Setup Script
# Sets up Redis, QuestDB, and PostgreSQL for development and testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if Docker is installed and running
check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Docker is installed and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    
    if docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    else
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    print_success "Docker Compose is available: $DOCKER_COMPOSE_CMD"
}

# Stop and remove existing containers
cleanup_existing() {
    print_status "Cleaning up existing containers..."
    
    $DOCKER_COMPOSE_CMD -f docker-compose.infrastructure.yml down -v 2>/dev/null || true
    
    # Remove any orphaned containers
    docker container prune -f 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Start infrastructure services
start_infrastructure() {
    print_status "Starting infrastructure services..."
    
    $DOCKER_COMPOSE_CMD -f docker-compose.infrastructure.yml up -d
    
    print_success "Infrastructure services started"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for services to be healthy..."
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker exec smoops-redis redis-cli ping &> /dev/null; then
            print_success "Redis is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Redis failed to start within 60 seconds"
        exit 1
    fi
    
    # Wait for QuestDB
    print_status "Waiting for QuestDB..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:9000/status &> /dev/null; then
            print_success "QuestDB is ready"
            break
        fi
        sleep 3
        timeout=$((timeout - 3))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "QuestDB failed to start within 120 seconds"
        exit 1
    fi
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker exec smoops-postgres pg_isready -U smoops_user -d smartmarketoops &> /dev/null; then
            print_success "PostgreSQL is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "PostgreSQL failed to start within 60 seconds"
        exit 1
    fi
}

# Create QuestDB tables
setup_questdb_tables() {
    print_status "Setting up QuestDB tables..."
    
    # Create tables using HTTP API
    curl -G "http://localhost:9000/exec" \
        --data-urlencode "query=CREATE TABLE IF NOT EXISTS metrics (
            timestamp TIMESTAMP,
            name SYMBOL,
            value DOUBLE,
            tags SYMBOL
        ) timestamp(timestamp) PARTITION BY DAY WAL;" 2>/dev/null || true
    
    curl -G "http://localhost:9000/exec" \
        --data-urlencode "query=CREATE TABLE IF NOT EXISTS trading_signals (
            timestamp TIMESTAMP,
            id SYMBOL,
            symbol SYMBOL,
            type SYMBOL,
            direction SYMBOL,
            strength SYMBOL,
            timeframe SYMBOL,
            source SYMBOL,
            price DOUBLE,
            target_price DOUBLE,
            stop_loss DOUBLE,
            confidence_score DOUBLE,
            expected_return DOUBLE,
            expected_risk DOUBLE,
            risk_reward_ratio DOUBLE
        ) timestamp(timestamp) PARTITION BY DAY WAL;" 2>/dev/null || true
    
    curl -G "http://localhost:9000/exec" \
        --data-urlencode "query=CREATE TABLE IF NOT EXISTS ml_predictions (
            timestamp TIMESTAMP,
            id SYMBOL,
            model_id SYMBOL,
            symbol SYMBOL,
            timeframe SYMBOL,
            prediction_type SYMBOL,
            values STRING,
            confidence_scores STRING
        ) timestamp(timestamp) PARTITION BY DAY WAL;" 2>/dev/null || true
    
    curl -G "http://localhost:9000/exec" \
        --data-urlencode "query=CREATE TABLE IF NOT EXISTS performance_metrics (
            timestamp TIMESTAMP,
            system SYMBOL,
            component SYMBOL,
            metric SYMBOL,
            unit SYMBOL,
            value DOUBLE,
            tags SYMBOL
        ) timestamp(timestamp) PARTITION BY DAY WAL;" 2>/dev/null || true
    
    print_success "QuestDB tables created"
}

# Test connections
test_connections() {
    print_status "Testing connections..."
    
    # Test Redis
    if docker exec smoops-redis redis-cli set test_key "test_value" &> /dev/null && \
       docker exec smoops-redis redis-cli get test_key &> /dev/null; then
        print_success "Redis connection test passed"
        docker exec smoops-redis redis-cli del test_key &> /dev/null
    else
        print_error "Redis connection test failed"
        exit 1
    fi
    
    # Test QuestDB
    if curl -G "http://localhost:9000/exec" \
        --data-urlencode "query=SELECT 1;" &> /dev/null; then
        print_success "QuestDB connection test passed"
    else
        print_error "QuestDB connection test failed"
        exit 1
    fi
    
    # Test PostgreSQL
    if docker exec smoops-postgres psql -U smoops_user -d smartmarketoops -c "SELECT 1;" &> /dev/null; then
        print_success "PostgreSQL connection test passed"
    else
        print_error "PostgreSQL connection test failed"
        exit 1
    fi
}

# Display service information
display_info() {
    print_success "Infrastructure setup completed successfully!"
    echo
    echo "Service Information:"
    echo "==================="
    echo "Redis:"
    echo "  - Host: localhost"
    echo "  - Port: 6379"
    echo "  - Management UI: http://localhost:8081 (run with --profile tools)"
    echo
    echo "QuestDB:"
    echo "  - HTTP API: http://localhost:9000"
    echo "  - PostgreSQL Wire: localhost:8812"
    echo "  - ILP (Line Protocol): localhost:9009"
    echo "  - Web Console: http://localhost:9000"
    echo
    echo "PostgreSQL:"
    echo "  - Host: localhost"
    echo "  - Port: 5432"
    echo "  - Database: smartmarketoops"
    echo "  - Username: smoops_user"
    echo "  - Password: smoops_password"
    echo
    echo "Environment Variables for .env:"
    echo "==============================="
    echo "REDIS_HOST=localhost"
    echo "REDIS_PORT=6379"
    echo "QUESTDB_HOST=localhost"
    echo "QUESTDB_PORT=9000"
    echo "QUESTDB_ILP_PORT=9009"
    echo "DATABASE_URL=postgresql://smoops_user:smoops_password@localhost:5432/smartmarketoops"
    echo
    echo "To stop services: $DOCKER_COMPOSE_CMD -f docker-compose.infrastructure.yml down"
    echo "To view logs: $DOCKER_COMPOSE_CMD -f docker-compose.infrastructure.yml logs -f"
}

# Main execution
main() {
    echo "SmartMarketOOPS Infrastructure Setup"
    echo "===================================="
    echo
    
    check_docker
    check_docker_compose
    cleanup_existing
    start_infrastructure
    wait_for_services
    setup_questdb_tables
    test_connections
    display_info
}

# Handle script arguments
case "${1:-}" in
    "start")
        start_infrastructure
        wait_for_services
        ;;
    "stop")
        print_status "Stopping infrastructure services..."
        $DOCKER_COMPOSE_CMD -f docker-compose.infrastructure.yml down
        print_success "Infrastructure services stopped"
        ;;
    "restart")
        cleanup_existing
        start_infrastructure
        wait_for_services
        ;;
    "status")
        $DOCKER_COMPOSE_CMD -f docker-compose.infrastructure.yml ps
        ;;
    "logs")
        $DOCKER_COMPOSE_CMD -f docker-compose.infrastructure.yml logs -f
        ;;
    "clean")
        print_status "Cleaning up all infrastructure data..."
        $DOCKER_COMPOSE_CMD -f docker-compose.infrastructure.yml down -v
        docker volume prune -f
        print_success "Infrastructure cleanup completed"
        ;;
    *)
        main
        ;;
esac
