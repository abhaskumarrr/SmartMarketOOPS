#!/bin/bash

# SmartMarketOOPS Docker Development Script
# This script helps manage Docker containers for development

set -e

# Add Docker to PATH
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

# Use Docker Compose v2 syntax
DOCKER_COMPOSE="docker compose"

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

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Check if .env file exists
check_env() {
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating from template..."
        cp .env.docker .env
        print_warning "Please update .env with your actual configuration values."
        print_warning "Especially set your Delta Exchange API credentials."
    fi
}

# Build all containers
build_containers() {
    print_status "Building Docker containers..."
    $DOCKER_COMPOSE build --parallel
    print_success "All containers built successfully!"
}

# Start all services
start_services() {
    print_status "Starting all services..."
    $DOCKER_COMPOSE up -d
    print_success "All services started!"

    print_status "Waiting for services to be ready..."
    sleep 15

    # Check service health
    check_health
}

# Stop all services
stop_services() {
    print_status "Stopping all services..."
    $DOCKER_COMPOSE down
    print_success "All services stopped!"
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Check backend
    if curl -f http://localhost:3005/health > /dev/null 2>&1; then
        print_success "Backend is healthy"
    else
        print_warning "Backend health check failed"
    fi
    
    # Check frontend
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        print_success "Frontend is healthy"
    else
        print_warning "Frontend health check failed"
    fi
    
    # Check ML system
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "ML System is healthy"
    else
        print_warning "ML System health check failed"
    fi
}

# Show logs
show_logs() {
    local service=$1
    if [ -z "$service" ]; then
        $DOCKER_COMPOSE logs -f
    else
        $DOCKER_COMPOSE logs -f "$service"
    fi
}

# Clean up containers and volumes
cleanup() {
    print_status "Cleaning up containers and volumes..."
    $DOCKER_COMPOSE down -v --remove-orphans
    docker system prune -f
    print_success "Cleanup completed!"
}

# Show service status
status() {
    print_status "Service Status:"
    $DOCKER_COMPOSE ps
    echo ""
    print_status "Service Health:"
    check_health
}

# Main script logic
case "$1" in
    "build")
        check_docker
        check_env
        build_containers
        ;;
    "start")
        check_docker
        check_env
        start_services
        ;;
    "stop")
        check_docker
        stop_services
        ;;
    "restart")
        check_docker
        stop_services
        start_services
        ;;
    "logs")
        check_docker
        show_logs "$2"
        ;;
    "health")
        check_docker
        check_health
        ;;
    "status")
        check_docker
        status
        ;;
    "cleanup")
        check_docker
        cleanup
        ;;
    "dev")
        check_docker
        check_env
        print_status "Starting development environment..."
        $DOCKER_COMPOSE down > /dev/null 2>&1 || true
        build_containers
        start_services
        print_success "Development environment is ready!"
        print_status "Access points:"
        echo "  Frontend: http://localhost:3000"
        echo "  Backend API: http://localhost:3005"
        echo "  ML System: http://localhost:8000"
        echo "  QuestDB Console: http://localhost:9000"
        echo "  Grafana: http://localhost:3001"
        ;;
    *)
        echo "SmartMarketOOPS Docker Development Script"
        echo ""
        echo "Usage: $0 {build|start|stop|restart|logs|health|status|cleanup|dev}"
        echo ""
        echo "Commands:"
        echo "  build    - Build all Docker containers"
        echo "  start    - Start all services"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart all services"
        echo "  logs     - Show logs (optionally for specific service)"
        echo "  health   - Check service health"
        echo "  status   - Show service status"
        echo "  cleanup  - Clean up containers and volumes"
        echo "  dev      - Full development setup (build + start + health check)"
        echo ""
        echo "Examples:"
        echo "  $0 dev                 # Start full development environment"
        echo "  $0 logs backend        # Show backend logs"
        echo "  $0 logs                # Show all logs"
        exit 1
        ;;
esac
