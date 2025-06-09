#!/bin/bash

# SmartMarketOOPS Docker Setup Validation Script
# This script validates the Docker configuration without requiring Docker to be installed

set -e

# Add Docker to PATH
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"

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

print_header() {
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

# Validation functions
validate_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        print_success "$description exists: $file"
        return 0
    else
        print_error "$description missing: $file"
        return 1
    fi
}

validate_dockerfile() {
    local dockerfile=$1
    local service=$2
    
    if [ -f "$dockerfile" ]; then
        print_status "Validating $service Dockerfile..."
        
        # Check for required instructions
        if grep -q "FROM" "$dockerfile"; then
            print_success "  ✓ Has FROM instruction"
        else
            print_error "  ✗ Missing FROM instruction"
        fi
        
        if grep -q "WORKDIR" "$dockerfile"; then
            print_success "  ✓ Has WORKDIR instruction"
        else
            print_warning "  ⚠ Missing WORKDIR instruction"
        fi
        
        if grep -q "EXPOSE" "$dockerfile"; then
            print_success "  ✓ Has EXPOSE instruction"
        else
            print_warning "  ⚠ Missing EXPOSE instruction"
        fi
        
        if grep -q "CMD\|ENTRYPOINT" "$dockerfile"; then
            print_success "  ✓ Has startup command"
        else
            print_error "  ✗ Missing startup command (CMD or ENTRYPOINT)"
        fi
        
        return 0
    else
        print_error "$service Dockerfile missing: $dockerfile"
        return 1
    fi
}

validate_compose_file() {
    local compose_file=$1
    
    if [ -f "$compose_file" ]; then
        print_status "Validating $compose_file..."
        
        # Check for required services
        local services=("postgres" "backend" "frontend" "ml-system" "redis")
        for service in "${services[@]}"; do
            if grep -q "$service:" "$compose_file"; then
                print_success "  ✓ Service defined: $service"
            else
                print_error "  ✗ Service missing: $service"
            fi
        done
        
        # Check for volumes
        if grep -q "volumes:" "$compose_file"; then
            print_success "  ✓ Has volume definitions"
        else
            print_warning "  ⚠ No volume definitions found"
        fi
        
        # Check for networks
        if grep -q "networks:" "$compose_file"; then
            print_success "  ✓ Has network definitions"
        else
            print_warning "  ⚠ No network definitions found"
        fi
        
        return 0
    else
        print_error "Compose file missing: $compose_file"
        return 1
    fi
}

validate_env_template() {
    local env_file=$1
    
    if [ -f "$env_file" ]; then
        print_status "Validating environment template..."
        
        # Check for required variables
        local required_vars=("POSTGRES_USER" "POSTGRES_PASSWORD" "POSTGRES_DB" "DELTA_EXCHANGE_API_KEY" "DELTA_EXCHANGE_SECRET")
        for var in "${required_vars[@]}"; do
            if grep -q "$var=" "$env_file"; then
                print_success "  ✓ Variable defined: $var"
            else
                print_error "  ✗ Variable missing: $var"
            fi
        done
        
        return 0
    else
        print_error "Environment template missing: $env_file"
        return 1
    fi
}

# Main validation
print_header "SmartMarketOOPS Docker Setup Validation"

print_status "Starting Docker configuration validation..."

# Track validation results
validation_errors=0

# Validate main docker-compose file
print_header "Docker Compose Configuration"
if ! validate_compose_file "docker-compose.yml"; then
    ((validation_errors++))
fi

# Validate override file
if ! validate_compose_file "docker-compose.override.yml"; then
    ((validation_errors++))
fi

# Validate Dockerfiles
print_header "Dockerfile Validation"
if ! validate_dockerfile "backend/Dockerfile" "Backend"; then
    ((validation_errors++))
fi

if ! validate_dockerfile "frontend/Dockerfile" "Frontend"; then
    ((validation_errors++))
fi

if ! validate_dockerfile "docker/Dockerfile.ml-system" "ML System"; then
    ((validation_errors++))
fi

# Validate environment configuration
print_header "Environment Configuration"
if ! validate_env_template ".env.docker"; then
    ((validation_errors++))
fi

# Validate scripts
print_header "Docker Scripts"
if ! validate_file "scripts/docker-dev.sh" "Docker development script"; then
    ((validation_errors++))
fi

# Validate application files
print_header "Application Files"
validate_file "frontend/package.json" "Frontend package.json"
validate_file "backend/package.json" "Backend package.json"
validate_file "requirements.txt" "Python requirements"
validate_file "main.py" "ML system main file"

# Check for health endpoints
print_header "Health Check Endpoints"
if [ -f "backend/src/routes/healthRoutes.ts" ]; then
    print_success "Backend health endpoint exists"
else
    print_warning "Backend health endpoint missing"
fi

if [ -f "frontend/src/app/api/health/route.ts" ]; then
    print_success "Frontend health endpoint exists"
else
    print_warning "Frontend health endpoint missing"
fi

# Final summary
print_header "Validation Summary"

if [ $validation_errors -eq 0 ]; then
    print_success "All Docker configurations are valid!"
    print_status "Your Docker setup is ready to use."
    echo ""
    print_status "Next steps:"
    echo "1. Install Docker Desktop if not already installed"
    echo "2. Copy .env.docker to .env and update with your credentials"
    echo "3. Run: ./scripts/docker-dev.sh dev"
    echo ""
else
    print_error "Found $validation_errors validation errors."
    print_status "Please fix the errors above before using Docker."
fi

# Docker installation check
print_header "Docker Installation Check"
if command -v docker > /dev/null 2>&1; then
    print_success "Docker is installed"
    docker --version
    
    if command -v docker > /dev/null 2>&1 && docker compose version > /dev/null 2>&1; then
        print_success "Docker Compose is installed"
        docker compose version
    elif command -v docker-compose > /dev/null 2>&1; then
        print_success "Docker Compose (legacy) is installed"
        docker-compose --version
    else
        print_warning "Docker Compose not found. Install Docker Desktop or docker-compose."
    fi
    
    if docker info > /dev/null 2>&1; then
        print_success "Docker daemon is running"
    else
        print_warning "Docker daemon is not running. Start Docker Desktop."
    fi
else
    print_warning "Docker is not installed."
    echo ""
    print_status "To install Docker:"
    echo "1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop"
    echo "2. Install and start Docker Desktop"
    echo "3. Run this validation script again"
fi

echo ""
print_status "Validation completed!"

exit $validation_errors
