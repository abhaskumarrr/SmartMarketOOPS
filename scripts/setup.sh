#!/bin/bash
#
# SmartMarketOOPS Setup Script
# 
# This script prepares the development environment by:
# - Checking dependencies
# - Setting up database
# - Installing npm packages
# - Building necessary components
# - Configuring environment variables
#

set -e  # Exit on error

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Log helpers
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "  _____                      _   __  __            _        _    ____   ____  _____   _____  "
    echo " / ____|                    | | |  \/  |          | |      | |  / __ \ / __ \|  __ \ / ____| "
    echo "| (___  _ __ ___   __ _ _ __| |_| \  / | __ _ _ __| | _____| |_| |  | | |  | | |__) | (___   "
    echo " \___ \| '_ \` _ \ / _\` | '__| __| |\/| |/ _\` | '__| |/ / _ \ __| |  | | |  | |  ___/ \___ \  "
    echo " ____) | | | | | | (_| | |  | |_| |  | | (_| | |  |   <  __/ |_| |__| | |__| | |     ____) | "
    echo "|_____/|_| |_| |_|\__,_|_|   \__|_|  |_|\__,_|_|  |_|\_\___|\__|\____/ \____/|_|    |_____/  "
    echo -e "${NC}"
    echo -e "${GREEN}Setup Script${NC} - Configure Development Environment"
    echo
}

# Check if required programs are installed
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Node.js
    if ! command -v node >/dev/null 2>&1; then
        log_error "Node.js is not installed. Please install Node.js v20 or higher."
        exit 1
    fi
    
    NODE_VERSION=$(node -v | cut -d 'v' -f 2)
    if [[ "$(echo "$NODE_VERSION" | cut -d '.' -f 1)" -lt 20 ]]; then
        log_warn "Node.js version $NODE_VERSION is below recommended version 20. This may cause issues."
    else
        log_success "Node.js v$NODE_VERSION is installed."
    fi
    
    # Check NPM
    if ! command -v npm >/dev/null 2>&1; then
        log_error "npm is not installed. Please install npm."
        exit 1
    fi
    log_success "npm is installed."
    
    # Check Python
    if ! command -v python3 >/dev/null 2>&1; then
        log_warn "Python 3 is not installed. Some ML features may not work."
    else
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d ' ' -f 2)
        log_success "Python $PYTHON_VERSION is installed."
    fi
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        log_warn "Docker is not installed. Required for containerized deployment."
    else
        log_success "Docker is installed."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        log_warn "Docker Compose is not installed. Required for multi-container deployment."
    else
        log_success "Docker Compose is installed."
    fi
    
    echo
}

# Check if .env file exists, create from example if not
setup_env_file() {
    log_info "Setting up environment variables..."
    
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        if [[ -f "$PROJECT_ROOT/example.env" ]]; then
            cp "$PROJECT_ROOT/example.env" "$PROJECT_ROOT/.env"
            log_success "Created .env file from example.env template."
            log_warn "Please update the .env file with your configuration values."
        else
            log_error "example.env not found. Cannot create .env file."
            touch "$PROJECT_ROOT/.env"
            log_warn "Created empty .env file. You will need to configure it manually."
        fi
    else
        log_success ".env file already exists."
    fi
    
    echo
}

# Install backend dependencies
setup_backend() {
    log_info "Setting up backend..."
    
    if [[ -d "$PROJECT_ROOT/backend" ]]; then
        cd "$PROJECT_ROOT/backend"
        
        if [[ ! -d "node_modules" ]]; then
            log_info "Installing backend dependencies..."
            npm install
            log_success "Backend dependencies installed."
        else
            log_success "Backend dependencies already installed."
        fi
        
        # Generate Prisma client
        if [[ -f "prisma/schema.prisma" ]]; then
            log_info "Generating Prisma client..."
            npx prisma generate
            log_success "Prisma client generated."
        fi
    else
        log_error "Backend directory not found!"
    fi
    
    echo
}

# Install frontend dependencies
setup_frontend() {
    log_info "Setting up frontend..."
    
    if [[ -d "$PROJECT_ROOT/frontend" ]]; then
        cd "$PROJECT_ROOT/frontend"
        
        if [[ ! -d "node_modules" ]]; then
            log_info "Installing frontend dependencies..."
            npm install
            log_success "Frontend dependencies installed."
        else
            log_success "Frontend dependencies already installed."
        fi
    else
        log_error "Frontend directory not found!"
    fi
    
    echo
}

# Set up Python environment for ML system
setup_ml_environment() {
    log_info "Setting up ML environment..."
    
    if command -v python3 >/dev/null 2>&1; then
        cd "$PROJECT_ROOT"
        
        # Create virtual environment if it doesn't exist
        if [[ ! -d "venv" ]]; then
            log_info "Creating Python virtual environment..."
            python3 -m venv venv
            log_success "Virtual environment created."
        else
            log_success "Virtual environment already exists."
        fi
        
        # Activate virtual environment and install dependencies
        if [[ -f "requirements.txt" ]]; then
            log_info "Installing Python dependencies..."
            source venv/bin/activate
            pip install --upgrade pip
            pip install -r requirements.txt
            deactivate
            log_success "Python dependencies installed."
        else
            log_warn "requirements.txt not found. Cannot install Python dependencies."
        fi
    else
        log_warn "Python not installed. Skipping ML environment setup."
    fi
    
    echo
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/backend/logs"
    mkdir -p "$PROJECT_ROOT/frontend/logs"
    
    # Create data directories
    mkdir -p "$PROJECT_ROOT/data/raw"
    mkdir -p "$PROJECT_ROOT/data/processed"
    mkdir -p "$PROJECT_ROOT/data/backtesting"
    
    # Create models directory
    mkdir -p "$PROJECT_ROOT/models/registry"
    
    log_success "Directories created."
    echo
}

# Check database connection
check_database_connection() {
    log_info "Checking database connection..."
    
    # Only try to check if environment variables are set
    if grep -q "POSTGRES_USER" "$PROJECT_ROOT/.env" && grep -q "POSTGRES_PASSWORD" "$PROJECT_ROOT/.env"; then
        cd "$PROJECT_ROOT/backend"
        
        if [[ -f "src/scripts/checkDbHealth.js" ]]; then
            log_info "Testing database connection..."
            node src/scripts/checkDbHealth.js
            if [[ $? -eq 0 ]]; then
                log_success "Database connection successful."
            else
                log_warn "Database connection failed. Make sure your database is running."
            fi
        else
            log_warn "Database health check script not found. Skipping connection test."
        fi
    else
        log_warn "Database credentials not found in .env file. Skipping connection test."
    fi
    
    echo
}

# Show final instructions
show_instructions() {
    echo -e "${GREEN}==== SmartMarketOOPS Setup Complete ====${NC}"
    echo
    echo "Next steps:"
    echo "1. Review and update the .env file with your configuration"
    echo "2. Start the development servers:"
    echo "   - Backend: cd backend && npm run dev"
    echo "   - Frontend: cd frontend && npm run dev"
    echo "3. For full system with ML components: ./start.sh"
    echo
    echo "Documentation:"
    echo "- API Docs: http://localhost:3006/api-docs"
    echo "- Frontend: http://localhost:3000"
    echo
}

# Main execution
print_banner
check_dependencies
setup_env_file
create_directories
setup_backend
setup_frontend
setup_ml_environment
check_database_connection
show_instructions

exit 0 