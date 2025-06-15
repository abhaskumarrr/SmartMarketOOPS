#!/bin/bash

# Unified Troubleshooting Script
# Combines functionality from:
# - fix_missing_dependencies.sh
# - fix_startup_errors.sh
# - fix_port_conflicts.sh
# - quick_fix_auth.sh

set -e

# Configuration
LOG_FILE="troubleshoot.log"
DEFAULT_PORTS=(3000 3001 3002 3003 3006)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-$NC}$1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check and fix missing dependencies
fix_dependencies() {
    log "Checking for missing dependencies..." "$YELLOW"
    
    # Check Python dependencies
    if [ -f "requirements.txt" ]; then
        log "Installing missing Python dependencies..." "$YELLOW"
        pip install -r requirements.txt
    fi
    
    # Check Node.js dependencies
    if [ -f "package.json" ]; then
        log "Installing missing Node.js dependencies..." "$YELLOW"
        npm install
    fi
    
    log "Dependencies check complete." "$GREEN"
}

# Fix port conflicts
fix_port_conflicts() {
    log "Checking for port conflicts..." "$YELLOW"
    
    for port in "${DEFAULT_PORTS[@]}"; do
        if lsof -i ":$port" > /dev/null; then
            log "Port $port is in use. Attempting to free it..." "$YELLOW"
            kill $(lsof -t -i ":$port") 2>/dev/null || true
            sleep 1
            if ! lsof -i ":$port" > /dev/null; then
                log "Successfully freed port $port" "$GREEN"
            else
                log "Could not free port $port. Please check manually." "$RED"
            fi
        fi
    done
    
    log "Port conflict check complete." "$GREEN"
}

# Fix startup errors
fix_startup_errors() {
    log "Attempting to fix common startup errors..." "$YELLOW"
    
    # Clear cache directories
    log "Clearing cache directories..." "$YELLOW"
    rm -rf node_modules/.cache
    rm -rf .next/cache
    
    # Reset database connections
    log "Resetting database connections..." "$YELLOW"
    if [ -f ".env" ]; then
        source .env
        if [ ! -z "$DATABASE_URL" ]; then
            npx prisma generate
        fi
    fi
    
    # Clear PM2 processes if running
    if command -v pm2 &> /dev/null; then
        log "Resetting PM2 processes..." "$YELLOW"
        pm2 delete all 2>/dev/null || true
        pm2 save
        pm2 reset all
    fi
    
    log "Startup error fixes complete." "$GREEN"
}

# Fix authentication issues
fix_auth() {
    log "Attempting to fix authentication issues..." "$YELLOW"
    
    # Check JWT secret
    if [ -f ".env" ]; then
        if ! grep -q "JWT_SECRET" .env; then
            log "JWT_SECRET not found in .env. Adding it..." "$YELLOW"
            echo "JWT_SECRET=$(openssl rand -base64 32)" >> .env
        fi
    fi
    
    # Clear authentication tokens
    log "Clearing stored authentication tokens..." "$YELLOW"
    rm -rf .auth_tokens/* 2>/dev/null || true
    
    # Reset API keys if needed
    if [ -f ".env" ] && grep -q "API_KEY" .env; then
        log "Regenerating API keys..." "$YELLOW"
        sed -i.bak '/API_KEY/d' .env
        echo "API_KEY=$(openssl rand -hex 32)" >> .env
    fi
    
    log "Authentication fixes complete." "$GREEN"
}

# Show help message
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -d, --dependencies    Fix missing dependencies"
    echo "  -p, --ports          Fix port conflicts"
    echo "  -s, --startup        Fix startup errors"
    echo "  -a, --auth           Fix authentication issues"
    echo "  -A, --all            Fix all issues"
    echo "  -h, --help           Show this help message"
    echo
    exit 0
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        show_help
    fi
    
    while [ "$1" != "" ]; do
        case $1 in
            -d | --dependencies)
                fix_dependencies
                ;;
            -p | --ports)
                fix_port_conflicts
                ;;
            -s | --startup)
                fix_startup_errors
                ;;
            -a | --auth)
                fix_auth
                ;;
            -A | --all)
                fix_dependencies
                fix_port_conflicts
                fix_startup_errors
                fix_auth
                ;;
            -h | --help)
                show_help
                ;;
            *)
                log "Unknown option: $1" "$RED"
                show_help
                ;;
        esac
        shift
    done
}

main "$@" 