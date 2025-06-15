#!/bin/bash

# Consolidated dashboard management script
# Combines functionality from:
# - start_dashboard.sh
# - fix_dashboard_issues.sh
# - launch_dashboard.sh

set -e

# Configuration
DASHBOARD_PORT=3000
API_PORT=3001
WEBSOCKET_PORT=3002
LOG_FILE="dashboard.log"

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

# Check system requirements
check_requirements() {
    log "Checking system requirements..." "$YELLOW"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log "Node.js is not installed. Please install Node.js 14 or higher." "$RED"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        log "npm is not installed. Please install npm." "$RED"
        exit 1
    fi
    
    # Check port availability
    if lsof -Pi :$DASHBOARD_PORT -sTCP:LISTEN -t >/dev/null ; then
        log "Port $DASHBOARD_PORT is already in use. Please free up the port or change DASHBOARD_PORT." "$RED"
        exit 1
    fi
    
    log "System requirements check passed." "$GREEN"
}

# Fix common dashboard issues
fix_dashboard_issues() {
    log "Fixing common dashboard issues..." "$YELLOW"
    
    # Clear npm cache
    npm cache clean --force
    
    # Remove node_modules and package-lock.json
    rm -rf node_modules package-lock.json
    
    # Reinstall dependencies
    npm install
    
    # Clear browser cache recommendation
    log "Please clear your browser cache if you continue to experience issues." "$YELLOW"
    
    log "Dashboard fixes applied." "$GREEN"
}

# Start the dashboard
start_dashboard() {
    log "Starting dashboard..." "$YELLOW"
    
    # Build the dashboard
    npm run build
    
    # Start the dashboard in production mode
    npm run start
}

# Launch the dashboard with monitoring
launch_dashboard() {
    log "Launching dashboard with monitoring..." "$YELLOW"
    
    # Start API server if not running
    if ! lsof -Pi :$API_PORT -sTCP:LISTEN -t >/dev/null ; then
        npm run api &
        log "API server started on port $API_PORT" "$GREEN"
    fi
    
    # Start WebSocket server if not running
    if ! lsof -Pi :$WEBSOCKET_PORT -sTCP:LISTEN -t >/dev/null ; then
        npm run websocket &
        log "WebSocket server started on port $WEBSOCKET_PORT" "$GREEN"
    fi
    
    # Start the dashboard with PM2
    if command -v pm2 &> /dev/null; then
        pm2 start npm --name "dashboard" -- start
        log "Dashboard started with PM2" "$GREEN"
    else
        start_dashboard
    fi
}

# Display help message
show_help() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  start       Start the dashboard normally"
    echo "  fix         Fix common dashboard issues"
    echo "  launch      Launch dashboard with API and WebSocket servers"
    echo "  help        Show this help message"
    echo
}

# Main script logic
case "$1" in
    "start")
        check_requirements
        start_dashboard
        ;;
    "fix")
        fix_dashboard_issues
        ;;
    "launch")
        check_requirements
        launch_dashboard
        ;;
    "help"|"")
        show_help
        ;;
    *)
        log "Unknown command: $1" "$RED"
        show_help
        exit 1
        ;;
esac 