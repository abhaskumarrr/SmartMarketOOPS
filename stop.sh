#!/bin/bash
# SmartMarketOOPS Shutdown Script
# Gracefully stops all services started by start.sh
# Usage: ./stop.sh

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
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/shutdown_$(date +%Y%m%d_%H%M%S).log"

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

# Stop ML system
stop_ml_system() {
  log "INFO" "Stopping ML system..."
  
  if [ -f "$LOG_DIR/ml_system.pid" ]; then
    ML_PID=$(cat "$LOG_DIR/ml_system.pid")
    
    if ps -p $ML_PID > /dev/null; then
      log "INFO" "Sending SIGTERM to ML system process (PID: $ML_PID)..."
      kill $ML_PID
      
      # Wait for process to terminate
      for i in {1..10}; do
        if ! ps -p $ML_PID > /dev/null; then
          break
        fi
        sleep 1
      done
      
      # Force kill if still running
      if ps -p $ML_PID > /dev/null; then
        log "WARN" "ML process did not terminate gracefully, forcing..."
        kill -9 $ML_PID
      fi
      
      log "INFO" "ML system stopped"
      rm "$LOG_DIR/ml_system.pid"
    else
      log "WARN" "ML system not running (PID: $ML_PID not found)"
    fi
  else
    log "WARN" "ML system PID file not found, attempting to find by process..."
    # Try to find python processes running main.py
    PIDS=$(pgrep -f "python.*main.py")
    if [ -n "$PIDS" ]; then
      log "INFO" "Found ML system processes: $PIDS"
      for PID in $PIDS; do
        log "INFO" "Killing process $PID..."
        kill $PID 2>/dev/null || kill -9 $PID 2>/dev/null
      done
      log "INFO" "ML system processes stopped"
    else
      log "INFO" "No ML system processes found running"
    fi
  fi
}

# Stop Docker services
stop_docker_services() {
  log "INFO" "Stopping Docker services..."
  
  # Check if Docker is running
  if ! command -v docker-compose &> /dev/null; then
    log "ERROR" "docker-compose command not found"
    return
  fi
  
  # Stop containers
  docker-compose down
  
  log "INFO" "Docker services stopped"
}

# Main function
main() {
  log "INFO" "Stopping SmartMarketOOPS services..."
  
  # Stop ML system first
  stop_ml_system
  
  # Then stop Docker services
  stop_docker_services
  
  log "INFO" "All services stopped successfully!"
}

# Execute main function
main 