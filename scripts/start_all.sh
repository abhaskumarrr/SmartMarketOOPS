#!/bin/bash

# ============================================================================
# SmartMarketOOPS - Unified Startup Script
# ============================================================================
# This script launches all the necessary services for the SmartMarketOOPS
# trading system: Backend, Frontend, and the ML Service.
#
# It ensures services are started in the correct order and includes health
# checks to verify that each component is running before proceeding.
# ============================================================================

# Function to print colored output
print_info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
  echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Function to check if a port is in use
is_port_in_use() {
  lsof -i :$1 >/dev/null
}

# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  print_error ".env file not found. Please copy example.env to .env and configure it."
  exit 1
fi

# --- 1. Start the ML Service ---
print_info "Starting the ML Service on port $ML_PORT..."
cd ml
if is_port_in_use $ML_PORT; then
  print_info "ML Service is already running on port $ML_PORT."
else
  python3 -m uvicorn src.api.app:app --host $HOST --port $ML_PORT &
  ML_PID=$!
  sleep 5 # Give the server a moment to start

  # Health Check
  if ! curl -s "http://$HOST:$ML_PORT/health" > /dev/null; then
    print_error "ML Service failed to start. Check ml/ directory for issues."
    kill $ML_PID
    exit 1
  fi
  print_success "ML Service started successfully (PID: $ML_PID)."
fi
cd ..

# --- 2. Start the Backend Service ---
print_info "Starting the Backend Service on port $BACKEND_PORT..."
cd backend
if is_port_in_use $BACKEND_PORT; then
  print_info "Backend Service is already running on port $BACKEND_PORT."
else
  npm run dev &
  BACKEND_PID=$!
  sleep 10 # Give the server time to compile and start

  # Health Check
  if ! curl -s "http://$HOST:$BACKEND_PORT/api/health" > /dev/null; then
    print_error "Backend Service failed to start. Check backend/ directory for issues."
    kill $BACKEND_PID
    exit 1
  fi
  print_success "Backend Service started successfully (PID: $BACKEND_PID)."
fi
cd ..

# --- 3. Start the Frontend Service ---
print_info "Starting the Frontend Service on port $FRONTEND_PORT..."
cd frontend
if is_port_in_use $FRONTEND_PORT; then
  print_info "Frontend Service is already running on port $FRONTEND_PORT."
else
  npm run dev &
  FRONTEND_PID=$!
  sleep 10 # Give Next.js time to compile

  # Health Check
  if ! curl -s "http://localhost:$FRONTEND_PORT" > /dev/null; then
    print_error "Frontend Service failed to start. Check frontend/ directory for issues."
    kill $FRONTEND_PID
    exit 1
  fi
  print_success "Frontend Service started successfully (PID: $FRONTEND_PID)."
fi
cd ..

print_success "All services for SmartMarketOOPS have been started."
print_info "Access the dashboard at http://localhost:$FRONTEND_PORT"
