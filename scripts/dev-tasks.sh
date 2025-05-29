#!/bin/bash
#
# Development Tasks Script for SMOOPs Trading Bot
# This script provides quick commands for common development tasks

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

# Function to display help message
show_help() {
  echo -e "${BOLD}SMOOPs Trading Bot - Development Tasks${NC}"
  echo "========================================="
  echo ""
  echo "Usage: $0 [task]"
  echo ""
  echo "Available tasks:"
  echo "  start       - Start all services in development mode"
  echo "  start:docker - Start all services using Docker"
  echo "  stop        - Stop all Docker services"
  echo "  restart     - Restart all Docker services"
  echo "  logs        - View logs for all Docker services"
  echo "  logs:backend - View logs for backend service"
  echo "  logs:frontend - View logs for frontend service"
  echo "  logs:ml     - View logs for ML service"
  echo "  db:migrate  - Run database migrations"
  echo "  db:reset    - Reset database (CAUTION: Deletes all data)"
  echo "  db:seed     - Seed database with sample data"
  echo "  lint        - Run ESLint on all JavaScript/TypeScript files"
  echo "  lint:fix    - Run ESLint with auto-fix on all JavaScript/TypeScript files"
  echo "  ml:train    - Train a new ML model"
  echo "  ml:predict  - Run predictions with a trained model"
  echo "  test        - Run all tests"
  echo "  test:backend - Run backend tests"
  echo "  test:frontend - Run frontend tests"
  echo "  test:ml     - Run ML tests"
  echo "  clean       - Clean up temporary files and build artifacts"
  echo "  help        - Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 start"
  echo "  $0 logs:backend"
  echo "  $0 db:migrate"
}

# Start all services in development mode
start_dev() {
  echo -e "${BLUE}Starting all services in development mode...${NC}"
  cd "$PROJECT_ROOT" && npm run dev
}

# Start all services using Docker
start_docker() {
  echo -e "${BLUE}Starting all services using Docker...${NC}"
  cd "$PROJECT_ROOT" && npm run docker:up
}

# Stop all Docker services
stop_docker() {
  echo -e "${BLUE}Stopping all Docker services...${NC}"
  cd "$PROJECT_ROOT" && npm run docker:down
}

# Restart all Docker services
restart_docker() {
  echo -e "${BLUE}Restarting all Docker services...${NC}"
  cd "$PROJECT_ROOT" && npm run docker:restart
}

# View logs for all Docker services
logs_all() {
  echo -e "${BLUE}Viewing logs for all services...${NC}"
  cd "$PROJECT_ROOT" && npm run docker:logs
}

# View logs for specific service
logs_service() {
  local service=$1
  echo -e "${BLUE}Viewing logs for $service service...${NC}"
  cd "$PROJECT_ROOT" && docker-compose logs -f "$service"
}

# Run database migrations
db_migrate() {
  echo -e "${BLUE}Running database migrations...${NC}"
  cd "$PROJECT_ROOT" && npm run db:migrate
}

# Reset database
db_reset() {
  echo -e "${RED}CAUTION: This will delete all data in the database!${NC}"
  read -p "Are you sure you want to continue? (y/N) " confirm
  if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo -e "${BLUE}Resetting database...${NC}"
    cd "$PROJECT_ROOT" && npm run db:reset
  else
    echo -e "${YELLOW}Database reset cancelled.${NC}"
  fi
}

# Seed database with sample data
db_seed() {
  echo -e "${BLUE}Seeding database with sample data...${NC}"
  cd "$PROJECT_ROOT/backend" && node src/scripts/seed.js
}

# Run linting on all JavaScript/TypeScript and Python files
lint() {
  echo -e "${BLUE}Running linters...${NC}"
  
  echo -e "${YELLOW}Running ESLint on JS/TS files...${NC}"
  cd "$PROJECT_ROOT" && npx eslint "**/*.{js,jsx,ts,tsx}" --ignore-pattern "node_modules/" --ignore-pattern "dist/"
  
  echo -e "${YELLOW}Running Ruff on Python files...${NC}"
  cd "$PROJECT_ROOT" && python -m ruff check .
  
  echo -e "${YELLOW}Running MyPy for type checking...${NC}"
  cd "$PROJECT_ROOT" && python -m mypy ml/src
  
  echo -e "${YELLOW}Running Bandit for security checks...${NC}"
  cd "$PROJECT_ROOT" && python -m bandit -r ml/src -ll
}

# Run linters with auto-fix
lint_fix() {
  echo -e "${BLUE}Running linters with auto-fix...${NC}"
  
  echo -e "${YELLOW}Running ESLint with auto-fix...${NC}"
  cd "$PROJECT_ROOT" && npx eslint "**/*.{js,jsx,ts,tsx}" --ignore-pattern "node_modules/" --ignore-pattern "dist/" --fix
  
  echo -e "${YELLOW}Running Ruff with auto-fix...${NC}"
  cd "$PROJECT_ROOT" && python -m ruff check --fix .
  
  echo -e "${YELLOW}Running Black formatter...${NC}"
  cd "$PROJECT_ROOT" && python -m black .
  
  echo -e "${YELLOW}Running isort...${NC}"
  cd "$PROJECT_ROOT" && python -m isort .
}

# Train a new ML model
ml_train() {
  echo -e "${BLUE}Training new ML model...${NC}"
  echo -e "${YELLOW}This is a placeholder. Update with actual ML training command.${NC}"
  echo "Example: cd $PROJECT_ROOT && python -m ml.main train --symbol BTC-USDT --model-type smc_transformer"
}

# Run predictions with a trained model
ml_predict() {
  echo -e "${BLUE}Running predictions with trained model...${NC}"
  echo -e "${YELLOW}This is a placeholder. Update with actual ML prediction command.${NC}"
  echo "Example: cd $PROJECT_ROOT && python -m ml.main predict --symbol BTC-USDT"
}

# Run all tests
test_all() {
  echo -e "${BLUE}Running all tests...${NC}"
  test_backend
  test_frontend
  test_ml
}

# Run backend tests
test_backend() {
  echo -e "${BLUE}Running backend tests...${NC}"
  cd "$PROJECT_ROOT/backend" && npm test
}

# Run frontend tests
test_frontend() {
  echo -e "${BLUE}Running frontend tests...${NC}"
  cd "$PROJECT_ROOT/frontend" && npm test
}

# Run ML tests
test_ml() {
  echo -e "${BLUE}Running ML tests...${NC}"
  cd "$PROJECT_ROOT/ml" && python -m unittest discover -s tests
}

# Clean up temporary files and build artifacts
clean() {
  echo -e "${BLUE}Cleaning up temporary files and build artifacts...${NC}"
  
  echo -e "${YELLOW}Cleaning backend...${NC}"
  cd "$PROJECT_ROOT/backend" && rm -rf node_modules/.cache
  
  echo -e "${YELLOW}Cleaning frontend...${NC}"
  cd "$PROJECT_ROOT/frontend" && rm -rf .next node_modules/.cache
  
  echo -e "${YELLOW}Cleaning ML...${NC}"
  cd "$PROJECT_ROOT/ml" && rm -rf __pycache__ **/__pycache__ .pytest_cache
  
  echo -e "${GREEN}Clean up complete.${NC}"
}

# Main function to handle command-line arguments
main() {
  if [ $# -eq 0 ]; then
    show_help
    exit 0
  fi

  case "$1" in
    start)
      start_dev
      ;;
    start:docker)
      start_docker
      ;;
    stop)
      stop_docker
      ;;
    restart)
      restart_docker
      ;;
    logs)
      logs_all
      ;;
    logs:backend)
      logs_service backend
      ;;
    logs:frontend)
      logs_service frontend
      ;;
    logs:ml)
      logs_service ml
      ;;
    db:migrate)
      db_migrate
      ;;
    db:reset)
      db_reset
      ;;
    db:seed)
      db_seed
      ;;
    lint)
      lint
      ;;
    lint:fix)
      lint_fix
      ;;
    ml:train)
      ml_train
      ;;
    ml:predict)
      ml_predict
      ;;
    test)
      test_all
      ;;
    test:backend)
      test_backend
      ;;
    test:frontend)
      test_frontend
      ;;
    test:ml)
      test_ml
      ;;
    clean)
      clean
      ;;
    help|--help|-h)
      show_help
      ;;
    *)
      echo -e "${RED}Unknown task: $1${NC}"
      show_help
      exit 1
      ;;
  esac
}

# Run the main function with all command-line arguments
main "$@" 