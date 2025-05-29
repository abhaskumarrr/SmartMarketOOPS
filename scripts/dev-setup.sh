#!/bin/bash
#
# Development Environment Setup Script for SMOOPs Trading Bot
# This script sets up a complete development environment for the project

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
EXAMPLE_ENV="$PROJECT_ROOT/example.env"

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

echo -e "${BOLD}SMOOPs Trading Bot - Developer Environment Setup${NC}"
echo "=================================================="
echo ""

# Check if Docker and Docker Compose are installed
check_dependencies() {
  echo -e "${BLUE}Checking dependencies...${NC}"
  
  if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed.${NC}"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
  fi
  
  if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
  fi
  
  if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed.${NC}"
    echo "Please install Node.js (v18+) from https://nodejs.org/"
    exit 1
  fi
  
  if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed.${NC}"
    echo "Please install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
  fi
  
  echo -e "${GREEN}All dependencies are installed.${NC}"
}

# Create .env file if it doesn't exist
setup_env() {
  echo -e "\n${BLUE}Setting up environment...${NC}"
  
  if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}No .env file found. Creating from example.env...${NC}"
    cp "$EXAMPLE_ENV" "$ENV_FILE"
    echo -e "${GREEN}Created .env file. Please edit it with your credentials.${NC}"
  else
    echo -e "${GREEN}Found existing .env file.${NC}"
  fi
  
  # Run the environment setup script with development environment
  bash "$SCRIPT_DIR/setup-env.sh"
  
  # Generate encryption key if needed
  if ! grep -q "ENCRYPTION_MASTER_KEY=.\+" "$ENV_FILE"; then
    echo -e "\n${BLUE}Generating encryption key...${NC}"
    node "$SCRIPT_DIR/generate-encryption-key.js"
  fi
  
  # Validate environment configuration
  echo -e "\n${BLUE}Validating environment configuration...${NC}"
  node "$SCRIPT_DIR/check-env.js"
}

# Install dependencies for all packages
install_dependencies() {
  echo -e "\n${BLUE}Installing dependencies...${NC}"
  
  echo -e "${YELLOW}Installing root project dependencies...${NC}"
  npm install
  
  echo -e "${YELLOW}Installing backend dependencies...${NC}"
  cd "$PROJECT_ROOT/backend" && npm install
  
  echo -e "${YELLOW}Installing frontend dependencies...${NC}"
  cd "$PROJECT_ROOT/frontend" && npm install
  
  echo -e "${YELLOW}Installing ML dependencies...${NC}"
  cd "$PROJECT_ROOT/ml" && pip install -r requirements.txt
  
  cd "$PROJECT_ROOT"
  echo -e "${GREEN}All dependencies installed.${NC}"
}

# Initialize database
init_database() {
  echo -e "\n${BLUE}Initializing database...${NC}"
  
  # Start PostgreSQL container if not running
  if ! docker ps | grep postgres &> /dev/null; then
    echo -e "${YELLOW}Starting PostgreSQL container...${NC}"
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
    sleep 5
  else
    echo -e "${GREEN}PostgreSQL container is already running.${NC}"
  fi
  
  # Generate Prisma client
  echo -e "${YELLOW}Generating Prisma client...${NC}"
  cd "$PROJECT_ROOT/backend" && npm run prisma:generate
  
  # Run database migrations
  echo -e "${YELLOW}Running database migrations...${NC}"
  cd "$PROJECT_ROOT/backend" && npm run prisma:migrate:dev
  
  cd "$PROJECT_ROOT"
  echo -e "${GREEN}Database initialized.${NC}"
}

# Create directory structure for ML data
setup_ml_directories() {
  echo -e "\n${BLUE}Setting up ML directories...${NC}"
  
  mkdir -p "$PROJECT_ROOT/ml/data/raw"
  mkdir -p "$PROJECT_ROOT/ml/data/processed"
  mkdir -p "$PROJECT_ROOT/ml/models"
  mkdir -p "$PROJECT_ROOT/ml/logs/tensorboard"
  
  echo -e "${GREEN}ML directories created.${NC}"
}

# Main function
main() {
  check_dependencies
  setup_env
  install_dependencies
  init_database
  setup_ml_directories
  
  echo -e "\n${GREEN}${BOLD}Development environment setup complete!${NC}"
  echo ""
  echo -e "${BOLD}Next steps:${NC}"
  echo "1. Review and update your .env file with credentials"
  echo "2. Start the development servers with: npm run dev"
  echo "3. Access the dashboard at: http://localhost:3000"
  echo ""
  echo -e "${YELLOW}For more information, see the README.md file.${NC}"
}

# Run the main function
main 