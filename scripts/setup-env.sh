#!/bin/bash
#
# Setup environment script for SMOOPs trading bot
# This script helps with environment configuration for different setups

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
NC="\033[0m" # No Color

# Check if running in development environment
is_dev_env() {
  if [ -z "$NODE_ENV" ] || [ "$NODE_ENV" = "development" ]; then
    return 0
  else
    return 1
  fi
}

# Print header
echo -e "${BOLD}SMOOPs Trading Bot Environment Setup${NC}"
echo "=================================="
echo ""

# Check if .env exists, create if not
if [ ! -f "$ENV_FILE" ]; then
  echo -e "${YELLOW}No .env file found. Creating from example.env...${NC}"
  cp "$EXAMPLE_ENV" "$ENV_FILE"
  echo -e "${GREEN}Created .env file. Please edit it with your credentials.${NC}"
else
  echo -e "${GREEN}Found existing .env file.${NC}"
fi

# Choose environment
echo ""
echo -e "${BOLD}Available environments:${NC}"
echo "1) Development (default)"
echo "2) Production"
echo "3) Testing"
read -p "Select environment [1]: " ENV_CHOICE

case "$ENV_CHOICE" in
  2)
    ENV_TYPE="production"
    ;;
  3)
    ENV_TYPE="testing"
    ;;
  *)
    ENV_TYPE="development"
    ;;
esac

echo -e "${GREEN}Setting up for ${BOLD}$ENV_TYPE${NC} ${GREEN}environment.${NC}"

# Update NODE_ENV in .env file
if grep -q "NODE_ENV=" "$ENV_FILE"; then
  sed -i.bak "s/NODE_ENV=.*/NODE_ENV=$ENV_TYPE/" "$ENV_FILE" && rm "$ENV_FILE.bak"
else
  echo "NODE_ENV=$ENV_TYPE" >> "$ENV_FILE"
fi

# For non-development environments, remind to update keys
if [ "$ENV_TYPE" != "development" ]; then
  echo ""
  echo -e "${YELLOW}IMPORTANT: For $ENV_TYPE environments, ensure you update:${NC}"
  echo " - API keys"
  echo " - Database credentials"
  echo " - Encryption keys"
  echo " - Any other sensitive information"
  echo ""
  echo -e "${BOLD}Security reminder:${NC} Never commit .env files to version control."
fi

# Check for required API keys
echo ""
echo -e "${BOLD}Checking for required API keys...${NC}"

missing_keys=0

# Check Delta Exchange keys
if ! grep -q "DELTA_EXCHANGE_API_KEY=.\+" "$ENV_FILE"; then
  echo -e "${YELLOW}⚠️  Missing Delta Exchange API key${NC}"
  missing_keys=$((missing_keys+1))
fi

if ! grep -q "DELTA_EXCHANGE_API_SECRET=.\+" "$ENV_FILE"; then
  echo -e "${YELLOW}⚠️  Missing Delta Exchange API secret${NC}"
  missing_keys=$((missing_keys+1))
fi

if [ $missing_keys -eq 0 ]; then
  echo -e "${GREEN}✓ All required API keys are set${NC}"
else
  echo -e "${YELLOW}⚠️  $missing_keys required API keys are missing. Please update your .env file.${NC}"
fi

echo ""
echo -e "${GREEN}${BOLD}Environment setup complete!${NC}"
echo ""
echo "Your environment is configured for: $ENV_TYPE"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "1. Start the services with: docker-compose up -d"
echo "2. Access the dashboard at: http://localhost:3000"
echo "" 