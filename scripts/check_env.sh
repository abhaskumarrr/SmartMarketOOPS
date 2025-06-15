#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if a variable exists in .env file
check_var() {
    local var_name=$1
    local env_file=$2
    if grep -q "^${var_name}=" "$env_file"; then
        echo -e "${GREEN}✓ $var_name${NC}"
        return 0
    else
        echo -e "${RED}✗ $var_name${NC}"
        return 1
    fi
}

# Function to compare .env with .env.example
check_env_file() {
    local env_file=$1
    local example_file=$2
    local missing_vars=0
    
    echo -e "\n${YELLOW}Checking $env_file against $example_file${NC}"
    
    if [ ! -f "$env_file" ]; then
        echo -e "${RED}Error: $env_file does not exist${NC}"
        return 1
    fi
    
    if [ ! -f "$example_file" ]; then
        echo -e "${RED}Error: $example_file does not exist${NC}"
        return 1
    fi
    
    # Get all variable names from .env.example
    local vars=$(grep -v '^#' "$example_file" | grep '=' | cut -d '=' -f1)
    
    for var in $vars; do
        if ! check_var "$var" "$env_file"; then
            ((missing_vars++))
        fi
    done
    
    if [ $missing_vars -eq 0 ]; then
        echo -e "${GREEN}All variables from $example_file are present in $env_file${NC}"
    else
        echo -e "${RED}Missing $missing_vars variables in $env_file${NC}"
    fi
    
    # Check for empty values
    echo -e "\n${YELLOW}Checking for empty values in $env_file${NC}"
    local empty_vars=$(grep -v '^#' "$env_file" | grep '=$' || true)
    if [ ! -z "$empty_vars" ]; then
        echo -e "${RED}Warning: The following variables have empty values:${NC}"
        echo "$empty_vars"
    else
        echo -e "${GREEN}No empty values found in $env_file${NC}"
    fi
}

# Main execution
echo -e "${YELLOW}Starting environment configuration check...${NC}"

# Check root .env
if [ -f ".env.example" ]; then
    check_env_file ".env" ".env.example"
fi

# Check backend .env
if [ -f "backend/.env.example" ]; then
    check_env_file "backend/.env" "backend/.env.example"
fi

# Check frontend .env files
if [ -f "frontend/.env.development" ]; then
    if [ -f "frontend/.env" ]; then
        check_env_file "frontend/.env" "frontend/.env.development"
    fi
    
    if [ -f "frontend/.env.production" ]; then
        echo -e "\n${YELLOW}Checking production environment variables...${NC}"
        check_env_file "frontend/.env.production" "frontend/.env.development"
    fi
fi

# Check Docker environment
if [ -f ".env.docker" ]; then
    if [ -f ".env.example" ]; then
        echo -e "\n${YELLOW}Checking Docker environment variables...${NC}"
        check_env_file ".env.docker" ".env.example"
    fi
fi 