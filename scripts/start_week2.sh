#!/bin/bash

# Start Week 2 components script
# This script starts all Week 2 components in the correct order

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting SmartMarketOOPS Week 2 Components${NC}"
echo "=================================================="

# Check if the environment file exists
if [ ! -f .env ]; then
    echo -e "${RED}Environment file (.env) not found. Creating from example.env...${NC}"
    cp example.env .env
    echo -e "${GREEN}Created .env file from example.env. Please update with your API keys.${NC}"
fi

# Create required directories if they don't exist
mkdir -p data/logs
mkdir -p ml/models/checkpoints
mkdir -p frontend/public/data

# Detect operating system for terminal commands
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    TERMINAL_CMD="osascript -e 'tell app \"Terminal\" to do script \"cd $(pwd) && "
    TERMINAL_END="\"'"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux with gnome-terminal
    if command -v gnome-terminal &> /dev/null; then
        TERMINAL_CMD="gnome-terminal -- bash -c \"cd $(pwd) && "
        TERMINAL_END="; exec bash\""
    # Linux with xterm
    elif command -v xterm &> /dev/null; then
        TERMINAL_CMD="xterm -e \"cd $(pwd) && "
        TERMINAL_END="; exec bash\" &"
    else
        echo -e "${RED}Could not find a suitable terminal emulator. Please manually start the components.${NC}"
        exit 1
    fi
else
    echo -e "${RED}Unsupported operating system. Please manually start the components.${NC}"
    echo -e "${YELLOW}Terminal 1: cd ml && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000${NC}"
    echo -e "${YELLOW}Terminal 2: cd ml && python week2_integration_launcher.py${NC}"
    echo -e "${YELLOW}Terminal 3: cd backend && npm run dev${NC}"
    echo -e "${YELLOW}Terminal 4: cd frontend && npm start${NC}"
    exit 1
fi

# Function to check if a port is already in use
is_port_in_use() {
    if command -v lsof &> /dev/null; then
        lsof -i:"$1" &> /dev/null
        return $?
    elif command -v netstat &> /dev/null; then
        netstat -tuln | grep -q ":$1 "
        return $?
    else
        # If we can't check, assume it's not in use
        return 1
    fi
}

# Check if ports are already in use
if is_port_in_use 8000; then
    echo -e "${RED}Port 8000 is already in use. Please stop the process using this port.${NC}"
    exit 1
fi

if is_port_in_use 3006; then
    echo -e "${RED}Port 3006 is already in use. Please stop the process using this port.${NC}"
    exit 1
fi

if is_port_in_use 3000; then
    echo -e "${RED}Port 3000 is already in use. Please stop the process using this port.${NC}"
    exit 1
fi

# Start ML Service
echo -e "${YELLOW}Starting ML Service on port 8000...${NC}"
eval "${TERMINAL_CMD}cd ml && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000${TERMINAL_END}"

# Wait for ML Service to start
echo -e "${YELLOW}Waiting for ML Service to start...${NC}"
sleep 5

# Start Week 2 Integration Manager
echo -e "${YELLOW}Starting Week 2 Integration Manager...${NC}"
eval "${TERMINAL_CMD}cd ml && python week2_integration_launcher.py${TERMINAL_END}"

# Wait for Integration Manager to start
echo -e "${YELLOW}Waiting for Integration Manager to start...${NC}"
sleep 3

# Start Backend
echo -e "${YELLOW}Starting Backend on port 3006...${NC}"
eval "${TERMINAL_CMD}cd backend && npm run dev${TERMINAL_END}"

# Wait for Backend to start
echo -e "${YELLOW}Waiting for Backend to start...${NC}"
sleep 5

# Start Frontend
echo -e "${YELLOW}Starting Frontend on port 3000...${NC}"
eval "${TERMINAL_CMD}cd frontend && npm start${TERMINAL_END}"

echo -e "\n${GREEN}All Week 2 components started!${NC}"
echo -e "${YELLOW}You can access the application at:${NC}"
echo -e "  - Frontend: ${GREEN}http://localhost:3000${NC}"
echo -e "  - Backend API: ${GREEN}http://localhost:3006${NC}"
echo -e "  - ML Service: ${GREEN}http://localhost:8000${NC}"
echo -e "\n${YELLOW}To verify all components are working correctly, run:${NC}"
echo -e "${GREEN}./scripts/verify_week2.sh${NC}" 