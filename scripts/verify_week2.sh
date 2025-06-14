#!/bin/bash

# Verification script for Week 2 components
# This script tests if all Week 2 components are properly installed and running

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}SmartMarketOOPS Week 2 Verification Script${NC}"
echo "==============================================="

# Check if the environment file exists
if [ -f .env ]; then
    echo -e "${GREEN}✓ Environment file exists${NC}"
else
    echo -e "${RED}✗ Environment file missing. Please run: cp example.env .env${NC}"
    exit 1
fi

# Function to check if a service is running on a port
check_service() {
    local port=$1
    local service_name=$2
    
    # Use curl to check if the service is responding
    if curl -s "http://localhost:$port" > /dev/null; then
        echo -e "${GREEN}✓ $service_name is running on port $port${NC}"
        return 0
    else
        echo -e "${RED}✗ $service_name is not running on port $port${NC}"
        return 1
    fi
}

# Check for required Python modules
echo -e "\n${YELLOW}Checking Python environment...${NC}"
python_modules=("numpy" "pandas" "websockets" "aiohttp" "asyncio" "ccxt")
all_modules_installed=true

for module in "${python_modules[@]}"; do
    if python -c "import $module" 2>/dev/null; then
        echo -e "${GREEN}✓ Python module $module is installed${NC}"
    else
        echo -e "${RED}✗ Python module $module is missing. Install it with: pip install $module${NC}"
        all_modules_installed=false
    fi
done

if [ "$all_modules_installed" = false ]; then
    echo -e "${YELLOW}Run: pip install -r ml/requirements.txt to install all required Python modules${NC}"
fi

# Check backend and ML services
echo -e "\n${YELLOW}Checking services...${NC}"
backend_running=false
ml_running=false

if check_service 3006 "Backend service"; then
    backend_running=true
fi

if check_service 8000 "ML service"; then
    ml_running=true
fi

# Check for Week 2 specific components
echo -e "\n${YELLOW}Verifying Week 2 components...${NC}"

# Check if real market data service file exists
if [ -f "ml/src/data/real_market_data_service.py" ]; then
    echo -e "${GREEN}✓ Real market data service file exists${NC}"
else
    echo -e "${RED}✗ Real market data service file is missing${NC}"
fi

# Check if multi symbol manager file exists
if [ -f "ml/src/trading/multi_symbol_manager.py" ]; then
    echo -e "${GREEN}✓ Multi-symbol trading manager file exists${NC}"
else
    echo -e "${RED}✗ Multi-symbol trading manager file is missing${NC}"
fi

# Check if advanced risk manager file exists
if [ -f "ml/src/risk/advanced_risk_manager.py" ]; then
    echo -e "${GREEN}✓ Advanced risk manager file exists${NC}"
else
    echo -e "${RED}✗ Advanced risk manager file is missing${NC}"
fi

# Check if the week2 integration launcher exists
if [ -f "ml/week2_integration_launcher.py" ]; then
    echo -e "${GREEN}✓ Week 2 integration launcher file exists${NC}"
else
    echo -e "${RED}✗ Week 2 integration launcher file is missing${NC}"
fi

# Check for documentation files
echo -e "\n${YELLOW}Checking documentation...${NC}"

if [ -f "docs/INSTALLATION_GUIDE.md" ]; then
    echo -e "${GREEN}✓ Installation guide exists${NC}"
else
    echo -e "${RED}✗ Installation guide is missing${NC}"
fi

if [ -f "docs/TROUBLESHOOTING.md" ]; then
    echo -e "${GREEN}✓ Troubleshooting guide exists${NC}"
else
    echo -e "${RED}✗ Troubleshooting guide is missing${NC}"
fi

# Test API endpoints if backend is running
if [ "$backend_running" = true ]; then
    echo -e "\n${YELLOW}Testing API endpoints...${NC}"
    
    # Test health endpoint
    if curl -s "http://localhost:3006/api/health" | grep -q "status"; then
        echo -e "${GREEN}✓ Health API endpoint working${NC}"
    else
        echo -e "${RED}✗ Health API endpoint not working${NC}"
    fi
    
    # Test Delta Exchange endpoint
    if curl -s "http://localhost:3006/api/delta-trading/health" | grep -q "status"; then
        echo -e "${GREEN}✓ Delta Exchange API endpoint working${NC}"
    else
        echo -e "${RED}✗ Delta Exchange API endpoint not working${NC}"
    fi
fi

# Test ML endpoints if running
if [ "$ml_running" = true ]; then
    echo -e "\n${YELLOW}Testing ML endpoints...${NC}"
    
    # Test health endpoint
    if curl -s "http://localhost:8000/health" | grep -q "status"; then
        echo -e "${GREEN}✓ ML service health endpoint working${NC}"
    else
        echo -e "${RED}✗ ML service health endpoint not working${NC}"
    fi
fi

# Verify multi-symbol configuration
echo -e "\n${YELLOW}Checking multi-symbol configuration...${NC}"
if [ -f "ml/src/config/symbols.json" ]; then
    echo -e "${GREEN}✓ Multi-symbol configuration file exists${NC}"
    
    # Check if it contains expected symbols
    if grep -q "BTCUSDT" "ml/src/config/symbols.json" && \
       grep -q "ETHUSDT" "ml/src/config/symbols.json"; then
        echo -e "${GREEN}✓ Multi-symbol configuration contains expected symbols${NC}"
    else
        echo -e "${RED}✗ Multi-symbol configuration may be missing expected symbols${NC}"
    fi
else
    echo -e "${RED}✗ Multi-symbol configuration file is missing${NC}"
fi

# Final summary
echo -e "\n${YELLOW}Week 2 Verification Summary${NC}"
echo "==============================================="

if [ "$backend_running" = true ] && [ "$ml_running" = true ] && \
   [ -f "ml/src/data/real_market_data_service.py" ] && \
   [ -f "ml/src/trading/multi_symbol_manager.py" ] && \
   [ -f "ml/src/risk/advanced_risk_manager.py" ] && \
   [ -f "ml/week2_integration_launcher.py" ] && \
   [ -f "docs/INSTALLATION_GUIDE.md" ] && \
   [ -f "docs/TROUBLESHOOTING.md" ]; then
    echo -e "${GREEN}All Week 2 components are installed and services are running${NC}"
    echo -e "You can now use the system with the enhanced Week 2 features!"
else
    echo -e "${RED}Some Week 2 components are missing or services are not running${NC}"
    echo -e "Please fix the issues highlighted above to ensure proper functioning."
fi

echo -e "\n${YELLOW}To start all Week 2 components:${NC}"
echo "1. Terminal 1: cd ml && python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"
echo "2. Terminal 2: cd ml && python week2_integration_launcher.py"
echo "3. Terminal 3: cd backend && npm run dev"
echo "4. Terminal 4: cd frontend && npm start" 