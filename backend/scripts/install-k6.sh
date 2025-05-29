#!/bin/bash

# Script to install k6 load testing tool
# https://k6.io/docs/getting-started/installation/

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN}      Installing k6 Load Testing Tool              ${NC}"
echo -e "${CYAN}====================================================${NC}"

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo -e "${YELLOW}Detected Linux OS${NC}"
  
  # Check if we're on Ubuntu/Debian
  if [ -f /etc/debian_version ]; then
    echo -e "${YELLOW}Detected Debian/Ubuntu - Installing via apt${NC}"
    
    # Add the k6 repository
    echo -e "${YELLOW}Adding k6 repository...${NC}"
    sudo apt-get update
    sudo apt-get install -y gnupg2 ca-certificates
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
    echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
    
    # Install k6
    echo -e "${YELLOW}Installing k6...${NC}"
    sudo apt-get update
    sudo apt-get install -y k6
  
  # Check if we're on RHEL/CentOS/Fedora
  elif [ -f /etc/redhat-release ]; then
    echo -e "${YELLOW}Detected RHEL/CentOS/Fedora - Installing via yum${NC}"
    
    # Add the k6 repository
    echo -e "${YELLOW}Adding k6 repository...${NC}"
    sudo dnf install -y dnf-plugins-core
    sudo dnf config-manager --add-repo https://dl.k6.io/rpm/repo.rpm.gpg
    
    # Install k6
    echo -e "${YELLOW}Installing k6...${NC}"
    sudo dnf install -y k6
  
  # Other Linux distributions - use the binary
  else
    echo -e "${YELLOW}Linux distribution not specifically supported - Installing binary directly${NC}"
    echo -e "${YELLOW}Downloading k6 binary...${NC}"
    curl -L https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz -o k6.tar.gz
    
    echo -e "${YELLOW}Extracting and installing...${NC}"
    tar -xzf k6.tar.gz
    sudo cp k6-v0.45.0-linux-amd64/k6 /usr/local/bin/
    rm -rf k6-v0.45.0-linux-amd64 k6.tar.gz
  fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo -e "${YELLOW}Detected macOS - Installing via Homebrew${NC}"
  
  # Check if Homebrew is installed
  if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew not found. Please install Homebrew first:${NC}"
    echo -e "${YELLOW}/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${NC}"
    exit 1
  fi
  
  # Install k6
  echo -e "${YELLOW}Installing k6...${NC}"
  brew install k6

elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  echo -e "${YELLOW}Detected Windows - Please install k6 manually:${NC}"
  echo -e "${YELLOW}1. Using Chocolatey: choco install k6${NC}"
  echo -e "${YELLOW}2. Or download from https://dl.k6.io/msi/k6-latest-amd64.msi${NC}"
  exit 1

else
  echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
  echo -e "${YELLOW}Please install k6 manually: https://k6.io/docs/getting-started/installation/${NC}"
  exit 1
fi

# Verify installation
if command -v k6 &> /dev/null; then
  echo -e "${GREEN}✓ k6 installed successfully!${NC}"
  k6 version
  
  echo -e "\n${YELLOW}Example k6 usage:${NC}"
  echo -e "${CYAN}Run a simple test:${NC} k6 run scripts/test.js"
  echo -e "${CYAN}Run with 10 virtual users:${NC} k6 run --vus 10 --duration 30s scripts/test.js"
  echo -e "${CYAN}Run load test through the performance API:${NC} curl -X POST http://localhost:3001/api/performance/load-test -H 'Content-Type: application/json' -d '{\"name\":\"Test\",\"stages\":[{\"duration\":30,\"target\":10}],\"targetEndpoints\":[\"/api/health\"]}')"
else
  echo -e "${RED}✗ k6 installation failed!${NC}"
  exit 1
fi

exit 0 