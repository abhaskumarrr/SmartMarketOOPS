#!/bin/bash

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}====================================================${NC}"
echo -e "${CYAN}      Running Bridge API Integration Tests           ${NC}"
echo -e "${CYAN}====================================================${NC}"

# Run Jest tests with the bridge.test.ts file
npx jest --config=jest.config.js --runInBand tests/integration/bridge.test.ts

# Check the exit status
if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Bridge API tests completed successfully!${NC}"
else
  echo -e "${RED}✗ Bridge API tests failed!${NC}"
  exit 1
fi

echo -e "${YELLOW}Tests complete.${NC}"
exit 0 