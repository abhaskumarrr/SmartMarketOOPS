#!/bin/bash
# Script to run all risk management related tests

# Set environment to test
export NODE_ENV=test

# Navigate to backend directory
cd "$(dirname "$0")/.."

# Define test files
UNIT_TESTS=(
  "tests/unit/trading/riskManagementService.test.ts"
  "tests/unit/trading/riskAssessmentService.test.ts"
  "tests/unit/trading/circuitBreakerService.test.ts"
  "tests/unit/controllers/riskController.test.ts"
)

INTEGRATION_TESTS=(
  "tests/integration/riskManagement.test.ts"
)

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== Running Risk Management Unit Tests =====${NC}"
for test in "${UNIT_TESTS[@]}"; do
  if [ -f "$test" ]; then
    echo -e "${YELLOW}Running test: ${test}${NC}"
    npx jest "$test" --verbose
    if [ $? -ne 0 ]; then
      echo -e "${RED}✗ Unit test failed: ${test}${NC}"
      exit 1
    else
      echo -e "${GREEN}✓ Unit test passed: ${test}${NC}"
    fi
  else
    echo -e "${RED}✗ Test file not found: ${test}${NC}"
    exit 1
  fi
done

echo -e "\n${YELLOW}===== Running Risk Management Integration Tests =====${NC}"
for test in "${INTEGRATION_TESTS[@]}"; do
  if [ -f "$test" ]; then
    echo -e "${YELLOW}Running test: ${test}${NC}"
    npx jest "$test" --verbose
    if [ $? -ne 0 ]; then
      echo -e "${RED}✗ Integration test failed: ${test}${NC}"
      exit 1
    else
      echo -e "${GREEN}✓ Integration test passed: ${test}${NC}"
    fi
  else
    echo -e "${RED}✗ Test file not found: ${test}${NC}"
    exit 1
  fi
done

echo -e "\n${GREEN}===== All Risk Management Tests Passed! =====${NC}"
exit 0 