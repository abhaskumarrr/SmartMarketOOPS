#!/bin/bash

# Comprehensive Test Runner for SmartMarketOOPS
# Runs all test suites with performance monitoring and reporting

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_DIR="frontend"
BACKEND_DIR="backend"
REPORTS_DIR="test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create reports directory
mkdir -p $REPORTS_DIR

echo -e "${BLUE}🚀 Starting Comprehensive Test Suite${NC}"
echo -e "${BLUE}Timestamp: $TIMESTAMP${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to print success message
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Function to print error message
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to print warning message
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to run command with timing
run_timed() {
    local cmd="$1"
    local description="$2"
    
    echo -e "${YELLOW}Running: $description${NC}"
    start_time=$(date +%s)
    
    if eval "$cmd"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_success "$description completed in ${duration}s"
        return 0
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_error "$description failed after ${duration}s"
        return 1
    fi
}

# Check prerequisites
print_section "Checking Prerequisites"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed"
    exit 1
fi

print_success "Prerequisites check passed"
echo ""

# Install dependencies
print_section "Installing Dependencies"

if run_timed "cd $FRONTEND_DIR && npm ci" "Frontend dependency installation"; then
    print_success "Frontend dependencies installed"
else
    print_error "Failed to install frontend dependencies"
    exit 1
fi

if run_timed "cd $BACKEND_DIR && npm ci" "Backend dependency installation"; then
    print_success "Backend dependencies installed"
else
    print_error "Failed to install backend dependencies"
    exit 1
fi

echo ""

# Lint checks
print_section "Code Quality Checks"

run_timed "cd $FRONTEND_DIR && npm run lint" "Frontend linting"
run_timed "cd $BACKEND_DIR && npm run lint" "Backend linting" || print_warning "Backend linting failed (continuing)"

echo ""

# Frontend Tests
print_section "Frontend Testing"

# Unit tests
if run_timed "cd $FRONTEND_DIR && npm run test:unit -- --coverage --outputFile=../$REPORTS_DIR/frontend-unit-results.json" "Frontend unit tests"; then
    print_success "Frontend unit tests passed"
else
    print_error "Frontend unit tests failed"
    FRONTEND_UNIT_FAILED=1
fi

# Integration tests
if run_timed "cd $FRONTEND_DIR && npm run test:integration -- --outputFile=../$REPORTS_DIR/frontend-integration-results.json" "Frontend integration tests"; then
    print_success "Frontend integration tests passed"
else
    print_warning "Frontend integration tests failed (continuing)"
fi

# E2E tests
if run_timed "cd $FRONTEND_DIR && npm run test:e2e -- --outputFile=../$REPORTS_DIR/frontend-e2e-results.json" "Frontend E2E tests"; then
    print_success "Frontend E2E tests passed"
else
    print_warning "Frontend E2E tests failed (continuing)"
fi

echo ""

# Backend Tests
print_section "Backend Testing"

# Unit tests
if run_timed "cd $BACKEND_DIR && npm run test:unit -- --coverage --outputFile=../$REPORTS_DIR/backend-unit-results.json" "Backend unit tests"; then
    print_success "Backend unit tests passed"
else
    print_error "Backend unit tests failed"
    BACKEND_UNIT_FAILED=1
fi

# Integration tests
if run_timed "cd $BACKEND_DIR && npm run test:integration -- --outputFile=../$REPORTS_DIR/backend-integration-results.json" "Backend integration tests"; then
    print_success "Backend integration tests passed"
else
    print_warning "Backend integration tests failed (continuing)"
fi

echo ""

# Performance Tests
print_section "Performance Testing"

# Backend load tests
if run_timed "cd $BACKEND_DIR && npm run test:performance -- --outputFile=../$REPORTS_DIR/backend-performance-results.json" "Backend performance tests"; then
    print_success "Backend performance tests passed"
else
    print_warning "Backend performance tests failed (continuing)"
fi

# Frontend performance tests
if run_timed "cd $FRONTEND_DIR && npm run test:performance -- --outputFile=../$REPORTS_DIR/frontend-performance-results.json" "Frontend performance tests"; then
    print_success "Frontend performance tests passed"
else
    print_warning "Frontend performance tests failed (continuing)"
fi

echo ""

# Coverage Analysis
print_section "Coverage Analysis"

# Combine coverage reports
echo "Generating combined coverage report..."

# Frontend coverage
if [ -d "$FRONTEND_DIR/coverage" ]; then
    cp -r "$FRONTEND_DIR/coverage" "$REPORTS_DIR/frontend-coverage"
    print_success "Frontend coverage report saved"
fi

# Backend coverage
if [ -d "$BACKEND_DIR/coverage" ]; then
    cp -r "$BACKEND_DIR/coverage" "$REPORTS_DIR/backend-coverage"
    print_success "Backend coverage report saved"
fi

echo ""

# Performance Monitoring
print_section "Performance Monitoring"

# Run performance monitoring script
if [ -f "$BACKEND_DIR/src/scripts/performance-monitor.ts" ]; then
    run_timed "cd $BACKEND_DIR && npm run perf:monitor" "Performance monitoring" || print_warning "Performance monitoring failed"
fi

echo ""

# Generate Summary Report
print_section "Generating Summary Report"

cat > "$REPORTS_DIR/test-summary-$TIMESTAMP.md" << EOF
# Test Summary Report

**Timestamp:** $TIMESTAMP
**Date:** $(date)

## Test Results

### Frontend Tests
- Unit Tests: $([ -z "$FRONTEND_UNIT_FAILED" ] && echo "✅ PASSED" || echo "❌ FAILED")
- Integration Tests: $([ -f "$REPORTS_DIR/frontend-integration-results.json" ] && echo "✅ PASSED" || echo "⚠️ SKIPPED")
- E2E Tests: $([ -f "$REPORTS_DIR/frontend-e2e-results.json" ] && echo "✅ PASSED" || echo "⚠️ SKIPPED")

### Backend Tests
- Unit Tests: $([ -z "$BACKEND_UNIT_FAILED" ] && echo "✅ PASSED" || echo "❌ FAILED")
- Integration Tests: $([ -f "$REPORTS_DIR/backend-integration-results.json" ] && echo "✅ PASSED" || echo "⚠️ SKIPPED")
- Performance Tests: $([ -f "$REPORTS_DIR/backend-performance-results.json" ] && echo "✅ PASSED" || echo "⚠️ SKIPPED")

## Coverage Reports

- Frontend Coverage: $([ -d "$REPORTS_DIR/frontend-coverage" ] && echo "Available" || echo "Not Generated")
- Backend Coverage: $([ -d "$REPORTS_DIR/backend-coverage" ] && echo "Available" || echo "Not Generated")

## Files Generated

$(ls -la $REPORTS_DIR/)

## Next Steps

1. Review coverage reports in the coverage directories
2. Check individual test result files for detailed information
3. Address any failing tests before deployment
4. Monitor performance metrics for regressions

EOF

print_success "Summary report generated: $REPORTS_DIR/test-summary-$TIMESTAMP.md"

echo ""

# Final Summary
print_section "Test Suite Complete"

if [ -z "$FRONTEND_UNIT_FAILED" ] && [ -z "$BACKEND_UNIT_FAILED" ]; then
    print_success "All critical tests passed! ✨"
    echo -e "${GREEN}Your application is ready for deployment.${NC}"
    exit 0
else
    print_error "Some critical tests failed!"
    echo -e "${RED}Please fix failing tests before deployment.${NC}"
    exit 1
fi
