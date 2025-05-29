# Delta Exchange API Testing Suite

This directory contains comprehensive tests for the Delta Exchange API integration. The tests are organized into different categories to cover all aspects of the API functionality.

## Test Structure

- **Unit Tests**: Located in `tests/unit/`, these test individual components in isolation with mocked dependencies.
- **Integration Tests**: Located in `tests/integration/`, these test the interaction between components.
- **Mock Services**: Located in `tests/mock/`, these provide mock implementations of external services.

## Running Tests

```bash
# Run all tests
npm test

# Run only unit tests
npm run test:unit

# Run only integration tests
npm run test:integration

# Run tests with coverage report
npm run test:coverage

# Run tests in watch mode (useful during development)
npm run test:watch
```

## Test Files

### Unit Tests

- **deltaApiService.test.js**: Tests the Delta Exchange API client service, covering market data, trading, and error handling.
- **deltaWebSocketService.test.js**: Tests the WebSocket connection for real-time data, including connection handling, channel subscriptions, and message processing.
- **deltaRateLimit.test.js**: Tests the rate limiting and backoff mechanisms to ensure robust handling of API limits.

### Integration Tests

- **deltaApiController.test.js**: Tests the API controllers that handle HTTP requests, validating the integration between controllers and services.

### Mock Services

- **deltaApiMock.js**: Provides mock implementations of the Delta Exchange API for offline testing.

## Testing Strategy

1. **Unit Testing**:
   - Test each API endpoint method in isolation
   - Verify correct request formatting
   - Test error handling for various response codes
   - Test retry logic and rate limiting
   - Test WebSocket connection and event handling

2. **Integration Testing**:
   - Test route handling and middleware integration
   - Test request validation
   - Test error handling across component boundaries
   - Test authentication flows

3. **Error Scenario Testing**:
   - Test API response errors (400, 401, 403, 404, 429, 500)
   - Test network failures (timeouts, connection resets)
   - Test malformed responses
   - Test rate limiting behavior

4. **Mock Services**:
   - Provide consistent testing data
   - Allow testing without external dependencies
   - Simulate various error conditions

## Coverage Goals

The test suite aims to achieve:

- Line coverage: >90%
- Branch coverage: >85%
- Function coverage: >95%

## Environment

Tests run with a dedicated test environment configuration specified in `test.env`. This ensures that tests don't interact with production systems. 