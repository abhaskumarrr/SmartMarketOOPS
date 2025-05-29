/**
 * Test setup file
 * Runs before each test
 */

// Load environment variables from test.env file if it exists
require('dotenv').config({ path: './test.env' });

// Configure global test timeouts
jest.setTimeout(30000); // 30 seconds

// Global setup code
beforeAll(async () => {
  // Any global setup before all tests run
  console.log('Starting test suite...');
});

// Global teardown code
afterAll(async () => {
  // Any global cleanup after all tests run
  console.log('Test suite completed.');
}); 