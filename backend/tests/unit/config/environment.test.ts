/**
 * Environment Configuration Tests
 * Tests the centralized environment configuration module
 */

import env from '../../../src/config/environment';

describe('Environment Configuration', () => {
  // Store original environment variables
  const originalEnv = { ...process.env };

  // Restore original environment variables after tests
  afterAll(() => {
    process.env = { ...originalEnv };
  });

  // Set up test environment variables before each test
  beforeEach(() => {
    // Reset environment variables
    process.env = { ...originalEnv };
    
    // Set test environment variables
    process.env.NODE_ENV = 'test';
    process.env.PORT = '4000';
    process.env.TRADING_MODE = 'test';
    process.env.FORCE_TESTNET = 'true';
    process.env.DATABASE_URL = 'postgresql://test:test@localhost:5432/test';
    process.env.REDIS_HOST = 'test-redis';
    process.env.REDIS_PORT = '6380';
    process.env.JWT_SECRET = 'test-jwt-secret';
    process.env.DELTA_EXCHANGE_API_KEY = 'test-api-key';
    process.env.DELTA_EXCHANGE_API_SECRET = 'test-api-secret';
    process.env.DELTA_EXCHANGE_TESTNET = 'true';
    
    // Clear the module cache to reload environment configuration
    jest.resetModules();
  });

  it('should load environment variables correctly', () => {
    // Import the module again to reload with new environment variables
    const env = require('../../../src/config/environment').default;
    
    // Test basic environment variables
    expect(env.NODE_ENV).toBe('test');
    expect(env.PORT).toBe(4000);
    expect(env.TRADING_MODE).toBe('test');
    expect(env.FORCE_TESTNET).toBe(true);
    
    // Test database configuration
    expect(env.DATABASE_URL).toBe('postgresql://test:test@localhost:5432/test');
    expect(env.REDIS_HOST).toBe('test-redis');
    expect(env.REDIS_PORT).toBe(6380);
    
    // Test security configuration
    expect(env.JWT_SECRET).toBe('test-jwt-secret');
    
    // Test Delta Exchange configuration
    expect(env.DELTA_EXCHANGE_API_KEY).toBe('test-api-key');
    expect(env.DELTA_EXCHANGE_API_SECRET).toBe('test-api-secret');
    expect(env.DELTA_EXCHANGE_TESTNET).toBe(true);
  });

  it('should provide default values for missing environment variables', () => {
    // Clear specific environment variables
    delete process.env.PORT;
    delete process.env.REDIS_HOST;
    delete process.env.REDIS_PORT;
    
    // Import the module again to reload with new environment variables
    const env = require('../../../src/config/environment').default;
    
    // Test default values
    expect(env.PORT).toBe(3006); // Default port
    expect(env.REDIS_HOST).toBe('localhost'); // Default Redis host
    expect(env.REDIS_PORT).toBe(6379); // Default Redis port
  });

  it('should correctly parse boolean environment variables', () => {
    // Set boolean environment variables
    process.env.FORCE_TESTNET = 'false';
    process.env.DELTA_EXCHANGE_TESTNET = 'false';
    process.env.ENABLE_METRICS = 'true';
    
    // Import the module again to reload with new environment variables
    const env = require('../../../src/config/environment').default;
    
    // Test boolean parsing
    expect(env.FORCE_TESTNET).toBe(false);
    expect(env.DELTA_EXCHANGE_TESTNET).toBe(false);
    expect(env.ENABLE_METRICS).toBe(true);
  });

  it('should correctly parse numeric environment variables', () => {
    // Set numeric environment variables
    process.env.PORT = '5000';
    process.env.REDIS_PORT = '7000';
    process.env.ML_BATCH_SIZE = '50';
    
    // Import the module again to reload with new environment variables
    const env = require('../../../src/config/environment').default;
    
    // Test numeric parsing
    expect(env.PORT).toBe(5000);
    expect(env.REDIS_PORT).toBe(7000);
    expect(env.ML_BATCH_SIZE).toBe(50);
  });

  it('should construct Delta Exchange base URL correctly', () => {
    // Test with testnet = true
    process.env.DELTA_EXCHANGE_TESTNET = 'true';
    let env = require('../../../src/config/environment').default;
    expect(env.DELTA_EXCHANGE_BASE_URL).toBe('https://cdn-ind.testnet.deltaex.org');
    
    // Test with testnet = false
    process.env.DELTA_EXCHANGE_TESTNET = 'false';
    env = require('../../../src/config/environment').default;
    expect(env.DELTA_EXCHANGE_BASE_URL).toBe('https://api.india.delta.exchange');
    
    // Test with custom base URL
    process.env.DELTA_EXCHANGE_BASE_URL = 'https://custom.delta.exchange';
    env = require('../../../src/config/environment').default;
    expect(env.DELTA_EXCHANGE_BASE_URL).toBe('https://custom.delta.exchange');
  });
});