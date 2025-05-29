/**
 * Unit tests for Delta Exchange API Rate Limiting and Backoff Mechanism
 */
const { createDeltaApiMock } = require('../mock/deltaApiMock');
const DeltaExchangeAPI = require('../../src/services/deltaApiService').default;

// Mock the apiKeyService
jest.mock('../../src/services/apiKeyService', () => ({
  getApiKey: jest.fn().mockResolvedValue({
    key: 'test-api-key',
    secret: 'test-api-secret'
  })
}));

// Mock the logger
jest.mock('../../src/utils/logger', () => ({
  createLogger: jest.fn().mockReturnValue({
    info: jest.fn(),
    debug: jest.fn(),
    warn: jest.fn(),
    error: jest.fn()
  })
}));

describe('Delta API Rate Limiting', () => {
  let deltaApi;
  let apiMock;
  
  beforeEach(async () => {
    // Create a new instance for each test with custom rate limit settings
    deltaApi = new DeltaExchangeAPI({ 
      testnet: true,
      rateLimit: {
        maxRetries: 3,
        initialDelay: 10, // 10ms for faster tests
        maxDelay: 100,    // 100ms for faster tests
        factor: 2         // Exponential backoff factor
      }
    });
    
    await deltaApi.initialize({
      key: 'test-api-key',
      secret: 'test-api-secret'
    });
    
    // Create a fresh API mock for each test
    apiMock = createDeltaApiMock(true);
    
    // Use fake timers for controlled testing of timeouts
    jest.useFakeTimers();
  });
  
  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });
  
  describe('Retry Mechanism', () => {
    test('should retry on rate limit with exponential backoff', async () => {
      // Setup mocks - sequence of responses
      // 1st attempt: 429 Too Many Requests
      apiMock.mockRateLimitError('/v2/products');
      
      // 2nd attempt (after delay): 429 Too Many Requests
      apiMock.mockRateLimitError('/v2/products');
      
      // 3rd attempt (after longer delay): 200 OK
      apiMock.mockGetProducts();
      
      // Create a promise to resolve when the API call completes
      const resultPromise = deltaApi.getMarkets();
      
      // Advance timers to trigger first retry (initial delay: 10ms)
      jest.advanceTimersByTime(10);
      
      // Advance timers to trigger second retry (delay: 10ms * 2 = 20ms)
      jest.advanceTimersByTime(20);
      
      // Wait for the promise to resolve
      const result = await resultPromise;
      
      // Verify the successful result
      expect(result.success).toBe(true);
      expect(result.result).toBeInstanceOf(Array);
      
      // Check that the retry delays were exponential
      expect(deltaApi._calculateRetryDelay(0)).toBe(10);  // Initial delay
      expect(deltaApi._calculateRetryDelay(1)).toBe(20);  // Initial * 2
      expect(deltaApi._calculateRetryDelay(2)).toBe(40);  // Initial * 2^2
    });
    
    test('should respect max retries and fail if exceeded', async () => {
      // Setup mocks - all attempts fail with 429
      apiMock.mockRateLimitError('/v2/products');
      apiMock.mockRateLimitError('/v2/products');
      apiMock.mockRateLimitError('/v2/products');
      apiMock.mockRateLimitError('/v2/products'); // For the 4th attempt if it happens
      
      // Execute with expect to reject
      const resultPromise = expect(deltaApi.getMarkets()).rejects.toThrow('Delta Exchange API Error');
      
      // Advance timers to trigger all retries
      jest.advanceTimersByTime(10);  // First retry
      jest.advanceTimersByTime(20);  // Second retry
      jest.advanceTimersByTime(40);  // Third retry
      
      // Wait for the promise to reject
      await resultPromise;
    });
    
    test('should respect max delay setting', async () => {
      // Set a small max delay
      deltaApi.rateLimit.maxDelay = 50;
      
      // Check that delay is capped at maxDelay
      expect(deltaApi._calculateRetryDelay(0)).toBe(10);   // Initial delay
      expect(deltaApi._calculateRetryDelay(1)).toBe(20);   // Initial * 2
      expect(deltaApi._calculateRetryDelay(2)).toBe(40);   // Initial * 2^2
      expect(deltaApi._calculateRetryDelay(3)).toBe(50);   // Capped at maxDelay
      expect(deltaApi._calculateRetryDelay(4)).toBe(50);   // Still capped at maxDelay
    });
    
    test('should handle non-rate-limit errors without retry', async () => {
      // Setup a regular API error (not rate limit)
      apiMock.mockApiError('/v2/products', 400, 'Bad Request');
      
      // Should reject immediately without retry
      await expect(deltaApi.getMarkets()).rejects.toThrow('Delta Exchange API Error');
      
      // No timers should be set for retries
      expect(setTimeout).not.toHaveBeenCalled();
    });
  });
  
  describe('Request Tracking', () => {
    test('should track request count for rate limiting', async () => {
      // Setup successful responses
      apiMock.mockGetProducts();
      apiMock.mockGetProducts();
      apiMock.mockGetProducts();
      
      // Execute multiple requests
      await deltaApi.getMarkets();
      await deltaApi.getMarkets();
      await deltaApi.getMarkets();
      
      // Check request tracking
      expect(deltaApi._requestCount).toBe(3);
    });
    
    test('should reset request count after window period', async () => {
      // Setup successful response
      apiMock.mockGetProducts();
      
      // Make a request
      await deltaApi.getMarkets();
      expect(deltaApi._requestCount).toBe(1);
      
      // Advance time past the window period (default 1 minute)
      jest.advanceTimersByTime(61 * 1000);
      
      // Request count should reset
      expect(deltaApi._requestCount).toBe(0);
    });
    
    test('should implement request throttling when approaching limits', async () => {
      // Set a low request limit for testing
      deltaApi._requestLimit = 5;
      deltaApi._requestCount = 4; // Already near the limit
      
      // Setup successful response
      apiMock.mockGetProducts();
      
      // Track when request is made
      const beforeRequest = Date.now();
      
      // Execute request near limit
      const requestPromise = deltaApi.getMarkets();
      
      // Request should be throttled with a delay
      jest.advanceTimersByTime(100); // Advance time to resolve throttling
      
      // Wait for the request to complete
      await requestPromise;
      
      // Verify throttling happened
      expect(setTimeout).toHaveBeenCalled();
    });
  });
  
  describe('Error Handling', () => {
    test('should handle network errors with retry', async () => {
      // Setup network error then success
      const networkError = new Error('Network Error');
      networkError.code = 'ECONNRESET';
      
      // Mock axios to throw network error first, then succeed
      const axiosMock = require('axios');
      jest.mock('axios', () => ({
        create: jest.fn().mockReturnValue({
          request: jest.fn()
            .mockRejectedValueOnce(networkError)
            .mockResolvedValueOnce({
              data: {
                success: true,
                result: [{ symbol: 'BTCUSDT' }]
              }
            })
        })
      }));
      
      // Create a new instance with the mocked axios
      const newDeltaApi = new DeltaExchangeAPI({ testnet: true });
      await newDeltaApi.initialize({
        key: 'test-api-key',
        secret: 'test-api-secret'
      });
      
      // Execute request
      const resultPromise = newDeltaApi.getMarkets();
      
      // Advance timers to trigger retry
      jest.advanceTimersByTime(1000);
      
      // Wait for the promise to resolve
      const result = await resultPromise;
      
      // Verify successful result after retry
      expect(result.success).toBe(true);
      
      // Restore original axios
      jest.unmock('axios');
    });
    
    test('should handle timeout errors with retry', async () => {
      // Setup timeout error
      const timeoutError = new Error('Timeout');
      timeoutError.code = 'ETIMEDOUT';
      
      // Mock axios to throw timeout error first, then succeed
      const axiosMock = require('axios');
      jest.mock('axios', () => ({
        create: jest.fn().mockReturnValue({
          request: jest.fn()
            .mockRejectedValueOnce(timeoutError)
            .mockResolvedValueOnce({
              data: {
                success: true,
                result: [{ symbol: 'BTCUSDT' }]
              }
            })
        })
      }));
      
      // Create a new instance with the mocked axios
      const newDeltaApi = new DeltaExchangeAPI({ testnet: true });
      await newDeltaApi.initialize({
        key: 'test-api-key',
        secret: 'test-api-secret'
      });
      
      // Execute request
      const resultPromise = newDeltaApi.getMarkets();
      
      // Advance timers to trigger retry
      jest.advanceTimersByTime(1000);
      
      // Wait for the promise to resolve
      const result = await resultPromise;
      
      // Verify successful result after retry
      expect(result.success).toBe(true);
      
      // Restore original axios
      jest.unmock('axios');
    });
  });
}); 