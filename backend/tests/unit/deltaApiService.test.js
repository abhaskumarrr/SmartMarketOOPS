/**
 * Unit tests for Delta Exchange API Service
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

describe('DeltaExchangeAPI', () => {
  let deltaApi;
  let apiMock;
  
  beforeEach(async () => {
    // Create a new instance for each test
    deltaApi = new DeltaExchangeAPI({ testnet: true });
    await deltaApi.initialize({
      key: 'test-api-key',
      secret: 'test-api-secret'
    });
    
    // Create a fresh API mock for each test
    apiMock = createDeltaApiMock(true);
  });
  
  afterEach(() => {
    // Ensure all nock mocks are consumed
    jest.clearAllMocks();
  });
  
  describe('Market Data Methods', () => {
    test('getServerTime should return server time', async () => {
      // Setup mock
      apiMock.mockServerTime();
      
      // Execute
      const result = await deltaApi.getServerTime();
      
      // Verify
      expect(result).toBeDefined();
      expect(result.server_time).toBeDefined();
      expect(result.server_time_iso).toBeDefined();
    });
    
    test('getMarkets should return available markets', async () => {
      // Setup mock
      apiMock.mockGetProducts();
      
      // Execute
      const result = await deltaApi.getMarkets();
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result).toBeInstanceOf(Array);
      expect(result.result.length).toBe(2);
      expect(result.result[0].symbol).toBe('BTCUSDT');
    });
    
    test('getOrderbook should return orderbook data', async () => {
      // Setup mock
      const productId = 1;
      apiMock.mockGetOrderBook(productId);
      
      // Execute
      const result = await deltaApi.getOrderbook('BTCUSDT');
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result.asks).toBeInstanceOf(Array);
      expect(result.result.bids).toBeInstanceOf(Array);
    });
  });
  
  describe('Account & Trading Methods', () => {
    test('getWalletBalances should return user balances', async () => {
      // Setup mock
      apiMock.mockGetWalletBalance();
      
      // Execute
      const result = await deltaApi.getWalletBalances();
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result).toBeInstanceOf(Array);
      expect(result.result.length).toBe(2);
      expect(result.result[0].asset).toBe('USDT');
    });
    
    test('getPositions should return user positions', async () => {
      // Setup mock
      apiMock.mockGetPositions();
      
      // Execute
      const result = await deltaApi.getPositions();
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result).toBeInstanceOf(Array);
      expect(result.result[0].symbol).toBe('BTCUSDT');
      expect(result.result[0].side).toBe('long');
    });
    
    test('getActiveOrders should return open orders', async () => {
      // Setup mock
      apiMock.mockGetOrders();
      
      // Execute
      const result = await deltaApi.getActiveOrders();
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result).toBeInstanceOf(Array);
      expect(result.result[0].status).toBe('open');
    });
    
    test('placeOrder should create a new order', async () => {
      // Setup mock
      apiMock.mockCreateOrder();
      
      // Execute
      const orderParams = {
        symbol: 'BTCUSDT',
        side: 'buy',
        size: 0.1,
        type: 'limit',
        price: 46000
      };
      const result = await deltaApi.placeOrder(orderParams);
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result.symbol).toBe('BTCUSDT');
      expect(result.result.side).toBe('buy');
    });
    
    test('cancelOrder should cancel an existing order', async () => {
      // Setup mock
      const orderId = '1001';
      apiMock.mockCancelOrder(orderId);
      
      // Execute
      const result = await deltaApi.cancelOrder(orderId);
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result.id).toBe(parseInt(orderId));
      expect(result.result.status).toBe('cancelled');
    });
    
    test('cancelAllOrders should cancel all open orders', async () => {
      // Setup mock
      apiMock.mockCancelAllOrders();
      
      // Execute
      const result = await deltaApi.cancelAllOrders();
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result.cancelled_ids).toBeInstanceOf(Array);
      expect(result.result.cancelled_ids.length).toBe(2);
    });
    
    test('getOrderHistory should return order history', async () => {
      // Setup mock
      apiMock.mockGetOrderHistory();
      
      // Execute
      const result = await deltaApi.getOrderHistory();
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result).toBeInstanceOf(Array);
      expect(result.result.length).toBe(2);
    });
    
    test('getTradeHistory should return trade history', async () => {
      // Setup mock
      apiMock.mockGetTradeHistory();
      
      // Execute
      const result = await deltaApi.getTradeHistory();
      
      // Verify
      expect(result.success).toBe(true);
      expect(result.result).toBeInstanceOf(Array);
      expect(result.result[0].order_id).toBe(901);
    });
  });
  
  describe('Error Handling', () => {
    test('should handle API errors properly', async () => {
      // Setup mock
      apiMock.mockApiError('/v2/products', 400, 'Invalid parameters');
      
      // Execute & Verify
      await expect(deltaApi.getMarkets()).rejects.toThrow('Delta Exchange API Error');
    });
    
    test('should handle rate limit errors with retry', async () => {
      // Mock the rate limit settings to speed up test
      deltaApi.rateLimit = {
        maxRetries: 2,
        initialDelay: 10,
        maxDelay: 50,
        factor: 2
      };
      
      // Setup mocks - first with rate limit, then with success
      apiMock.mockRateLimitError('/v2/products');
      apiMock.mockGetProducts();
      
      // Execute
      const result = await deltaApi.getMarkets();
      
      // Verify
      expect(result.success).toBe(true);
    });
    
    test('should throw error if retry limit exceeded', async () => {
      // Mock the rate limit settings to speed up test
      deltaApi.rateLimit = {
        maxRetries: 1,
        initialDelay: 10,
        maxDelay: 50,
        factor: 2
      };
      
      // Setup mocks - both with rate limit errors
      apiMock.mockRateLimitError('/v2/products');
      apiMock.mockRateLimitError('/v2/products');
      
      // Execute & Verify
      await expect(deltaApi.getMarkets()).rejects.toThrow('Delta Exchange API Error');
    });
  });
}); 