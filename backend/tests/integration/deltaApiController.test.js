/**
 * Integration tests for Delta Exchange API Controller
 */
const request = require('supertest');
const express = require('express');
const jwt = require('jsonwebtoken');
const { createDeltaApiMock } = require('../mock/deltaApiMock');

// Mock models
jest.mock('@prisma/client', () => {
  const mockPrismaClient = {
    apiKey: {
      findFirst: jest.fn().mockResolvedValue({
        id: 1,
        userId: 'test-user-id',
        encryptedData: 'encrypted-data',
        createdAt: new Date(),
        updatedAt: new Date()
      })
    },
    tradeLog: {
      create: jest.fn().mockResolvedValue({
        id: 1
      })
    },
    $disconnect: jest.fn()
  };
  
  return {
    PrismaClient: jest.fn(() => mockPrismaClient)
  };
});

// Mock encryption
jest.mock('../../src/utils/encryption', () => ({
  decrypt: jest.fn().mockReturnValue({
    apiKey: 'test-api-key',
    apiSecret: 'test-api-secret',
    testnet: true
  })
}));

// Mock auth middleware
jest.mock('../../src/middleware/auth', () => ({
  protect: (req, res, next) => {
    req.user = {
      id: 'test-user-id',
      email: 'test@example.com'
    };
    next();
  }
}));

// Create a test app
const app = express();
app.use(express.json());

// Setup the routes
const deltaApiRoutes = require('../../src/routes/deltaApiRoutes');
app.use('/api/delta', deltaApiRoutes);

describe('Delta API Controller', () => {
  let apiMock;
  
  beforeEach(() => {
    // Create a fresh API mock for each test
    apiMock = createDeltaApiMock(true);
  });
  
  describe('Market Data Endpoints', () => {
    test('GET /api/delta/products should return products', async () => {
      // Setup mock
      apiMock.mockGetProducts();
      
      // Execute
      const response = await request(app)
        .get('/api/delta/products')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result).toBeInstanceOf(Array);
      expect(response.body.data.result.length).toBe(2);
    });
    
    test('GET /api/delta/products/:id/orderbook should return orderbook', async () => {
      // Setup mock
      const productId = 1;
      apiMock.mockGetOrderBook(productId);
      
      // Execute
      const response = await request(app)
        .get(`/api/delta/products/${productId}/orderbook`)
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result.asks).toBeInstanceOf(Array);
      expect(response.body.data.result.bids).toBeInstanceOf(Array);
    });
    
    test('GET /api/delta/products/:id/trades should return recent trades', async () => {
      // Setup mock
      const productId = 1;
      apiMock.mockGetRecentTrades(productId);
      
      // Execute
      const response = await request(app)
        .get(`/api/delta/products/${productId}/trades`)
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result).toBeInstanceOf(Array);
    });
  });
  
  describe('Account Endpoints', () => {
    test('GET /api/delta/balance should return wallet balances', async () => {
      // Setup mock
      apiMock.mockGetWalletBalance();
      
      // Execute
      const response = await request(app)
        .get('/api/delta/balance')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result).toBeInstanceOf(Array);
      expect(response.body.data.result[0].asset).toBe('USDT');
    });
    
    test('GET /api/delta/positions should return positions', async () => {
      // Setup mock
      apiMock.mockGetPositions();
      
      // Execute
      const response = await request(app)
        .get('/api/delta/positions')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result).toBeInstanceOf(Array);
      expect(response.body.data.result[0].symbol).toBe('BTCUSDT');
    });
  });
  
  describe('Order & Trading Endpoints', () => {
    test('GET /api/delta/orders should return active orders', async () => {
      // Setup mock
      apiMock.mockGetOrders();
      
      // Execute
      const response = await request(app)
        .get('/api/delta/orders')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result).toBeInstanceOf(Array);
    });
    
    test('POST /api/delta/orders should create a new order', async () => {
      // Setup mock
      apiMock.mockCreateOrder();
      
      // Execute
      const response = await request(app)
        .post('/api/delta/orders')
        .send({
          product_id: 1,
          size: '0.1',
          side: 'buy',
          order_type: 'limit',
          limit_price: '46000.0'
        })
        .set('Accept', 'application/json')
        .set('Content-Type', 'application/json');
      
      // Verify
      expect(response.status).toBe(201);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result.symbol).toBe('BTCUSDT');
    });
    
    test('DELETE /api/delta/orders/:id should cancel an order', async () => {
      // Setup mock
      const orderId = '1001';
      apiMock.mockCancelOrder(orderId);
      
      // Execute
      const response = await request(app)
        .delete(`/api/delta/orders/${orderId}`)
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result.status).toBe('cancelled');
    });
    
    test('DELETE /api/delta/orders should cancel all orders', async () => {
      // Setup mock
      apiMock.mockCancelAllOrders();
      
      // Execute
      const response = await request(app)
        .delete('/api/delta/orders')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result.cancelled_ids).toBeInstanceOf(Array);
    });
    
    test('GET /api/delta/orders/history should return order history', async () => {
      // Setup mock
      apiMock.mockGetOrderHistory();
      
      // Execute
      const response = await request(app)
        .get('/api/delta/orders/history')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result).toBeInstanceOf(Array);
      expect(response.body.data.result.length).toBe(2);
    });
    
    test('GET /api/delta/fills should return trade history', async () => {
      // Setup mock
      apiMock.mockGetTradeHistory();
      
      // Execute
      const response = await request(app)
        .get('/api/delta/fills')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data.result).toBeInstanceOf(Array);
      expect(response.body.data.result[0].order_id).toBe(901);
    });
  });
  
  describe('Error Handling', () => {
    test('should return 400 on API validation errors', async () => {
      // Setup mock with a 400 error
      apiMock.mockApiError('/v2/orders', 400, 'Invalid order parameters');
      
      // Execute
      const response = await request(app)
        .post('/api/delta/orders')
        .send({
          // Missing required fields
        })
        .set('Accept', 'application/json')
        .set('Content-Type', 'application/json');
      
      // Verify
      expect(response.status).toBe(400);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeDefined();
    });
    
    test('should return 404 for non-existent resources', async () => {
      // Setup mock with a 404 error
      const nonExistentOrderId = '9999';
      apiMock.mockApiError(`/v2/orders/${nonExistentOrderId}`, 404, 'Order not found');
      
      // Execute
      const response = await request(app)
        .delete(`/api/delta/orders/${nonExistentOrderId}`)
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(404);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeDefined();
    });
    
    test('should handle 429 rate limit errors with retry', async () => {
      // Setup mocks - first with rate limit, then with success
      apiMock.mockRateLimitError('/v2/products');
      apiMock.mockGetProducts();
      
      // Execute
      const response = await request(app)
        .get('/api/delta/products')
        .set('Accept', 'application/json');
      
      // Verify successful response after retry
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
    });
    
    test('should return 401 when API keys are invalid', async () => {
      // Mock decryption to return invalid keys
      require('../../src/utils/encryption').decrypt.mockReturnValueOnce({
        apiKey: 'invalid-key',
        apiSecret: 'invalid-secret',
        testnet: true
      });
      
      // Setup mock with a 401 error
      apiMock.mockApiError('/v2/wallet/balances', 401, 'Invalid API key');
      
      // Execute
      const response = await request(app)
        .get('/api/delta/balance')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(401);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('authentication');
    });
  });

  describe('Edge Cases', () => {
    test('should handle server errors gracefully', async () => {
      // Setup mock with a 500 error
      apiMock.mockApiError('/v2/products', 500, 'Internal server error');
      
      // Execute
      const response = await request(app)
        .get('/api/delta/products')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(500);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeDefined();
    });

    test('should handle timeout errors', async () => {
      // Setup mock with a timeout error
      apiMock.mockApiError('/v2/positions', 504, 'Gateway timeout');
      
      // Execute
      const response = await request(app)
        .get('/api/delta/positions')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(504);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toBeDefined();
    });
    
    test('should handle malformed JSON responses', async () => {
      // Setup scope directly to return invalid JSON
      const scope = require('nock')('https://testnet-api.delta.exchange');
      scope.get('/v2/products').reply(200, 'Not a JSON response');
      
      // Execute
      const response = await request(app)
        .get('/api/delta/products')
        .set('Accept', 'application/json');
      
      // Verify
      expect(response.status).toBe(500);
      expect(response.body.success).toBe(false);
      expect(response.body.error).toContain('parsing');
    });
  });
}); 