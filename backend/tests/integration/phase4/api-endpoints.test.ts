/**
 * Phase 4: Production Readiness - API Endpoints Integration Tests
 * Target: Complete testing of API endpoints and database operations
 */

import { describe, beforeAll, afterAll, beforeEach, afterEach, it, expect, jest } from '@jest/globals';
import request from 'supertest';
import { Server } from 'http';
import app from '../../../src/app';
import prisma from '../../../src/config/prisma-readonly';

describe('Phase 4: API Endpoints Integration Tests', () => {
  let server: Server;
  let authToken: string;
  let testUserId: string;

  beforeAll(async () => {
    // Start the server
    server = app.listen(0); // Use random available port

    // Create test user and get auth token
    const testUser = await prisma.user.create({
      data: {
        email: 'test@trading.com',
        password: 'hashedpassword123',
        name: 'Test Trader',
        role: 'USER',
        isEmailVerified: true,
      }
    });
    testUserId = testUser.id;

    // Get auth token (simulate login)
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'test@trading.com',
        password: 'password123'
      });

    authToken = loginResponse.body.token;
  });

  afterAll(async () => {
    // Cleanup test data
    await prisma.order.deleteMany({ where: { userId: testUserId } });
    await prisma.position.deleteMany({ where: { userId: testUserId } });
    await prisma.tradingSignal.deleteMany({});
    await prisma.user.delete({ where: { id: testUserId } });
    
    // Close connections
    await prisma.$disconnect();
    server.close();
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Authentication Endpoints', () => {
    it('POST /api/auth/register should create new user with proper validation', async () => {
      const userData = {
        email: 'newuser@trading.com',
        password: 'SecurePass123!',
        name: 'New Trader',
      };

      const response = await request(app)
        .post('/api/auth/register')
        .send(userData)
        .expect(201);

      expect(response.body).toHaveProperty('user');
      expect(response.body).toHaveProperty('token');
      expect(response.body.user.email).toBe(userData.email);
      expect(response.body.user.password).toBeUndefined(); // Password should not be returned

      // Cleanup
      await prisma.user.delete({ where: { email: userData.email } });
    });

    it('POST /api/auth/login should authenticate valid users', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@trading.com',
          password: 'password123'
        })
        .expect(200);

      expect(response.body).toHaveProperty('token');
      expect(response.body).toHaveProperty('user');
      expect(response.body.user.email).toBe('test@trading.com');
    });

    it('POST /api/auth/login should reject invalid credentials', async () => {
      const response = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'test@trading.com',
          password: 'wrongpassword'
        })
        .expect(401);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Invalid credentials');
    });

    it('should implement rate limiting on login attempts', async () => {
      // Make multiple failed login attempts
      const promises = Array(6).fill(null).map(() =>
        request(app)
          .post('/api/auth/login')
          .send({
            email: 'test@trading.com',
            password: 'wrongpassword'
          })
      );

      const responses = await Promise.all(promises);
      
      // Last request should be rate limited
      expect(responses[5].status).toBe(429);
      expect(responses[5].body.error).toContain('Too many requests');
    });
  });

  describe('Trading Signal Endpoints', () => {
    it('GET /api/signals should return paginated trading signals', async () => {
      // Create test signals
      const signals = await Promise.all([
        prisma.tradingSignal.create({
          data: {
            symbol: 'BTCUSDT',
            direction: 'buy',
            confidence: 0.85,
            currentPrice: 50000,
            targetPrice: 51000,
            stopLoss: 49500,
          }
        }),
        prisma.tradingSignal.create({
          data: {
            symbol: 'ETHUSDT',
            direction: 'sell',
            confidence: 0.78,
            currentPrice: 3000,
            targetPrice: 2950,
            stopLoss: 3050,
          }
        }),
      ]);

      const response = await request(app)
        .get('/api/signals?page=1&limit=10')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('signals');
      expect(response.body).toHaveProperty('pagination');
      expect(response.body.signals).toHaveLength(2);
      expect(response.body.pagination.total).toBe(2);

      // Cleanup
      await prisma.tradingSignal.deleteMany({
        where: { id: { in: signals.map(s => s.id) } }
      });
    });

    it('POST /api/signals should create new trading signals with validation', async () => {
      const signalData = {
        symbol: 'BTCUSDT',
        direction: 'buy',
        confidence: 0.92,
        currentPrice: 50000,
        targetPrice: 52000,
        stopLoss: 48500,
      };

      const response = await request(app)
        .post('/api/signals')
        .set('Authorization', `Bearer ${authToken}`)
        .send(signalData)
        .expect(201);

      expect(response.body).toHaveProperty('signal');
      expect(response.body.signal.symbol).toBe(signalData.symbol);
      expect(response.body.signal.confidence).toBe(signalData.confidence);

      // Verify in database
      const savedSignal = await prisma.tradingSignal.findUnique({
        where: { id: response.body.signal.id }
      });
      expect(savedSignal).toBeTruthy();

      // Cleanup
      await prisma.tradingSignal.delete({ where: { id: response.body.signal.id } });
    });

    it('should reject signals with invalid data', async () => {
      const invalidSignalData = {
        symbol: '', // Invalid: empty symbol
        direction: 'invalid', // Invalid: not buy/sell
        confidence: 1.5, // Invalid: > 1
        currentPrice: -1000, // Invalid: negative price
      };

      const response = await request(app)
        .post('/api/signals')
        .set('Authorization', `Bearer ${authToken}`)
        .send(invalidSignalData)
        .expect(400);

      expect(response.body).toHaveProperty('errors');
      expect(response.body.errors).toBeInstanceOf(Array);
      expect(response.body.errors.length).toBeGreaterThan(0);
    });
  });

  describe('Order Management Endpoints', () => {
    it('POST /api/orders should create new orders with risk validation', async () => {
      const orderData = {
        symbol: 'BTCUSDT',
        type: 'market',
        side: 'buy',
        quantity: 0.01,
      };

      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .send(orderData)
        .expect(201);

      expect(response.body).toHaveProperty('order');
      expect(response.body.order.symbol).toBe(orderData.symbol);
      expect(response.body.order.status).toBe('pending');

      // Verify in database
      const savedOrder = await prisma.order.findUnique({
        where: { id: response.body.order.id }
      });
      expect(savedOrder).toBeTruthy();
      expect(savedOrder?.userId).toBe(testUserId);

      // Cleanup
      await prisma.order.delete({ where: { id: response.body.order.id } });
    });

    it('GET /api/orders should return user orders with filtering', async () => {
      // Create test orders
      const orders = await Promise.all([
        prisma.order.create({
          data: {
            userId: testUserId,
            symbol: 'BTCUSDT',
            type: 'market',
            side: 'buy',
            quantity: 0.01,
            status: 'filled',
            price: 50000,
          }
        }),
        prisma.order.create({
          data: {
            userId: testUserId,
            symbol: 'ETHUSDT',
            type: 'limit',
            side: 'sell',
            quantity: 0.1,
            status: 'pending',
            price: 3000,
          }
        }),
      ]);

      const response = await request(app)
        .get('/api/orders?status=pending&symbol=ETHUSDT')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('orders');
      expect(response.body.orders).toHaveLength(1);
      expect(response.body.orders[0].symbol).toBe('ETHUSDT');
      expect(response.body.orders[0].status).toBe('pending');

      // Cleanup
      await prisma.order.deleteMany({
        where: { id: { in: orders.map(o => o.id) } }
      });
    });

    it('PUT /api/orders/:id/cancel should cancel pending orders', async () => {
      // Create test order
      const order = await prisma.order.create({
        data: {
          userId: testUserId,
          symbol: 'BTCUSDT',
          type: 'limit',
          side: 'buy',
          quantity: 0.01,
          status: 'pending',
          price: 48000,
        }
      });

      const response = await request(app)
        .put(`/api/orders/${order.id}/cancel`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('order');
      expect(response.body.order.status).toBe('cancelled');

      // Verify in database
      const updatedOrder = await prisma.order.findUnique({
        where: { id: order.id }
      });
      expect(updatedOrder?.status).toBe('cancelled');

      // Cleanup
      await prisma.order.delete({ where: { id: order.id } });
    });

    it('should prevent canceling filled orders', async () => {
      // Create filled order
      const order = await prisma.order.create({
        data: {
          userId: testUserId,
          symbol: 'BTCUSDT',
          type: 'market',
          side: 'buy',
          quantity: 0.01,
          status: 'filled',
          price: 50000,
        }
      });

      const response = await request(app)
        .put(`/api/orders/${order.id}/cancel`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Cannot cancel filled order');

      // Cleanup
      await prisma.order.delete({ where: { id: order.id } });
    });
  });

  describe('Portfolio & Performance Endpoints', () => {
    it('GET /api/portfolio should return current portfolio state', async () => {
      // Create test positions
      const positions = await Promise.all([
        prisma.position.create({
          data: {
            userId: testUserId,
            symbol: 'BTCUSDT',
            side: 'long',
            size: 0.5,
            entryPrice: 50000,
            currentPrice: 51000,
            unrealizedPnl: 500,
          }
        }),
        prisma.position.create({
          data: {
            userId: testUserId,
            symbol: 'ETHUSDT',
            side: 'short',
            size: 2.0,
            entryPrice: 3000,
            currentPrice: 2950,
            unrealizedPnl: 100,
          }
        }),
      ]);

      const response = await request(app)
        .get('/api/portfolio')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('positions');
      expect(response.body).toHaveProperty('totalValue');
      expect(response.body).toHaveProperty('unrealizedPnl');
      expect(response.body.positions).toHaveLength(2);
      expect(response.body.unrealizedPnl).toBe(600); // 500 + 100

      // Cleanup
      await prisma.position.deleteMany({
        where: { id: { in: positions.map(p => p.id) } }
      });
    });

    it('GET /api/performance/metrics should return trading performance analytics', async () => {
      // Create test trading history
      const orders = await Promise.all([
        prisma.order.create({
          data: {
            userId: testUserId,
            symbol: 'BTCUSDT',
            type: 'market',
            side: 'buy',
            quantity: 0.01,
            status: 'filled',
            price: 50000,
            pnl: 100,
          }
        }),
        prisma.order.create({
          data: {
            userId: testUserId,
            symbol: 'ETHUSDT',
            type: 'market',
            side: 'sell',
            quantity: 0.1,
            status: 'filled',
            price: 3000,
            pnl: -50,
          }
        }),
      ]);

      const response = await request(app)
        .get('/api/performance/metrics?period=30d')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('winRate');
      expect(response.body).toHaveProperty('totalPnl');
      expect(response.body).toHaveProperty('sharpeRatio');
      expect(response.body).toHaveProperty('maxDrawdown');
      expect(response.body.totalPnl).toBe(50); // 100 - 50

      // Cleanup
      await prisma.order.deleteMany({
        where: { id: { in: orders.map(o => o.id) } }
      });
    });
  });

  describe('Market Data Endpoints', () => {
    it('GET /api/market/data/:symbol should return real-time market data', async () => {
      const response = await request(app)
        .get('/api/market/data/BTCUSDT')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('symbol');
      expect(response.body).toHaveProperty('price');
      expect(response.body).toHaveProperty('volume');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body.symbol).toBe('BTCUSDT');
    });

    it('GET /api/market/orderbook/:symbol should return order book data', async () => {
      const response = await request(app)
        .get('/api/market/orderbook/BTCUSDT?depth=10')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toHaveProperty('bids');
      expect(response.body).toHaveProperty('asks');
      expect(response.body).toHaveProperty('timestamp');
      expect(response.body.bids).toBeInstanceOf(Array);
      expect(response.body.asks).toBeInstanceOf(Array);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle database connection failures gracefully', async () => {
      // Mock database error
      const originalFindMany = prisma.order.findMany;
      prisma.order.findMany = jest.fn().mockRejectedValue(new Error('Database connection failed'));

      const response = await request(app)
        .get('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(500);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Internal server error');

      // Restore original function
      prisma.order.findMany = originalFindMany;
    });

    it('should validate request body size limits', async () => {
      const largePayload = {
        symbol: 'BTCUSDT',
        notes: 'A'.repeat(10000), // Very large string
      };

      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .send(largePayload)
        .expect(413);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Payload too large');
    });

    it('should handle malformed JSON requests', async () => {
      const response = await request(app)
        .post('/api/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .set('Content-Type', 'application/json')
        .send('{ invalid json }')
        .expect(400);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Invalid JSON');
    });

    it('should enforce authentication on protected endpoints', async () => {
      const response = await request(app)
        .get('/api/orders')
        .expect(401);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Unauthorized');
    });

    it('should handle invalid JWT tokens', async () => {
      const response = await request(app)
        .get('/api/orders')
        .set('Authorization', 'Bearer invalid.jwt.token')
        .expect(401);

      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('Invalid token');
    });
  });

  describe('CORS and Security Headers', () => {
    it('should include proper CORS headers', async () => {
      const response = await request(app)
        .options('/api/orders')
        .set('Origin', 'http://localhost:3000')
        .expect(200);

      expect(response.headers['access-control-allow-origin']).toBeDefined();
      expect(response.headers['access-control-allow-methods']).toBeDefined();
      expect(response.headers['access-control-allow-headers']).toBeDefined();
    });

    it('should include security headers', async () => {
      const response = await request(app)
        .get('/api/health')
        .expect(200);

      expect(response.headers['x-content-type-options']).toBe('nosniff');
      expect(response.headers['x-frame-options']).toBe('DENY');
      expect(response.headers['x-xss-protection']).toBe('1; mode=block');
    });
  });

  describe('Performance and Response Times', () => {
    it('should respond to health check within 100ms', async () => {
      const startTime = Date.now();
      
      await request(app)
        .get('/api/health')
        .expect(200);

      const responseTime = Date.now() - startTime;
      expect(responseTime).toBeLessThan(100);
    });

    it('should handle concurrent requests efficiently', async () => {
      const promises = Array(10).fill(null).map(() =>
        request(app)
          .get('/api/market/data/BTCUSDT')
          .set('Authorization', `Bearer ${authToken}`)
      );

      const startTime = Date.now();
      const responses = await Promise.all(promises);
      const totalTime = Date.now() - startTime;

      // All requests should succeed
      responses.forEach(response => {
        expect(response.status).toBe(200);
      });

      // Should handle 10 concurrent requests in under 2 seconds
      expect(totalTime).toBeLessThan(2000);
    });
  });
});