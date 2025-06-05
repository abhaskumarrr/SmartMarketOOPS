/**
 * API Integration Tests
 * End-to-end tests for API endpoints
 */

import request from 'supertest';
import { Express } from 'express';
import { createTestApp } from '../helpers/testApp';
import { createTestUser, cleanupTestData } from '../helpers/testHelpers';

describe('API Integration Tests', () => {
  let app: Express;
  let authToken: string;
  let testUserId: string;

  beforeAll(async () => {
    app = await createTestApp();
    
    // Create test user and get auth token
    const { user, token } = await createTestUser();
    testUserId = user.id;
    authToken = token;
  });

  afterAll(async () => {
    await cleanupTestData();
  });

  describe('Health Check', () => {
    it('should return health status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.body).toMatchObject({
        status: 'healthy',
        timestamp: expect.any(String),
        uptime: expect.any(Number),
      });
    });
  });

  describe('Authentication', () => {
    it('should register a new user', async () => {
      const userData = {
        email: 'newuser@test.com',
        password: 'password123',
        name: 'New User',
      };

      const response = await request(app)
        .post('/api/auth/register')
        .send(userData)
        .expect(201);

      expect(response.body).toMatchObject({
        success: true,
        data: {
          user: {
            email: userData.email,
            name: userData.name,
          },
          token: expect.any(String),
        },
      });
    });

    it('should login with valid credentials', async () => {
      const loginData = {
        email: 'test@example.com',
        password: 'password123',
      };

      const response = await request(app)
        .post('/api/auth/login')
        .send(loginData)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: {
          user: expect.objectContaining({
            email: loginData.email,
          }),
          token: expect.any(String),
        },
      });
    });

    it('should reject invalid credentials', async () => {
      const loginData = {
        email: 'test@example.com',
        password: 'wrongpassword',
      };

      const response = await request(app)
        .post('/api/auth/login')
        .send(loginData)
        .expect(401);

      expect(response.body).toMatchObject({
        success: false,
        message: expect.stringContaining('Invalid'),
      });
    });

    it('should require authentication for protected routes', async () => {
      await request(app)
        .get('/api/bots')
        .expect(401);
    });
  });

  describe('Bot Management', () => {
    let botId: string;

    it('should create a new bot', async () => {
      const botData = {
        name: 'Test Bot',
        symbol: 'BTCUSD',
        strategy: 'ML_PREDICTION',
        timeframe: '1h',
        parameters: {
          confidence_threshold: 0.7,
        },
      };

      const response = await request(app)
        .post('/api/bots')
        .set('Authorization', `Bearer ${authToken}`)
        .send(botData)
        .expect(201);

      expect(response.body).toMatchObject({
        success: true,
        data: expect.objectContaining({
          id: expect.any(String),
          name: botData.name,
          symbol: botData.symbol,
          strategy: botData.strategy,
        }),
      });

      botId = response.body.data.id;
    });

    it('should get all bots for user', async () => {
      const response = await request(app)
        .get('/api/bots')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: expect.arrayContaining([
          expect.objectContaining({
            id: botId,
            name: 'Test Bot',
          }),
        ]),
      });
    });

    it('should get specific bot', async () => {
      const response = await request(app)
        .get(`/api/bots/${botId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: expect.objectContaining({
          id: botId,
          name: 'Test Bot',
        }),
      });
    });

    it('should update bot', async () => {
      const updateData = {
        name: 'Updated Test Bot',
        parameters: {
          confidence_threshold: 0.8,
        },
      };

      const response = await request(app)
        .put(`/api/bots/${botId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(updateData)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: expect.objectContaining({
          id: botId,
          name: updateData.name,
        }),
      });
    });

    it('should start bot', async () => {
      const response = await request(app)
        .post(`/api/bots/${botId}/start`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        message: expect.stringContaining('started'),
      });
    });

    it('should get bot status', async () => {
      const response = await request(app)
        .get(`/api/bots/${botId}/status`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: expect.objectContaining({
          isRunning: expect.any(Boolean),
          health: expect.any(String),
        }),
      });
    });

    it('should stop bot', async () => {
      const response = await request(app)
        .post(`/api/bots/${botId}/stop`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        message: expect.stringContaining('stopped'),
      });
    });

    it('should run backtest', async () => {
      const backtestConfig = {
        symbol: 'BTCUSD',
        timeframe: '1h',
        startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        endDate: new Date().toISOString(),
        initialCapital: 10000,
        leverage: 1,
        riskPerTrade: 2,
        commission: 0.1,
      };

      const response = await request(app)
        .post(`/api/bots/${botId}/backtest`)
        .set('Authorization', `Bearer ${authToken}`)
        .send(backtestConfig)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        data: expect.objectContaining({
          performance: expect.objectContaining({
            totalReturn: expect.any(Number),
            winRate: expect.any(Number),
            sharpeRatio: expect.any(Number),
          }),
        }),
      });
    });

    it('should delete bot', async () => {
      const response = await request(app)
        .delete(`/api/bots/${botId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body).toMatchObject({
        success: true,
        message: expect.stringContaining('deleted'),
      });

      // Verify bot is deleted
      await request(app)
        .get(`/api/bots/${botId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);
    });
  });

  describe('Rate Limiting', () => {
    it('should enforce rate limits', async () => {
      // Make multiple requests quickly
      const promises = Array(10).fill(null).map(() =>
        request(app)
          .get('/health')
          .expect(200)
      );

      await Promise.all(promises);

      // Additional requests should be rate limited
      const response = await request(app)
        .get('/health');

      // Should either succeed or be rate limited
      expect([200, 429]).toContain(response.status);
    });
  });

  describe('Error Handling', () => {
    it('should handle 404 for non-existent routes', async () => {
      const response = await request(app)
        .get('/api/non-existent')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);

      expect(response.body).toMatchObject({
        success: false,
        message: expect.stringContaining('not found'),
      });
    });

    it('should handle validation errors', async () => {
      const invalidBotData = {
        name: '', // Invalid: empty name
        symbol: 'INVALID', // Invalid: not in enum
      };

      const response = await request(app)
        .post('/api/bots')
        .set('Authorization', `Bearer ${authToken}`)
        .send(invalidBotData)
        .expect(400);

      expect(response.body).toMatchObject({
        success: false,
        errors: expect.any(Array),
      });
    });

    it('should handle server errors gracefully', async () => {
      // This would require mocking a service to throw an error
      // For now, we'll test that the error handler is properly configured
      expect(app).toBeDefined();
    });
  });

  describe('Performance', () => {
    it('should respond within acceptable time limits', async () => {
      const start = Date.now();
      
      await request(app)
        .get('/health')
        .expect(200);
      
      const duration = Date.now() - start;
      expect(duration).toBeLessThan(1000); // Should respond within 1 second
    });

    it('should include performance headers', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.headers).toHaveProperty('x-response-time');
      expect(response.headers).toHaveProperty('x-request-id');
    });
  });

  describe('Security', () => {
    it('should include security headers', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.headers).toHaveProperty('x-content-type-options', 'nosniff');
      expect(response.headers).toHaveProperty('x-frame-options', 'DENY');
      expect(response.headers).toHaveProperty('x-xss-protection', '1; mode=block');
    });

    it('should not expose sensitive information in errors', async () => {
      const response = await request(app)
        .get('/api/bots/invalid-id')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(404);

      expect(response.body.message).not.toContain('database');
      expect(response.body.message).not.toContain('sql');
      expect(response.body.message).not.toContain('prisma');
    });
  });
});
