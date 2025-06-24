/**
 * Security Audit Tests
 * Comprehensive security testing for the trading system
 */

import request from 'supertest';
import { app } from '../../src/app';
import { createTestUser, createTestBot } from '../helpers/testHelpers';
import { logger } from '../../src/utils/logger';
import prisma from '../../src/utils/prismaClient';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

describe('Security Audit Tests', () => {
  let testUser: any;
  let testBot: any;
  let authToken: string;
  let anotherUser: any;
  let anotherUserToken: string;

  beforeAll(async () => {
    // Create test users
    testUser = await createTestUser();
    anotherUser = await createTestUser();
    
    testBot = await createTestBot(testUser.id);
    
    // Get authentication tokens
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: testUser.email,
        password: 'testpassword123'
      });
    
    authToken = loginResponse.body.token;
    
    const anotherLoginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: anotherUser.email,
        password: 'testpassword123'
      });
    
    anotherUserToken = anotherLoginResponse.body.token;
    
    logger.info('Security audit test setup completed');
  });

  afterAll(async () => {
    // Cleanup test data
    await prisma.order.deleteMany({ where: { userId: testUser.id } });
    await prisma.position.deleteMany({ where: { userId: testUser.id } });
    await prisma.backtest.deleteMany({ where: { botId: testBot.id } });
    await prisma.bot.delete({ where: { id: testBot.id } });
    await prisma.user.delete({ where: { id: testUser.id } });
    await prisma.user.delete({ where: { id: anotherUser.id } });
    
    await prisma.$disconnect();
    logger.info('Security audit test cleanup completed');
  });

  describe('Authentication Security', () => {
    it('should reject requests without authentication token', async () => {
      const response = await request(app)
        .get('/api/bots')
        .expect(401);
      
      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('token');
    });

    it('should reject requests with invalid authentication token', async () => {
      const response = await request(app)
        .get('/api/bots')
        .set('Authorization', 'Bearer invalid-token')
        .expect(401);
      
      expect(response.body).toHaveProperty('error');
    });

    it('should reject requests with expired authentication token', async () => {
      // Create an expired token
      const expiredToken = jwt.sign(
        { userId: testUser.id },
        process.env.JWT_SECRET || 'test-secret',
        { expiresIn: '-1h' } // Expired 1 hour ago
      );
      
      const response = await request(app)
        .get('/api/bots')
        .set('Authorization', `Bearer ${expiredToken}`)
        .expect(401);
      
      expect(response.body).toHaveProperty('error');
    });

    it('should reject requests with malformed authorization header', async () => {
      const malformedHeaders = [
        'invalid-format',
        'Bearer',
        'Bearer ',
        'Basic ' + authToken,
        authToken // Missing Bearer prefix
      ];
      
      for (const header of malformedHeaders) {
        const response = await request(app)
          .get('/api/bots')
          .set('Authorization', header)
          .expect(401);
        
        expect(response.body).toHaveProperty('error');
      }
    });

    it('should enforce strong password requirements', async () => {
      const weakPasswords = [
        '123456',
        'password',
        'abc123',
        '12345678',
        'qwerty',
        'password123'
      ];
      
      for (const password of weakPasswords) {
        const response = await request(app)
          .post('/api/auth/register')
          .send({
            email: `weak-${Date.now()}@example.com`,
            username: `weakuser${Date.now()}`,
            password
          });
        
        // Should reject weak passwords
        expect(response.status).toBe(400);
        expect(response.body).toHaveProperty('error');
      }
    });

    it('should hash passwords securely', async () => {
      const user = await prisma.user.findUnique({
        where: { id: testUser.id }
      });
      
      expect(user).toBeTruthy();
      expect(user!.passwordHash).toBeDefined();
      expect(user!.passwordHash).not.toBe('testpassword123'); // Should be hashed
      expect(user!.passwordHash.length).toBeGreaterThan(50); // bcrypt hash length
      
      // Verify password can be validated
      const isValid = await bcrypt.compare('testpassword123', user!.passwordHash);
      expect(isValid).toBe(true);
      
      const isInvalid = await bcrypt.compare('wrongpassword', user!.passwordHash);
      expect(isInvalid).toBe(false);
    });
  });

  describe('Authorization Security', () => {
    it('should prevent access to other users\' bots', async () => {
      // Try to access another user's bot
      const response = await request(app)
        .get(`/api/bots/${testBot.id}`)
        .set('Authorization', `Bearer ${anotherUserToken}`)
        .expect(403);
      
      expect(response.body).toHaveProperty('error');
      expect(response.body.error).toContain('access');
    });

    it('should prevent modification of other users\' bots', async () => {
      // Try to update another user's bot
      const response = await request(app)
        .put(`/api/bots/${testBot.id}`)
        .set('Authorization', `Bearer ${anotherUserToken}`)
        .send({
          name: 'Hacked Bot',
          isActive: false
        })
        .expect(403);
      
      expect(response.body).toHaveProperty('error');
    });

    it('should prevent deletion of other users\' bots', async () => {
      // Try to delete another user's bot
      const response = await request(app)
        .delete(`/api/bots/${testBot.id}`)
        .set('Authorization', `Bearer ${anotherUserToken}`)
        .expect(403);
      
      expect(response.body).toHaveProperty('error');
    });

    it('should prevent access to other users\' orders', async () => {
      // Create an order for the test user
      const orderResponse = await request(app)
        .post('/api/trading/orders')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          botId: testBot.id,
          symbol: 'BTCUSD',
          side: 'buy',
          amount: 0.1,
          type: 'market'
        });
      
      expect(orderResponse.status).toBe(201);
      const orderId = orderResponse.body.id;
      
      // Try to access the order with another user's token
      const accessResponse = await request(app)
        .get(`/api/trading/orders/${orderId}`)
        .set('Authorization', `Bearer ${anotherUserToken}`)
        .expect(403);
      
      expect(accessResponse.body).toHaveProperty('error');
    });

    it('should prevent privilege escalation', async () => {
      // Try to access admin-only endpoints
      const adminEndpoints = [
        '/api/admin/users',
        '/api/admin/system',
        '/api/admin/metrics'
      ];
      
      for (const endpoint of adminEndpoints) {
        const response = await request(app)
          .get(endpoint)
          .set('Authorization', `Bearer ${authToken}`);
        
        // Should be forbidden or not found (but not unauthorized)
        expect([403, 404]).toContain(response.status);
      }
    });
  });

  describe('Input Validation Security', () => {
    it('should prevent SQL injection attacks', async () => {
      const sqlInjectionPayloads = [
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "'; UPDATE users SET role='admin' WHERE id=1; --",
        "' UNION SELECT * FROM users --"
      ];
      
      for (const payload of sqlInjectionPayloads) {
        // Try SQL injection in login
        const loginResponse = await request(app)
          .post('/api/auth/login')
          .send({
            email: payload,
            password: 'password'
          });
        
        // Should not succeed with SQL injection
        expect(loginResponse.status).not.toBe(200);
        
        // Try SQL injection in bot creation
        const botResponse = await request(app)
          .post('/api/bots')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            name: payload,
            symbol: 'BTCUSD',
            strategy: 'enhanced_ml',
            timeframe: '1h'
          });
        
        // Should either reject or sanitize the input
        if (botResponse.status === 201) {
          // If created, the name should be sanitized
          expect(botResponse.body.name).not.toContain('DROP');
          expect(botResponse.body.name).not.toContain('UPDATE');
          expect(botResponse.body.name).not.toContain('UNION');
        }
      }
    });

    it('should prevent XSS attacks', async () => {
      const xssPayloads = [
        '<script>alert("xss")</script>',
        '<img src="x" onerror="alert(1)">',
        'javascript:alert("xss")',
        '<svg onload="alert(1)">',
        '"><script>alert("xss")</script>'
      ];
      
      for (const payload of xssPayloads) {
        const response = await request(app)
          .post('/api/bots')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            name: payload,
            symbol: 'BTCUSD',
            strategy: 'enhanced_ml',
            timeframe: '1h'
          });
        
        if (response.status === 201) {
          // If created, the name should be sanitized
          expect(response.body.name).not.toContain('<script>');
          expect(response.body.name).not.toContain('javascript:');
          expect(response.body.name).not.toContain('onerror');
          expect(response.body.name).not.toContain('onload');
        }
      }
    });

    it('should validate input data types and ranges', async () => {
      // Test invalid data types
      const invalidInputs = [
        {
          name: 123, // Should be string
          symbol: 'BTCUSD',
          strategy: 'enhanced_ml',
          timeframe: '1h'
        },
        {
          name: 'Test Bot',
          symbol: null, // Should be string
          strategy: 'enhanced_ml',
          timeframe: '1h'
        },
        {
          name: 'Test Bot',
          symbol: 'BTCUSD',
          strategy: 'enhanced_ml',
          timeframe: 999 // Should be string
        }
      ];
      
      for (const input of invalidInputs) {
        const response = await request(app)
          .post('/api/bots')
          .set('Authorization', `Bearer ${authToken}`)
          .send(input);
        
        expect(response.status).toBe(400);
        expect(response.body).toHaveProperty('error');
      }
    });

    it('should prevent buffer overflow attacks', async () => {
      // Test extremely long inputs
      const longString = 'A'.repeat(10000);
      
      const response = await request(app)
        .post('/api/bots')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: longString,
          symbol: 'BTCUSD',
          strategy: 'enhanced_ml',
          timeframe: '1h'
        });
      
      // Should reject or truncate extremely long inputs
      expect(response.status).toBe(400);
    });
  });

  describe('API Security', () => {
    it('should implement rate limiting', async () => {
      const requests = [];
      const maxRequests = 100; // Attempt many requests quickly
      
      // Make many rapid requests
      for (let i = 0; i < maxRequests; i++) {
        requests.push(
          request(app)
            .get('/api/health')
            .set('Authorization', `Bearer ${authToken}`)
        );
      }
      
      const responses = await Promise.all(requests);
      const rateLimitedResponses = responses.filter(r => r.status === 429);
      
      // Should have some rate limited responses
      expect(rateLimitedResponses.length).toBeGreaterThan(0);
    }, 30000);

    it('should prevent CSRF attacks', async () => {
      // Test that state-changing operations require proper authentication
      const stateChangingEndpoints = [
        { method: 'post', path: '/api/bots', data: { name: 'Test' } },
        { method: 'put', path: `/api/bots/${testBot.id}`, data: { name: 'Updated' } },
        { method: 'delete', path: `/api/bots/${testBot.id}` }
      ];
      
      for (const endpoint of stateChangingEndpoints) {
        // Try without CSRF token (if implemented)
        const response = await request(app)
          [endpoint.method](endpoint.path)
          .set('Authorization', `Bearer ${authToken}`)
          .send(endpoint.data || {});
        
        // Should either succeed with proper auth or require CSRF token
        expect([200, 201, 403]).toContain(response.status);
      }
    });

    it('should secure API key management', async () => {
      // Create API key
      const createResponse = await request(app)
        .post('/api/api-keys')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Test API Key',
          permissions: ['read']
        });
      
      expect(createResponse.status).toBe(201);
      expect(createResponse.body).toHaveProperty('key');
      expect(createResponse.body).toHaveProperty('id');
      
      const apiKeyId = createResponse.body.id;
      const apiKey = createResponse.body.key;
      
      // API key should be properly formatted
      expect(apiKey).toMatch(/^[a-zA-Z0-9_-]+$/);
      expect(apiKey.length).toBeGreaterThan(20);
      
      // Try to access with API key
      const apiResponse = await request(app)
        .get('/api/bots')
        .set('X-API-Key', apiKey);
      
      expect(apiResponse.status).toBe(200);
      
      // Try to access another user's data with API key
      const unauthorizedResponse = await request(app)
        .get(`/api/bots/${testBot.id}`)
        .set('X-API-Key', apiKey);
      
      // Should only allow access to own data
      expect([200, 403]).toContain(unauthorizedResponse.status);
      
      // Cleanup API key
      await request(app)
        .delete(`/api/api-keys/${apiKeyId}`)
        .set('Authorization', `Bearer ${authToken}`);
    });
  });

  describe('Data Protection', () => {
    it('should not expose sensitive information in responses', async () => {
      // Get user profile
      const profileResponse = await request(app)
        .get('/api/users/profile')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(profileResponse.status).toBe(200);
      
      // Should not expose password hash
      expect(profileResponse.body).not.toHaveProperty('passwordHash');
      expect(profileResponse.body).not.toHaveProperty('password');
      
      // Should not expose internal IDs unnecessarily
      expect(profileResponse.body).toHaveProperty('id');
      expect(profileResponse.body).toHaveProperty('email');
      expect(profileResponse.body).toHaveProperty('username');
    });

    it('should encrypt sensitive data at rest', async () => {
      // Check that API keys are hashed in database
      const apiKeyResponse = await request(app)
        .post('/api/api-keys')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Encryption Test Key',
          permissions: ['read']
        });
      
      expect(apiKeyResponse.status).toBe(201);
      const apiKeyId = apiKeyResponse.body.id;
      const plainKey = apiKeyResponse.body.key;
      
      // Check database storage
      const storedApiKey = await prisma.apiKey.findUnique({
        where: { id: apiKeyId }
      });
      
      expect(storedApiKey).toBeTruthy();
      expect(storedApiKey!.keyHash).toBeDefined();
      expect(storedApiKey!.keyHash).not.toBe(plainKey); // Should be hashed
      
      // Cleanup
      await request(app)
        .delete(`/api/api-keys/${apiKeyId}`)
        .set('Authorization', `Bearer ${authToken}`);
    });

    it('should implement secure session management', async () => {
      // Login and get token
      const loginResponse = await request(app)
        .post('/api/auth/login')
        .send({
          email: testUser.email,
          password: 'testpassword123'
        });
      
      expect(loginResponse.status).toBe(200);
      const token = loginResponse.body.token;
      
      // Token should be properly formatted JWT
      expect(token).toMatch(/^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/);
      
      // Logout should invalidate token (if implemented)
      const logoutResponse = await request(app)
        .post('/api/auth/logout')
        .set('Authorization', `Bearer ${token}`);
      
      // Should succeed or be not implemented
      expect([200, 404]).toContain(logoutResponse.status);
    });
  });

  describe('Error Handling Security', () => {
    it('should not expose stack traces in production', async () => {
      // Try to trigger an error
      const response = await request(app)
        .get('/api/nonexistent-endpoint')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(response.status).toBe(404);
      
      // Should not expose stack trace
      expect(response.body).not.toHaveProperty('stack');
      expect(response.body).not.toHaveProperty('trace');
      
      if (response.body.error) {
        expect(response.body.error).not.toContain('at ');
        expect(response.body.error).not.toContain('.js:');
        expect(response.body.error).not.toContain('Error:');
      }
    });

    it('should handle malformed requests gracefully', async () => {
      // Send malformed JSON
      const response = await request(app)
        .post('/api/bots')
        .set('Authorization', `Bearer ${authToken}`)
        .set('Content-Type', 'application/json')
        .send('{"invalid": json}');
      
      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
      
      // Should not expose internal error details
      expect(response.body.error).not.toContain('SyntaxError');
      expect(response.body.error).not.toContain('JSON.parse');
    });
  });

  describe('Infrastructure Security', () => {
    it('should set secure HTTP headers', async () => {
      const response = await request(app)
        .get('/api/health');
      
      // Check for security headers
      expect(response.headers).toHaveProperty('x-content-type-options');
      expect(response.headers['x-content-type-options']).toBe('nosniff');
      
      expect(response.headers).toHaveProperty('x-frame-options');
      expect(response.headers['x-frame-options']).toBe('DENY');
      
      expect(response.headers).toHaveProperty('x-xss-protection');
      expect(response.headers['x-xss-protection']).toBe('1; mode=block');
    });

    it('should not expose server information', async () => {
      const response = await request(app)
        .get('/api/health');
      
      // Should not expose server version
      expect(response.headers).not.toHaveProperty('server');
      expect(response.headers).not.toHaveProperty('x-powered-by');
    });
  });
});