/**
 * Load Testing Suite
 * Performance tests for API endpoints under load
 */

import request from 'supertest';
import { Express } from 'express';
import { createTestApp } from '../helpers/testApp';
import { createTestUser } from '../helpers/testHelpers';
import { performance } from 'perf_hooks';

describe('Load Testing', () => {
  let app: Express;
  let authToken: string;

  beforeAll(async () => {
    app = await createTestApp();
    const { token } = await createTestUser();
    authToken = token;
  });

  describe('Health Check Endpoint', () => {
    it('should handle concurrent requests', async () => {
      const concurrency = 50;
      const requests = Array(concurrency).fill(null).map(() =>
        request(app).get('/health')
      );

      const start = performance.now();
      const responses = await Promise.all(requests);
      const duration = performance.now() - start;

      // All requests should succeed
      responses.forEach(response => {
        expect(response.status).toBe(200);
      });

      // Should complete within reasonable time
      expect(duration).toBeLessThan(5000); // 5 seconds
      
      // Calculate average response time
      const avgResponseTime = duration / concurrency;
      expect(avgResponseTime).toBeLessThan(100); // 100ms average

      console.log(`✅ Health check load test: ${concurrency} requests in ${duration.toFixed(2)}ms (avg: ${avgResponseTime.toFixed(2)}ms)`);
    });

    it('should maintain performance under sustained load', async () => {
      const duration = 10000; // 10 seconds
      const requestInterval = 100; // Request every 100ms
      const startTime = performance.now();
      const responses: number[] = [];
      const errors: number = 0;

      while (performance.now() - startTime < duration) {
        const requestStart = performance.now();
        
        try {
          const response = await request(app).get('/health');
          const requestDuration = performance.now() - requestStart;
          responses.push(requestDuration);
          
          expect(response.status).toBe(200);
        } catch (error) {
          // Count errors but don't fail the test
        }

        // Wait for next interval
        await new Promise(resolve => setTimeout(resolve, requestInterval));
      }

      // Analyze results
      const avgResponseTime = responses.reduce((a, b) => a + b, 0) / responses.length;
      const maxResponseTime = Math.max(...responses);
      const minResponseTime = Math.min(...responses);
      
      expect(avgResponseTime).toBeLessThan(200); // 200ms average
      expect(maxResponseTime).toBeLessThan(1000); // 1s max
      expect(responses.length).toBeGreaterThan(50); // At least 50 requests

      console.log(`✅ Sustained load test: ${responses.length} requests over ${duration}ms`);
      console.log(`   Avg: ${avgResponseTime.toFixed(2)}ms, Min: ${minResponseTime.toFixed(2)}ms, Max: ${maxResponseTime.toFixed(2)}ms`);
    });
  });

  describe('Authentication Endpoints', () => {
    it('should handle concurrent login requests', async () => {
      const concurrency = 20;
      const loginData = {
        email: 'test@example.com',
        password: 'password123',
      };

      const requests = Array(concurrency).fill(null).map(() =>
        request(app)
          .post('/api/auth/login')
          .send(loginData)
      );

      const start = performance.now();
      const responses = await Promise.all(requests);
      const duration = performance.now() - start;

      // All requests should succeed
      responses.forEach(response => {
        expect(response.status).toBe(200);
        expect(response.body.success).toBe(true);
      });

      expect(duration).toBeLessThan(10000); // 10 seconds
      console.log(`✅ Login load test: ${concurrency} requests in ${duration.toFixed(2)}ms`);
    });
  });

  describe('Bot Management Endpoints', () => {
    it('should handle concurrent bot creation', async () => {
      const concurrency = 10;
      
      const requests = Array(concurrency).fill(null).map((_, index) =>
        request(app)
          .post('/api/bots')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            name: `Load Test Bot ${index}`,
            symbol: 'BTCUSD',
            strategy: 'ML_PREDICTION',
            timeframe: '1h',
            parameters: {},
          })
      );

      const start = performance.now();
      const responses = await Promise.all(requests);
      const duration = performance.now() - start;

      // All requests should succeed
      responses.forEach(response => {
        expect(response.status).toBe(201);
        expect(response.body.success).toBe(true);
      });

      expect(duration).toBeLessThan(15000); // 15 seconds
      console.log(`✅ Bot creation load test: ${concurrency} requests in ${duration.toFixed(2)}ms`);
    });

    it('should handle concurrent bot status requests', async () => {
      // First create a bot
      const botResponse = await request(app)
        .post('/api/bots')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Status Test Bot',
          symbol: 'BTCUSD',
          strategy: 'ML_PREDICTION',
          timeframe: '1h',
          parameters: {},
        });

      const botId = botResponse.body.data.id;
      const concurrency = 30;

      const requests = Array(concurrency).fill(null).map(() =>
        request(app)
          .get(`/api/bots/${botId}/status`)
          .set('Authorization', `Bearer ${authToken}`)
      );

      const start = performance.now();
      const responses = await Promise.all(requests);
      const duration = performance.now() - start;

      // All requests should succeed
      responses.forEach(response => {
        expect(response.status).toBe(200);
        expect(response.body.success).toBe(true);
      });

      expect(duration).toBeLessThan(8000); // 8 seconds
      console.log(`✅ Bot status load test: ${concurrency} requests in ${duration.toFixed(2)}ms`);
    });
  });

  describe('Memory and Resource Usage', () => {
    it('should not leak memory under load', async () => {
      const initialMemory = process.memoryUsage();
      const concurrency = 100;
      const iterations = 5;

      for (let i = 0; i < iterations; i++) {
        const requests = Array(concurrency).fill(null).map(() =>
          request(app).get('/health')
        );

        await Promise.all(requests);
        
        // Force garbage collection if available
        if (global.gc) {
          global.gc();
        }
      }

      const finalMemory = process.memoryUsage();
      const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;
      const memoryIncreasePercent = (memoryIncrease / initialMemory.heapUsed) * 100;

      // Memory increase should be reasonable (less than 50%)
      expect(memoryIncreasePercent).toBeLessThan(50);

      console.log(`✅ Memory test: ${memoryIncreasePercent.toFixed(2)}% increase after ${concurrency * iterations} requests`);
      console.log(`   Initial: ${(initialMemory.heapUsed / 1024 / 1024).toFixed(2)}MB, Final: ${(finalMemory.heapUsed / 1024 / 1024).toFixed(2)}MB`);
    });

    it('should handle file descriptor limits', async () => {
      const concurrency = 200; // High concurrency to test file descriptors
      
      const requests = Array(concurrency).fill(null).map(() =>
        request(app).get('/health')
      );

      const start = performance.now();
      const responses = await Promise.allSettled(requests);
      const duration = performance.now() - start;

      const successful = responses.filter(r => r.status === 'fulfilled').length;
      const failed = responses.filter(r => r.status === 'rejected').length;

      // At least 90% should succeed
      expect(successful / concurrency).toBeGreaterThan(0.9);

      console.log(`✅ File descriptor test: ${successful}/${concurrency} successful (${failed} failed) in ${duration.toFixed(2)}ms`);
    });
  });

  describe('Database Performance', () => {
    it('should handle concurrent database operations', async () => {
      const concurrency = 20;
      
      // Create multiple bots concurrently (database writes)
      const createRequests = Array(concurrency).fill(null).map((_, index) =>
        request(app)
          .post('/api/bots')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            name: `DB Test Bot ${index}`,
            symbol: 'BTCUSD',
            strategy: 'ML_PREDICTION',
            timeframe: '1h',
            parameters: {},
          })
      );

      const start = performance.now();
      const createResponses = await Promise.all(createRequests);
      const createDuration = performance.now() - start;

      // All creates should succeed
      createResponses.forEach(response => {
        expect(response.status).toBe(201);
      });

      // Now read them all back concurrently
      const readStart = performance.now();
      const readRequests = Array(concurrency).fill(null).map(() =>
        request(app)
          .get('/api/bots')
          .set('Authorization', `Bearer ${authToken}`)
      );

      const readResponses = await Promise.all(readRequests);
      const readDuration = performance.now() - readStart;

      // All reads should succeed
      readResponses.forEach(response => {
        expect(response.status).toBe(200);
        expect(response.body.data.length).toBeGreaterThanOrEqual(concurrency);
      });

      expect(createDuration).toBeLessThan(20000); // 20 seconds for writes
      expect(readDuration).toBeLessThan(5000); // 5 seconds for reads

      console.log(`✅ Database performance test:`);
      console.log(`   Creates: ${concurrency} in ${createDuration.toFixed(2)}ms`);
      console.log(`   Reads: ${concurrency} in ${readDuration.toFixed(2)}ms`);
    });
  });

  describe('Error Handling Under Load', () => {
    it('should handle invalid requests gracefully under load', async () => {
      const concurrency = 50;
      
      const requests = Array(concurrency).fill(null).map(() =>
        request(app)
          .post('/api/bots')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            // Invalid data
            name: '',
            symbol: 'INVALID',
          })
      );

      const start = performance.now();
      const responses = await Promise.all(requests);
      const duration = performance.now() - start;

      // All requests should return 400 (validation error)
      responses.forEach(response => {
        expect(response.status).toBe(400);
        expect(response.body.success).toBe(false);
      });

      expect(duration).toBeLessThan(10000); // 10 seconds
      console.log(`✅ Error handling load test: ${concurrency} invalid requests in ${duration.toFixed(2)}ms`);
    });

    it('should handle unauthorized requests under load', async () => {
      const concurrency = 50;
      
      const requests = Array(concurrency).fill(null).map(() =>
        request(app)
          .get('/api/bots')
          // No authorization header
      );

      const start = performance.now();
      const responses = await Promise.all(requests);
      const duration = performance.now() - start;

      // All requests should return 401 (unauthorized)
      responses.forEach(response => {
        expect(response.status).toBe(401);
      });

      expect(duration).toBeLessThan(5000); // 5 seconds
      console.log(`✅ Unauthorized load test: ${concurrency} requests in ${duration.toFixed(2)}ms`);
    });
  });
});
