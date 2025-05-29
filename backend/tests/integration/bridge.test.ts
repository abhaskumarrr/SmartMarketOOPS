/**
 * Integration Tests for Bridge API
 * Tests the integration between ML and Trading systems
 */

import request from 'supertest';
import { PrismaClient } from '@prisma/client';
import app from '../../src/server';
import { createTestUser, createTestToken, cleanupTestUser } from '../mock/authTestHelpers';

const prisma = new PrismaClient();

describe('Bridge API Integration Tests', () => {
  let token: string;
  let userId: string;

  beforeAll(async () => {
    // Create test user
    const testUser = await createTestUser();
    userId = testUser.id;
    token = await createTestToken(testUser);
  });

  afterAll(async () => {
    // Clean up test data
    await cleanupTestUser(userId);
    await prisma.$disconnect();
  });

  describe('ML Bridge Service Health Endpoints', () => {
    it('should return bridge health status', async () => {
      const response = await request(app)
        .get('/api/bridge/health')
        .set('Authorization', `Bearer ${token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('mlSystemStatus');
      expect(response.body).toHaveProperty('tradingSystemStatus');
      expect(response.body).toHaveProperty('latency');
    });

    it('should return ML system connection status', async () => {
      const response = await request(app)
        .get('/api/bridge/ml-health')
        .set('Authorization', `Bearer ${token}`);

      // Even if ML system is not available, the API should respond
      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('status');
      expect(response.body).toHaveProperty('latency');
    });
  });

  describe('ML Model Endpoints', () => {
    it('should handle get all models request even if no models exist', async () => {
      const response = await request(app)
        .get('/api/bridge/models')
        .set('Authorization', `Bearer ${token}`);

      // API should handle case when ML system is not available
      expect(response.status).toBe(200).or(expect.response.status).toBe(500);
      
      if (response.status === 200) {
        expect(Array.isArray(response.body)).toBe(true);
      }
    });

    it('should reject requests without authentication', async () => {
      const response = await request(app)
        .get('/api/bridge/models');

      expect(response.status).toBe(401);
    });
  });

  describe('Prediction Endpoints', () => {
    it('should validate prediction request inputs', async () => {
      const response = await request(app)
        .post('/api/bridge/predict')
        .set('Authorization', `Bearer ${token}`)
        .send({}); // Empty payload

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should validate prediction and signal request inputs', async () => {
      const response = await request(app)
        .post('/api/bridge/predict-and-signal')
        .set('Authorization', `Bearer ${token}`)
        .send({}); // Empty payload

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should validate batch prediction request inputs', async () => {
      const response = await request(app)
        .post('/api/bridge/predict-batch')
        .set('Authorization', `Bearer ${token}`)
        .send({}); // Empty payload

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });
  });

  describe('Training Endpoints', () => {
    it('should validate training request inputs', async () => {
      const response = await request(app)
        .post('/api/bridge/training')
        .set('Authorization', `Bearer ${token}`)
        .send({}); // Empty payload

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });

    it('should validate training status request', async () => {
      const response = await request(app)
        .get('/api/bridge/training/invalid-id')
        .set('Authorization', `Bearer ${token}`);

      // Even with invalid ID, API should respond (likely with error)
      expect(response.status).toBe(200).or(expect.response.status).toBe(500).or(expect.response.status).toBe(404);
    });
  });

  describe('Backtest Endpoints', () => {
    it('should validate backtest request inputs', async () => {
      const response = await request(app)
        .post('/api/bridge/backtest')
        .set('Authorization', `Bearer ${token}`)
        .send({}); // Empty payload

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error');
    });
  });
}); 