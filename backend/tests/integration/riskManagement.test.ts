import request from 'supertest';
import { PrismaClient } from '@prisma/client';
import { Server } from 'http';
import jwt from 'jsonwebtoken';
import { app } from '../../src/server';

// Initialize variables
let server: Server;
let prisma: PrismaClient;
let authToken: string;
let userId: string;

// Test user data
const testUser = {
  email: 'test.risk@example.com',
  password: 'Password123!',
  name: 'Risk Test User',
};

// Create a valid JWT token for authentication
const createToken = (id: string): string => {
  return jwt.sign(
    { id },
    process.env.JWT_SECRET || 'test-secret',
    { expiresIn: '1h' }
  );
};

describe('Risk Management Integration Tests', () => {
  beforeAll(async () => {
    // Start server
    server = app.listen(0); // Use any available port
    
    // Connect to test database
    prisma = new PrismaClient();
    
    // Create test user
    try {
      const existingUser = await prisma.user.findUnique({
        where: { email: testUser.email }
      });
      
      if (existingUser) {
        userId = existingUser.id;
      } else {
        const newUser = await prisma.user.create({
          data: {
            email: testUser.email,
            password: testUser.password, // Note: In a real app, this would be hashed
            name: testUser.name,
            role: 'USER'
          }
        });
        userId = newUser.id;
      }
      
      // Create auth token
      authToken = createToken(userId);
      
    } catch (error) {
      console.error('Setup error:', error);
      throw error;
    }
  });
  
  afterAll(async () => {
    // Close server and database connection
    server.close();
    await prisma.$disconnect();
  });
  
  describe('Risk Settings API', () => {
    test('GET /api/risk/settings - Should return risk settings', async () => {
      const response = await request(app)
        .get('/api/risk/settings')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
    });
    
    test('PUT /api/risk/settings - Should update risk settings', async () => {
      const updateData = {
        maxPositionSize: 0.05,
        maxDrawdown: 5,
        riskLevel: 'LOW',
        defaultRiskPerTrade: 0.5
      };
      
      const response = await request(app)
        .put('/api/risk/settings')
        .set('Authorization', `Bearer ${authToken}`)
        .send(updateData);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toMatchObject(updateData);
    });
  });
  
  describe('Position Sizing API', () => {
    test('POST /api/risk/position-size - Should calculate position size', async () => {
      const positionParams = {
        accountBalance: 10000,
        symbol: 'BTC/USD',
        riskAmount: 100,
        entryPrice: 40000,
        stopLossPrice: 39000,
        method: 'FIXED_AMOUNT'
      };
      
      const response = await request(app)
        .post('/api/risk/position-size')
        .set('Authorization', `Bearer ${authToken}`)
        .send(positionParams);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveProperty('positionSize');
      expect(response.body.data).toHaveProperty('riskAmount');
      expect(response.body.data).toHaveProperty('riskPercentage');
    });
  });
  
  describe('Risk Assessment API', () => {
    test('POST /api/risk/assess-trade - Should assess trade risk', async () => {
      const tradeParams = {
        symbol: 'BTC/USD',
        direction: 'long',
        entryPrice: 40000,
        positionSize: 0.01,
        stopLossPrice: 39000,
        takeProfitPrice: 42000,
        accountBalance: 10000
      };
      
      const response = await request(app)
        .post('/api/risk/assess-trade')
        .set('Authorization', `Bearer ${authToken}`)
        .send(tradeParams);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveProperty('riskLevel');
      expect(response.body.data).toHaveProperty('isWithinRiskTolerance');
      expect(response.body.data).toHaveProperty('factors');
    });
    
    test('GET /api/risk/portfolio - Should assess portfolio risk', async () => {
      const response = await request(app)
        .get('/api/risk/portfolio?accountBalance=10000')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(response.body.data).toHaveProperty('overallRiskLevel');
      expect(response.body.data).toHaveProperty('metrics');
    });
  });
  
  describe('Risk Alerts API', () => {
    test('GET /api/risk/alerts - Should get risk alerts', async () => {
      const response = await request(app)
        .get('/api/risk/alerts')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(Array.isArray(response.body.data)).toBe(true);
    });
  });
  
  describe('Circuit Breakers API', () => {
    test('GET /api/risk/circuit-breakers - Should get active circuit breakers', async () => {
      const response = await request(app)
        .get('/api/risk/circuit-breakers')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(response.status).toBe(200);
      expect(response.body.success).toBe(true);
      expect(Array.isArray(response.body.data)).toBe(true);
    });
    
    // Note: We can't reliably test resetting a circuit breaker in integration tests
    // since we would need to create one first, which depends on specific conditions
  });
}); 