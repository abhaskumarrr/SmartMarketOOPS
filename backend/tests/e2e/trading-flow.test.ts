/**
 * End-to-End Trading Flow Tests
 * Tests the complete trading pipeline from signal generation to order execution
 */

import request from 'supertest';
import { app } from '../../src/app';
import prisma from '../../src/utils/prismaClient';
import { EnhancedMLModel } from '../../../ml/src/enhanced_ml_model';
import { FibonacciMLModel } from '../../../ml/src/fibonacci_ml_model';
import { createTestUser, createTestBot, generateTestMarketData } from '../helpers/testHelpers';
import { logger } from '../../src/utils/logger';

describe('End-to-End Trading Flow', () => {
  let testUser: any;
  let testBot: any;
  let authToken: string;
  let enhancedModel: EnhancedMLModel;
  let fibonacciModel: FibonacciMLModel;

  beforeAll(async () => {
    // Create test user and bot
    testUser = await createTestUser();
    testBot = await createTestBot(testUser.id);
    
    // Get authentication token
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: testUser.email,
        password: 'testpassword123'
      });
    
    authToken = loginResponse.body.token;
    
    // Initialize ML models
    enhancedModel = new EnhancedMLModel();
    fibonacciModel = new FibonacciMLModel();
    
    logger.info('E2E test setup completed');
  });

  afterAll(async () => {
    // Cleanup test data
    await prisma.order.deleteMany({ where: { userId: testUser.id } });
    await prisma.position.deleteMany({ where: { userId: testUser.id } });
    await prisma.backtest.deleteMany({ where: { botId: testBot.id } });
    await prisma.bot.delete({ where: { id: testBot.id } });
    await prisma.user.delete({ where: { id: testUser.id } });
    
    await prisma.$disconnect();
    logger.info('E2E test cleanup completed');
  });

  describe('Complete Trading Pipeline', () => {
    it('should execute complete trading flow: signal → risk check → order → execution', async () => {
      // Step 1: Generate market data
      const marketData = generateTestMarketData('BTCUSD', 100);
      
      // Step 2: Generate ML predictions
      const enhancedPrediction = enhancedModel.predict(marketData);
      const fibonacciPrediction = fibonacciModel.predict(marketData);
      
      expect(enhancedPrediction).toHaveProperty('action');
      expect(enhancedPrediction).toHaveProperty('confidence');
      expect(fibonacciPrediction).toHaveProperty('action');
      expect(fibonacciPrediction).toHaveProperty('fibonacci_level');
      
      // Step 3: Create trading signal
      const signalResponse = await request(app)
        .post('/api/signals')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          botId: testBot.id,
          symbol: 'BTCUSD',
          action: enhancedPrediction.action,
          strength: 'STRONG',
          confidenceScore: enhancedPrediction.confidence * 100,
          source: 'enhanced_ml_model',
          metadata: {
            enhancedPrediction,
            fibonacciPrediction
          }
        });
      
      expect(signalResponse.status).toBe(201);
      expect(signalResponse.body).toHaveProperty('id');
      
      const signalId = signalResponse.body.id;
      
      // Step 4: Risk assessment
      const riskResponse = await request(app)
        .post('/api/risk/assess')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          userId: testUser.id,
          botId: testBot.id,
          symbol: 'BTCUSD',
          action: enhancedPrediction.action,
          amount: enhancedPrediction.position_size || 0.1,
          price: 50000
        });
      
      expect(riskResponse.status).toBe(200);
      expect(riskResponse.body).toHaveProperty('approved');
      
      // Step 5: Create order if risk approved
      if (riskResponse.body.approved) {
        const orderResponse = await request(app)
          .post('/api/trading/orders')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            botId: testBot.id,
            symbol: 'BTCUSD',
            side: enhancedPrediction.action === 'buy' ? 'buy' : 'sell',
            amount: riskResponse.body.adjustedAmount || enhancedPrediction.position_size,
            type: 'market',
            signalId: signalId
          });
        
        expect(orderResponse.status).toBe(201);
        expect(orderResponse.body).toHaveProperty('id');
        expect(orderResponse.body.status).toBe('pending');
        
        // Step 6: Simulate order execution
        const orderId = orderResponse.body.id;
        
        // Wait for order processing
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Check order status
        const orderStatusResponse = await request(app)
          .get(`/api/trading/orders/${orderId}`)
          .set('Authorization', `Bearer ${authToken}`);
        
        expect(orderStatusResponse.status).toBe(200);
        expect(['pending', 'filled', 'partially_filled']).toContain(orderStatusResponse.body.status);
        
        // Step 7: Check position creation (if order filled)
        if (orderStatusResponse.body.status === 'filled') {
          const positionsResponse = await request(app)
            .get('/api/positions')
            .set('Authorization', `Bearer ${authToken}`)
            .query({ symbol: 'BTCUSD' });
          
          expect(positionsResponse.status).toBe(200);
          expect(positionsResponse.body.positions.length).toBeGreaterThan(0);
          
          const position = positionsResponse.body.positions[0];
          expect(position.symbol).toBe('BTCUSD');
          expect(position.status).toBe('Open');
        }
      }
      
      logger.info('Complete trading flow test passed');
    }, 30000); // 30 second timeout for complete flow

    it('should handle risk rejection properly', async () => {
      // Create a high-risk order that should be rejected
      const riskResponse = await request(app)
        .post('/api/risk/assess')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          userId: testUser.id,
          botId: testBot.id,
          symbol: 'BTCUSD',
          action: 'buy',
          amount: 10.0, // Very large amount
          price: 50000
        });
      
      expect(riskResponse.status).toBe(200);
      
      // If risk is rejected, order creation should fail
      if (!riskResponse.body.approved) {
        const orderResponse = await request(app)
          .post('/api/trading/orders')
          .set('Authorization', `Bearer ${authToken}`)
          .send({
            botId: testBot.id,
            symbol: 'BTCUSD',
            side: 'buy',
            amount: 10.0,
            type: 'market'
          });
        
        expect(orderResponse.status).toBe(400);
        expect(orderResponse.body).toHaveProperty('error');
      }
      
      logger.info('Risk rejection test passed');
    });

    it('should process backtest end-to-end', async () => {
      // Step 1: Create backtest configuration
      const backtestConfig = {
        botId: testBot.id,
        symbol: 'BTCUSD',
        timeframe: '1h',
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-01-31'),
        initialCapital: 10000,
        leverage: 1.0,
        riskPerTrade: 0.02
      };
      
      // Step 2: Run backtest
      const backtestResponse = await request(app)
        .post('/api/bots/backtest')
        .set('Authorization', `Bearer ${authToken}`)
        .send(backtestConfig);
      
      expect(backtestResponse.status).toBe(201);
      expect(backtestResponse.body).toHaveProperty('id');
      
      const backtestId = backtestResponse.body.id;
      
      // Step 3: Wait for backtest completion
      let backtestStatus = 'running';
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds max wait
      
      while (backtestStatus === 'running' && attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const statusResponse = await request(app)
          .get(`/api/bots/backtest/${backtestId}`)
          .set('Authorization', `Bearer ${authToken}`);
        
        expect(statusResponse.status).toBe(200);
        backtestStatus = statusResponse.body.status;
        attempts++;
      }
      
      expect(backtestStatus).toBe('completed');
      
      // Step 4: Verify backtest results
      const resultsResponse = await request(app)
        .get(`/api/bots/backtest/${backtestId}`)
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(resultsResponse.status).toBe(200);
      expect(resultsResponse.body).toHaveProperty('performance');
      expect(resultsResponse.body.performance).toHaveProperty('totalReturn');
      expect(resultsResponse.body.performance).toHaveProperty('sharpeRatio');
      expect(resultsResponse.body.performance).toHaveProperty('maxDrawdown');
      
      logger.info('Backtest end-to-end test passed');
    }, 60000); // 60 second timeout for backtest
  });

  describe('ML Model Integration', () => {
    it('should integrate Enhanced ML Model predictions', async () => {
      const marketData = generateTestMarketData('ETHUSD', 50);
      
      // Test model prediction
      const prediction = enhancedModel.predict(marketData);
      
      expect(prediction).toHaveProperty('action');
      expect(['buy', 'sell', 'hold']).toContain(prediction.action);
      expect(prediction.confidence).toBeGreaterThanOrEqual(0);
      expect(prediction.confidence).toBeLessThanOrEqual(1);
      
      // Test signal creation from ML prediction
      const signalResponse = await request(app)
        .post('/api/signals')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          botId: testBot.id,
          symbol: 'ETHUSD',
          action: prediction.action,
          strength: prediction.confidence > 0.7 ? 'STRONG' : 'WEAK',
          confidenceScore: prediction.confidence * 100,
          source: 'enhanced_ml_model',
          metadata: prediction
        });
      
      expect(signalResponse.status).toBe(201);
      
      logger.info('Enhanced ML Model integration test passed');
    });

    it('should integrate Fibonacci ML Model predictions', async () => {
      const marketData = generateTestMarketData('ETHUSD', 50);
      
      // Test model prediction
      const prediction = fibonacciModel.predict(marketData);
      
      expect(prediction).toHaveProperty('action');
      expect(prediction).toHaveProperty('fibonacci_level');
      expect(prediction).toHaveProperty('trend_direction');
      expect(['buy', 'sell', 'hold']).toContain(prediction.action);
      
      // Test signal creation from Fibonacci prediction
      const signalResponse = await request(app)
        .post('/api/signals')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          botId: testBot.id,
          symbol: 'ETHUSD',
          action: prediction.action,
          strength: prediction.confidence > 0.6 ? 'STRONG' : 'WEAK',
          confidenceScore: prediction.confidence * 100,
          source: 'fibonacci_ml_model',
          metadata: prediction
        });
      
      expect(signalResponse.status).toBe(201);
      
      logger.info('Fibonacci ML Model integration test passed');
    });
  });

  describe('Event Processing Integration', () => {
    it('should process events through all processors', async () => {
      // Create a market data event
      const marketDataEvent = {
        id: `test-event-${Date.now()}`,
        type: 'market_data',
        timestamp: Date.now(),
        data: {
          symbol: 'BTCUSD',
          price: 50000,
          volume: 1000,
          previousPrice: 49500
        }
      };
      
      // Send event to processing pipeline
      const eventResponse = await request(app)
        .post('/api/events/process')
        .set('Authorization', `Bearer ${authToken}`)
        .send(marketDataEvent);
      
      expect(eventResponse.status).toBe(200);
      expect(eventResponse.body).toHaveProperty('processed');
      expect(eventResponse.body.processed).toBe(true);
      
      // Verify event was processed by checking system metrics
      const metricsResponse = await request(app)
        .get('/api/metrics/system')
        .set('Authorization', `Bearer ${authToken}`);
      
      expect(metricsResponse.status).toBe(200);
      expect(metricsResponse.body).toHaveProperty('eventsProcessed');
      
      logger.info('Event processing integration test passed');
    });
  });

  describe('Database Integration', () => {
    it('should maintain data consistency across operations', async () => {
      // Create order
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
      
      // Verify order in database
      const order = await prisma.order.findUnique({
        where: { id: orderId }
      });
      
      expect(order).toBeTruthy();
      expect(order!.userId).toBe(testUser.id);
      expect(order!.botId).toBe(testBot.id);
      expect(order!.symbol).toBe('BTCUSD');
      
      // Update order status
      await prisma.order.update({
        where: { id: orderId },
        data: { status: 'filled' }
      });
      
      // Verify update
      const updatedOrder = await prisma.order.findUnique({
        where: { id: orderId }
      });
      
      expect(updatedOrder!.status).toBe('filled');
      
      logger.info('Database integration test passed');
    });
  });
});