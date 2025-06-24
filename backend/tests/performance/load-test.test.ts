/**
 * Performance Load Tests
 * Tests system performance under various load conditions
 */

import request from 'supertest';
import { app } from '../../src/app';
import { createTestUser, createTestBot, generateLoadTestData, waitForCondition } from '../helpers/testHelpers';
import { logger } from '../../src/utils/logger';
import prisma from '../../src/utils/prismaClient';

describe('Performance Load Tests', () => {
  let testUsers: any[] = [];
  let authTokens: string[] = [];
  
  beforeAll(async () => {
    // Create multiple test users for load testing
    const loadTestData = generateLoadTestData(5, 2); // 5 users, 2 bots each
    
    for (const userData of loadTestData) {
      const user = await createTestUser({
        email: userData.email,
        username: userData.username,
        password: userData.password
      });
      
      // Get auth token
      const loginResponse = await request(app)
        .post('/api/auth/login')
        .send({
          email: user.email,
          password: user.password
        });
      
      const token = loginResponse.body.token;
      authTokens.push(token);
      
      // Create bots for this user
      const bots = [];
      for (const botData of userData.bots) {
        const bot = await createTestBot(user.id, botData);
        bots.push(bot);
      }
      
      testUsers.push({ ...user, bots, token });
    }
    
    logger.info(`Created ${testUsers.length} test users for load testing`);
  }, 30000);

  afterAll(async () => {
    // Cleanup all test data
    for (const user of testUsers) {
      try {
        await prisma.order.deleteMany({ where: { userId: user.id } });
        await prisma.position.deleteMany({ where: { userId: user.id } });
        await prisma.tradingSignal.deleteMany({ where: { userId: user.id } });
        
        for (const bot of user.bots) {
          await prisma.backtest.deleteMany({ where: { botId: bot.id } });
          await prisma.decisionLog.deleteMany({ where: { botId: bot.id } });
        }
        
        await prisma.bot.deleteMany({ where: { userId: user.id } });
        await prisma.user.delete({ where: { id: user.id } });
      } catch (error) {
        logger.error('Error cleaning up test user:', error);
      }
    }
    
    await prisma.$disconnect();
    logger.info('Load test cleanup completed');
  });

  describe('API Endpoint Performance', () => {
    it('should handle concurrent user authentication', async () => {
      const startTime = Date.now();
      const concurrentRequests = 20;
      
      // Create concurrent login requests
      const loginPromises = Array(concurrentRequests).fill(0).map(async (_, index) => {
        const userIndex = index % testUsers.length;
        const user = testUsers[userIndex];
        
        const response = await request(app)
          .post('/api/auth/login')
          .send({
            email: user.email,
            password: user.password
          });
        
        return {
          status: response.status,
          responseTime: Date.now() - startTime
        };
      });
      
      const results = await Promise.all(loginPromises);
      const endTime = Date.now();
      
      // Analyze results
      const successfulRequests = results.filter(r => r.status === 200).length;
      const averageResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
      const totalTime = endTime - startTime;
      const requestsPerSecond = concurrentRequests / (totalTime / 1000);
      
      expect(successfulRequests).toBe(concurrentRequests);
      expect(averageResponseTime).toBeLessThan(2000); // Less than 2 seconds
      expect(requestsPerSecond).toBeGreaterThan(5); // At least 5 RPS
      
      logger.info('Authentication performance:', {
        concurrentRequests,
        successfulRequests,
        averageResponseTime,
        requestsPerSecond,
        totalTime
      });
    }, 30000);

    it('should handle concurrent trading signal creation', async () => {
      const startTime = Date.now();
      const concurrentRequests = 50;
      
      // Create concurrent signal creation requests
      const signalPromises = Array(concurrentRequests).fill(0).map(async (_, index) => {
        const userIndex = index % testUsers.length;
        const user = testUsers[userIndex];
        const bot = user.bots[index % user.bots.length];
        
        const requestStart = Date.now();
        
        const response = await request(app)
          .post('/api/signals')
          .set('Authorization', `Bearer ${user.token}`)
          .send({
            botId: bot.id,
            symbol: bot.symbol,
            action: ['buy', 'sell', 'hold'][index % 3],
            strength: ['WEAK', 'MODERATE', 'STRONG'][index % 3],
            confidenceScore: Math.random() * 100,
            source: 'load_test',
            metadata: { loadTestIndex: index }
          });
        
        return {
          status: response.status,
          responseTime: Date.now() - requestStart,
          signalId: response.body.id
        };
      });
      
      const results = await Promise.all(signalPromises);
      const endTime = Date.now();
      
      // Analyze results
      const successfulRequests = results.filter(r => r.status === 201).length;
      const averageResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
      const totalTime = endTime - startTime;
      const requestsPerSecond = concurrentRequests / (totalTime / 1000);
      
      expect(successfulRequests).toBeGreaterThan(concurrentRequests * 0.9); // 90% success rate
      expect(averageResponseTime).toBeLessThan(1000); // Less than 1 second
      expect(requestsPerSecond).toBeGreaterThan(10); // At least 10 RPS
      
      logger.info('Signal creation performance:', {
        concurrentRequests,
        successfulRequests,
        averageResponseTime,
        requestsPerSecond,
        totalTime
      });
    }, 30000);

    it('should handle concurrent market data requests', async () => {
      const startTime = Date.now();
      const concurrentRequests = 100;
      const symbols = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD', 'DOTUSD'];
      
      // Create concurrent market data requests
      const marketDataPromises = Array(concurrentRequests).fill(0).map(async (_, index) => {
        const userIndex = index % testUsers.length;
        const user = testUsers[userIndex];
        const symbol = symbols[index % symbols.length];
        
        const requestStart = Date.now();
        
        const response = await request(app)
          .get(`/api/market-data/${symbol}`)
          .set('Authorization', `Bearer ${user.token}`);
        
        return {
          status: response.status,
          responseTime: Date.now() - requestStart,
          symbol
        };
      });
      
      const results = await Promise.all(marketDataPromises);
      const endTime = Date.now();
      
      // Analyze results
      const successfulRequests = results.filter(r => r.status === 200).length;
      const averageResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
      const totalTime = endTime - startTime;
      const requestsPerSecond = concurrentRequests / (totalTime / 1000);
      
      expect(successfulRequests).toBeGreaterThan(concurrentRequests * 0.95); // 95% success rate
      expect(averageResponseTime).toBeLessThan(500); // Less than 500ms
      expect(requestsPerSecond).toBeGreaterThan(20); // At least 20 RPS
      
      logger.info('Market data performance:', {
        concurrentRequests,
        successfulRequests,
        averageResponseTime,
        requestsPerSecond,
        totalTime
      });
    }, 30000);
  });

  describe('Database Performance', () => {
    it('should handle concurrent database operations', async () => {
      const startTime = Date.now();
      const concurrentOperations = 50;
      
      // Create concurrent database operations
      const dbPromises = Array(concurrentOperations).fill(0).map(async (_, index) => {
        const userIndex = index % testUsers.length;
        const user = testUsers[userIndex];
        const bot = user.bots[index % user.bots.length];
        
        const operationStart = Date.now();
        
        // Mix of different database operations
        const operations = [
          // Create signal
          () => prisma.tradingSignal.create({
            data: {
              userId: user.id,
              botId: bot.id,
              symbol: bot.symbol,
              action: 'buy',
              strength: 'MODERATE',
              confidenceScore: 75,
              source: 'db_load_test'
            }
          }),
          // Query positions
          () => prisma.position.findMany({
            where: { userId: user.id },
            take: 10
          }),
          // Query orders
          () => prisma.order.findMany({
            where: { userId: user.id },
            take: 10,
            orderBy: { createdAt: 'desc' }
          }),
          // Query bot performance
          () => prisma.bot.findUnique({
            where: { id: bot.id },
            include: {
              positions: { take: 5 },
              orders: { take: 5 }
            }
          })
        ];
        
        const operation = operations[index % operations.length];
        
        try {
          await operation();
          return {
            success: true,
            responseTime: Date.now() - operationStart,
            operation: index % operations.length
          };
        } catch (error) {
          return {
            success: false,
            responseTime: Date.now() - operationStart,
            operation: index % operations.length,
            error: error.message
          };
        }
      });
      
      const results = await Promise.all(dbPromises);
      const endTime = Date.now();
      
      // Analyze results
      const successfulOperations = results.filter(r => r.success).length;
      const averageResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
      const totalTime = endTime - startTime;
      const operationsPerSecond = concurrentOperations / (totalTime / 1000);
      
      expect(successfulOperations).toBeGreaterThan(concurrentOperations * 0.95); // 95% success rate
      expect(averageResponseTime).toBeLessThan(200); // Less than 200ms
      expect(operationsPerSecond).toBeGreaterThan(50); // At least 50 OPS
      
      logger.info('Database performance:', {
        concurrentOperations,
        successfulOperations,
        averageResponseTime,
        operationsPerSecond,
        totalTime
      });
    }, 30000);

    it('should handle large dataset queries efficiently', async () => {
      // Create a large dataset for testing
      const user = testUsers[0];
      const bot = user.bots[0];
      const recordCount = 1000;
      
      logger.info(`Creating ${recordCount} test records...`);
      
      // Create many signals
      const signalData = Array(recordCount).fill(0).map((_, index) => ({
        userId: user.id,
        botId: bot.id,
        symbol: bot.symbol,
        action: ['buy', 'sell', 'hold'][index % 3],
        strength: ['WEAK', 'MODERATE', 'STRONG'][index % 3],
        confidenceScore: Math.random() * 100,
        source: 'large_dataset_test',
        createdAt: new Date(Date.now() - index * 60000) // 1 minute apart
      }));
      
      await prisma.tradingSignal.createMany({
        data: signalData
      });
      
      // Test pagination performance
      const pageSize = 50;
      const pages = 10;
      const startTime = Date.now();
      
      for (let page = 0; page < pages; page++) {
        const pageStart = Date.now();
        
        const signals = await prisma.tradingSignal.findMany({
          where: { userId: user.id },
          orderBy: { createdAt: 'desc' },
          skip: page * pageSize,
          take: pageSize
        });
        
        const pageTime = Date.now() - pageStart;
        
        expect(signals.length).toBeLessThanOrEqual(pageSize);
        expect(pageTime).toBeLessThan(100); // Less than 100ms per page
      }
      
      const totalTime = Date.now() - startTime;
      const averagePageTime = totalTime / pages;
      
      expect(averagePageTime).toBeLessThan(50); // Less than 50ms average
      
      logger.info('Large dataset query performance:', {
        recordCount,
        pages,
        pageSize,
        totalTime,
        averagePageTime
      });
      
      // Cleanup large dataset
      await prisma.tradingSignal.deleteMany({
        where: { source: 'large_dataset_test' }
      });
    }, 60000);
  });

  describe('Memory and Resource Usage', () => {
    it('should maintain stable memory usage under load', async () => {
      const initialMemory = process.memoryUsage();
      const iterations = 100;
      
      // Perform memory-intensive operations
      for (let i = 0; i < iterations; i++) {
        const user = testUsers[i % testUsers.length];
        
        // Create and immediately clean up data
        const signal = await prisma.tradingSignal.create({
          data: {
            userId: user.id,
            botId: user.bots[0].id,
            symbol: 'BTCUSD',
            action: 'buy',
            strength: 'MODERATE',
            confidenceScore: 75,
            source: 'memory_test'
          }
        });
        
        await prisma.tradingSignal.delete({
          where: { id: signal.id }
        });
        
        // Force garbage collection every 10 iterations
        if (i % 10 === 0 && global.gc) {
          global.gc();
        }
      }
      
      const finalMemory = process.memoryUsage();
      const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;
      const memoryIncreasePercent = (memoryIncrease / initialMemory.heapUsed) * 100;
      
      // Memory increase should be reasonable (less than 50%)
      expect(memoryIncreasePercent).toBeLessThan(50);
      
      logger.info('Memory usage test:', {
        initialMemory: Math.round(initialMemory.heapUsed / 1024 / 1024),
        finalMemory: Math.round(finalMemory.heapUsed / 1024 / 1024),
        memoryIncrease: Math.round(memoryIncrease / 1024 / 1024),
        memoryIncreasePercent: Math.round(memoryIncreasePercent * 100) / 100,
        iterations
      });
    }, 30000);
  });

  describe('Stress Testing', () => {
    it('should handle system stress gracefully', async () => {
      const stressTestDuration = 10000; // 10 seconds
      const requestInterval = 100; // Every 100ms
      const startTime = Date.now();
      
      let requestCount = 0;
      let successCount = 0;
      let errorCount = 0;
      const responseTimes: number[] = [];
      
      // Create stress test interval
      const stressInterval = setInterval(async () => {
        if (Date.now() - startTime > stressTestDuration) {
          clearInterval(stressInterval);
          return;
        }
        
        requestCount++;
        const user = testUsers[requestCount % testUsers.length];
        const requestStart = Date.now();
        
        try {
          const response = await request(app)
            .get('/api/health')
            .set('Authorization', `Bearer ${user.token}`);
          
          const responseTime = Date.now() - requestStart;
          responseTimes.push(responseTime);
          
          if (response.status === 200) {
            successCount++;
          } else {
            errorCount++;
          }
        } catch (error) {
          errorCount++;
        }
      }, requestInterval);
      
      // Wait for stress test to complete
      await new Promise(resolve => setTimeout(resolve, stressTestDuration + 1000));
      
      // Analyze stress test results
      const averageResponseTime = responseTimes.reduce((sum, time) => sum + time, 0) / responseTimes.length;
      const maxResponseTime = Math.max(...responseTimes);
      const successRate = (successCount / requestCount) * 100;
      
      expect(successRate).toBeGreaterThan(90); // 90% success rate under stress
      expect(averageResponseTime).toBeLessThan(1000); // Less than 1 second average
      expect(maxResponseTime).toBeLessThan(5000); // Less than 5 seconds max
      
      logger.info('Stress test results:', {
        duration: stressTestDuration,
        requestCount,
        successCount,
        errorCount,
        successRate: Math.round(successRate * 100) / 100,
        averageResponseTime: Math.round(averageResponseTime),
        maxResponseTime
      });
    }, 15000);
  });
});