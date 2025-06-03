#!/usr/bin/env node

/**
 * Test Systems Script
 * Comprehensive testing for QuestDB migration and Redis Streams
 */

import { mockQuestdbService } from '../services/mockQuestdbService';
import { mockRedisStreamsService } from '../services/mockRedisStreamsService';
import { redisConnection, validateRedisEnvironment } from '../config/redis';
import { logger } from '../utils/logger';
import { createEventId, createCorrelationId } from '../types/events';

interface TestResult {
  name: string;
  success: boolean;
  duration: number;
  details?: any;
  error?: string;
}

class SystemTester {
  private results: TestResult[] = [];

  public async runAllTests(): Promise<void> {
    logger.info('🧪 Starting comprehensive system tests...');

    // Test QuestDB (Mock)
    await this.testQuestDBMock();

    // Test Redis Connection
    await this.testRedisConnection();

    // Test Redis Streams (Mock)
    await this.testRedisStreamsMock();

    // Test Integration
    await this.testIntegration();

    // Display results
    this.displayResults();
  }

  private async testQuestDBMock(): Promise<void> {
    const testName = 'QuestDB Mock Service';
    const startTime = Date.now();

    try {
      logger.info('📊 Testing QuestDB Mock Service...');

      // Initialize service
      await mockQuestdbService.initialize();

      // Test metric insertion
      await mockQuestdbService.insertMetric({
        timestamp: new Date(),
        name: 'test_metric',
        value: 42.5,
        tags: { source: 'test', environment: 'development' },
      });

      // Test batch metric insertion
      const metrics = Array.from({ length: 100 }, (_, i) => ({
        timestamp: new Date(Date.now() - i * 60000),
        name: `test_metric_${i % 5}`,
        value: Math.random() * 100,
        tags: { batch: 'true', index: i.toString() },
      }));

      await mockQuestdbService.insertMetrics(metrics);

      // Test trading signal insertion
      await mockQuestdbService.insertTradingSignal({
        timestamp: new Date(),
        id: createEventId(),
        symbol: 'BTCUSD',
        type: 'ENTRY',
        direction: 'LONG',
        strength: 'STRONG',
        timeframe: '1h',
        price: 45000,
        targetPrice: 47000,
        stopLoss: 43000,
        confidenceScore: 85,
        expectedReturn: 0.044,
        expectedRisk: 0.022,
        riskRewardRatio: 2.0,
        source: 'test-model',
      });

      // Test query functionality
      const queryResult = await mockQuestdbService.executeQuery('SELECT count() FROM metrics');
      
      // Get statistics
      const stats = mockQuestdbService.getStats();

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          totalRecords: stats.totalRecords,
          tableStats: stats.tableStats,
          queryResult,
        },
      });

      logger.info(`✅ ${testName} passed`);

    } catch (error) {
      this.results.push({
        name: testName,
        success: false,
        duration: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      logger.error(`❌ ${testName} failed:`, error);
    }
  }

  private async testRedisConnection(): Promise<void> {
    const testName = 'Redis Connection';
    const startTime = Date.now();

    try {
      logger.info('🔌 Testing Redis Connection...');

      // Validate environment
      const envValidation = validateRedisEnvironment();
      if (!envValidation.valid) {
        throw new Error(`Environment validation failed: ${envValidation.errors.join(', ')}`);
      }

      // Test connection
      await redisConnection.connect();
      
      const healthCheck = await redisConnection.healthCheck();
      if (!healthCheck) {
        throw new Error('Redis health check failed');
      }

      const stats = redisConnection.getStats();

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          isConnected: stats.isConnected,
          config: {
            host: stats.config.host,
            port: stats.config.port,
            db: stats.config.db,
          },
        },
      });

      logger.info(`✅ ${testName} passed`);

    } catch (error) {
      this.results.push({
        name: testName,
        success: false,
        duration: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      logger.error(`❌ ${testName} failed:`, error);
    }
  }

  private async testRedisStreamsMock(): Promise<void> {
    const testName = 'Redis Streams Mock Service';
    const startTime = Date.now();

    try {
      logger.info('📡 Testing Redis Streams Mock Service...');

      // Initialize service
      await mockRedisStreamsService.initialize();

      // Test health check
      const healthCheck = await mockRedisStreamsService.healthCheck();
      if (!healthCheck) {
        throw new Error('Redis Streams Mock health check failed');
      }

      // Test event publishing
      const testEvent = {
        id: createEventId(),
        type: 'MARKET_DATA_RECEIVED' as const,
        timestamp: Date.now(),
        version: '1.0',
        source: 'test-system',
        correlationId: createCorrelationId(),
        data: {
          symbol: 'BTCUSD',
          exchange: 'test-exchange',
          price: 45000,
          volume: 1.5,
          timestamp: Date.now(),
        },
      };

      const messageId = await mockRedisStreamsService.publishEvent('market-data-stream', testEvent);

      // Test batch publishing
      const batchEvents = Array.from({ length: 10 }, (_, i) => ({
        id: createEventId(),
        type: 'MARKET_DATA_RECEIVED' as const,
        timestamp: Date.now(),
        version: '1.0',
        source: 'test-batch',
        data: {
          symbol: 'BTCUSD',
          exchange: 'test-exchange',
          price: 45000 + i * 10,
          volume: Math.random() * 2,
          timestamp: Date.now(),
        },
      }));

      const batchMessageIds = await mockRedisStreamsService.publishEvents('market-data-stream', batchEvents);

      // Test reading events
      const readEvents = await mockRedisStreamsService.readEvents('market-data-stream', {
        groupName: 'test-group',
        consumerName: 'test-consumer',
        count: 5,
      });

      // Get service stats
      const stats = await mockRedisStreamsService.getStats();
      const mockStats = mockRedisStreamsService.getMockStats();

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          messageId,
          batchMessageIds: batchMessageIds.length,
          readEvents: readEvents.length,
          stats,
          mockStats,
        },
      });

      logger.info(`✅ ${testName} passed`);

    } catch (error) {
      this.results.push({
        name: testName,
        success: false,
        duration: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      logger.error(`❌ ${testName} failed:`, error);
    }
  }

  private async testEventDrivenSystem(): Promise<void> {
    const testName = 'Event-Driven Trading System';
    const startTime = Date.now();

    try {
      logger.info('🚀 Testing Event-Driven Trading System...');

      // Test system health before starting
      const initialHealth = eventDrivenTradingSystem.getHealthStatus();
      
      // Test event publishing without starting the full system
      const marketDataMessageId = await eventDrivenTradingSystem.publishMarketDataEvent(
        'BTCUSD',
        45000,
        1.5,
        'test-exchange'
      );

      const signalMessageId = await eventDrivenTradingSystem.publishTradingSignalEvent({
        signalId: createEventId(),
        symbol: 'BTCUSD',
        signalType: 'ENTRY',
        direction: 'LONG',
        strength: 'STRONG',
        timeframe: '1h',
        price: 45000,
        confidenceScore: 85,
        expectedReturn: 0.044,
        expectedRisk: 0.022,
        riskRewardRatio: 2.0,
        modelSource: 'test-model',
      });

      const systemStats = eventDrivenTradingSystem.getStats();
      const systemConfig = eventDrivenTradingSystem.getConfig();

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          initialHealth,
          marketDataMessageId,
          signalMessageId,
          systemStats,
          systemConfig,
        },
      });

      logger.info(`✅ ${testName} passed`);

    } catch (error) {
      this.results.push({
        name: testName,
        success: false,
        duration: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      logger.error(`❌ ${testName} failed:`, error);
    }
  }

  private async testIntegration(): Promise<void> {
    const testName = 'System Integration';
    const startTime = Date.now();

    try {
      logger.info('🔗 Testing System Integration...');

      // Test data flow: Market Data -> QuestDB
      const marketData = {
        timestamp: new Date(),
        name: 'btc_price',
        value: 45000,
        tags: { symbol: 'BTCUSD', exchange: 'test' },
      };

      await mockQuestdbService.insertMetric(marketData);

      // Test event flow: Market Data Event -> Redis Streams
      const marketEvent = {
        id: createEventId(),
        type: 'MARKET_DATA_RECEIVED' as const,
        timestamp: Date.now(),
        version: '1.0',
        source: 'integration-test',
        data: {
          symbol: 'BTCUSD',
          exchange: 'test-exchange',
          price: 45000,
          volume: 1.5,
          timestamp: Date.now(),
        },
      };

      const eventMessageId = await redisStreamsService.publishEvent('market-data-stream', marketEvent);

      // Verify data storage
      const questdbStats = mockQuestdbService.getStats();
      const redisStats = await redisStreamsService.getStats();

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          eventMessageId,
          questdbStats,
          redisStats: redisStats.isInitialized,
        },
      });

      logger.info(`✅ ${testName} passed`);

    } catch (error) {
      this.results.push({
        name: testName,
        success: false,
        duration: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
      });

      logger.error(`❌ ${testName} failed:`, error);
    }
  }

  private displayResults(): void {
    logger.info('\n📊 Test Results Summary:');
    logger.info('=' .repeat(80));

    let totalTests = this.results.length;
    let passedTests = this.results.filter(r => r.success).length;
    let failedTests = totalTests - passedTests;

    this.results.forEach(result => {
      const status = result.success ? '✅ PASS' : '❌ FAIL';
      const duration = `${result.duration}ms`;
      
      logger.info(`${status} ${result.name.padEnd(30)} (${duration})`);
      
      if (!result.success && result.error) {
        logger.error(`     Error: ${result.error}`);
      }
      
      if (result.details) {
        logger.info(`     Details: ${JSON.stringify(result.details, null, 2)}`);
      }
    });

    logger.info('=' .repeat(80));
    logger.info(`Total Tests: ${totalTests}`);
    logger.info(`Passed: ${passedTests}`);
    logger.info(`Failed: ${failedTests}`);
    logger.info(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);

    if (failedTests > 0) {
      logger.error(`\n❌ ${failedTests} test(s) failed. Please check the errors above.`);
      process.exit(1);
    } else {
      logger.info(`\n🎉 All tests passed successfully!`);
    }
  }

  public async cleanup(): Promise<void> {
    try {
      await mockQuestdbService.shutdown();
      await redisStreamsService.shutdown();
      await redisConnection.disconnect();
      logger.info('🧹 Cleanup completed');
    } catch (error) {
      logger.error('❌ Cleanup failed:', error);
    }
  }
}

// Main execution
async function main() {
  const tester = new SystemTester();
  
  try {
    await tester.runAllTests();
  } catch (error) {
    logger.error('💥 Test execution failed:', error);
    process.exit(1);
  } finally {
    await tester.cleanup();
  }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  logger.info('🛑 Received SIGINT, cleaning up...');
  const tester = new SystemTester();
  await tester.cleanup();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  logger.info('🛑 Received SIGTERM, cleaning up...');
  const tester = new SystemTester();
  await tester.cleanup();
  process.exit(0);
});

// Run tests if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { SystemTester };
