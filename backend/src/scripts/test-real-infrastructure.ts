#!/usr/bin/env node

/**
 * Real Infrastructure Test Script
 * Comprehensive testing with actual Redis and QuestDB instances
 */

import { questdbConnection } from '../config/questdb';
import { questdbService } from '../services/questdbService';
import { redisConnection, validateRedisEnvironment } from '../config/redis';
import { redisStreamsService } from '../services/redisStreamsService';
import { eventDrivenTradingSystem } from '../services/eventDrivenTradingSystem';
import { logger } from '../utils/logger';
import { createEventId, createCorrelationId } from '../types/events';

interface TestResult {
  name: string;
  success: boolean;
  duration: number;
  details?: any;
  error?: string;
  metrics?: {
    throughput?: number;
    latency?: number;
    errorRate?: number;
  };
}

class RealInfrastructureTester {
  private results: TestResult[] = [];

  public async runAllTests(): Promise<void> {
    logger.info('🧪 Starting comprehensive real infrastructure tests...');

    // Test QuestDB
    await this.testQuestDBConnection();
    await this.testQuestDBOperations();
    await this.testQuestDBPerformance();

    // Test Redis
    await this.testRedisConnection();
    await this.testRedisStreams();
    await this.testRedisStreamsPerformance();

    // Test Integration
    await this.testEndToEndIntegration();
    await this.testEventProcessingPipeline();

    // Display results
    this.displayResults();
  }

  private async testQuestDBConnection(): Promise<void> {
    const testName = 'QuestDB Connection';
    const startTime = Date.now();

    try {
      logger.info('📊 Testing QuestDB Connection...');

      // Test connection
      await questdbConnection.connect();
      
      const healthCheck = await questdbConnection.healthCheck();
      if (!healthCheck) {
        throw new Error('QuestDB health check failed');
      }

      const stats = questdbConnection.getStats();

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          isConnected: stats.isConnected,
          config: stats.clientOptions,
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

  private async testQuestDBOperations(): Promise<void> {
    const testName = 'QuestDB Operations';
    const startTime = Date.now();

    try {
      logger.info('📊 Testing QuestDB Operations...');

      // Initialize service
      await questdbService.initialize();

      // Test metric insertion
      await questdbService.insertMetric({
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

      await questdbService.insertMetrics(metrics);

      // Test trading signal insertion
      await questdbService.insertTradingSignal({
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
      const queryResult = await questdbService.executeQuery('SELECT count() FROM metrics');
      const tableStats = await questdbService.getTableStats('metrics');

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          metricsInserted: metrics.length + 1,
          queryResult,
          tableStats,
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

  private async testQuestDBPerformance(): Promise<void> {
    const testName = 'QuestDB Performance';
    const startTime = Date.now();

    try {
      logger.info('📊 Testing QuestDB Performance...');

      const batchSize = 1000;
      const batches = 5;
      let totalInserted = 0;
      let totalErrors = 0;

      for (let batch = 0; batch < batches; batch++) {
        const batchStartTime = Date.now();
        
        const metrics = Array.from({ length: batchSize }, (_, i) => ({
          timestamp: new Date(Date.now() - (batch * batchSize + i) * 1000),
          name: `perf_test_metric_${i % 10}`,
          value: Math.random() * 1000,
          tags: { 
            batch: batch.toString(),
            test: 'performance',
            symbol: ['BTCUSD', 'ETHUSD', 'ADAUSD'][i % 3],
          },
        }));

        try {
          await questdbService.insertMetrics(metrics);
          totalInserted += batchSize;
        } catch (error) {
          totalErrors++;
          logger.error(`Batch ${batch} failed:`, error);
        }

        const batchDuration = Date.now() - batchStartTime;
        logger.info(`Batch ${batch + 1}/${batches}: ${batchSize} records in ${batchDuration}ms (${(batchSize / (batchDuration / 1000)).toFixed(0)} records/sec)`);
      }

      const totalDuration = Date.now() - startTime;
      const throughput = totalInserted / (totalDuration / 1000);
      const errorRate = (totalErrors / batches) * 100;

      this.results.push({
        name: testName,
        success: totalErrors === 0,
        duration: totalDuration,
        details: {
          totalInserted,
          totalErrors,
          batches,
          batchSize,
        },
        metrics: {
          throughput,
          errorRate,
        },
      });

      logger.info(`✅ ${testName} completed: ${totalInserted} records, ${throughput.toFixed(0)} records/sec`);

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

  private async testRedisStreams(): Promise<void> {
    const testName = 'Redis Streams Operations';
    const startTime = Date.now();

    try {
      logger.info('📡 Testing Redis Streams Operations...');

      // Initialize service
      await redisStreamsService.initialize();

      // Test health check
      const healthCheck = await redisStreamsService.healthCheck();
      if (!healthCheck) {
        throw new Error('Redis Streams health check failed');
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

      const messageId = await redisStreamsService.publishEvent('market-data-stream', testEvent);

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

      const batchMessageIds = await redisStreamsService.publishEvents('market-data-stream', batchEvents);

      // Test reading events
      const readEvents = await redisStreamsService.readEvents('market-data-stream', {
        groupName: 'test-group',
        consumerName: 'test-consumer',
        count: 5,
      });

      // Get service stats
      const stats = await redisStreamsService.getStats();

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          messageId,
          batchMessageIds: batchMessageIds.length,
          readEvents: readEvents.length,
          stats,
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

  private async testRedisStreamsPerformance(): Promise<void> {
    const testName = 'Redis Streams Performance';
    const startTime = Date.now();

    try {
      logger.info('📡 Testing Redis Streams Performance...');

      const eventCount = 1000;
      const events = Array.from({ length: eventCount }, (_, i) => ({
        id: createEventId(),
        type: 'MARKET_DATA_RECEIVED' as const,
        timestamp: Date.now(),
        version: '1.0',
        source: 'performance-test',
        data: {
          symbol: ['BTCUSD', 'ETHUSD', 'ADAUSD'][i % 3],
          exchange: 'test-exchange',
          price: 45000 + Math.random() * 1000,
          volume: Math.random() * 10,
          timestamp: Date.now(),
        },
      }));

      // Test batch publishing performance
      const publishStartTime = Date.now();
      const messageIds = await redisStreamsService.publishEvents('market-data-stream', events);
      const publishDuration = Date.now() - publishStartTime;
      const publishThroughput = eventCount / (publishDuration / 1000);

      this.results.push({
        name: testName,
        success: true,
        duration: Date.now() - startTime,
        details: {
          eventsPublished: eventCount,
          messageIds: messageIds.length,
          publishDuration,
        },
        metrics: {
          throughput: publishThroughput,
        },
      });

      logger.info(`✅ ${testName} completed: ${eventCount} events, ${publishThroughput.toFixed(0)} events/sec`);

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

  private async testEndToEndIntegration(): Promise<void> {
    const testName = 'End-to-End Integration';
    const startTime = Date.now();

    try {
      logger.info('🔗 Testing End-to-End Integration...');

      // Test data flow: Market Data Event -> Redis Streams -> QuestDB
      const marketEvent = {
        id: createEventId(),
        type: 'MARKET_DATA_RECEIVED' as const,
        timestamp: Date.now(),
        version: '1.0',
        source: 'integration-test',
        correlationId: createCorrelationId(),
        data: {
          symbol: 'BTCUSD',
          exchange: 'test-exchange',
          price: 45000,
          volume: 1.5,
          timestamp: Date.now(),
        },
      };

      // Publish to Redis Streams
      const eventMessageId = await redisStreamsService.publishEvent('market-data-stream', marketEvent);

      // Store corresponding metric in QuestDB
      await questdbService.insertMetric({
        timestamp: new Date(marketEvent.data.timestamp),
        name: 'btc_price',
        value: marketEvent.data.price,
        tags: { 
          symbol: marketEvent.data.symbol, 
          exchange: marketEvent.data.exchange,
          source: 'integration-test',
        },
      });

      // Verify data storage
      const questdbStats = await questdbService.getTableStats('metrics');
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

  private async testEventProcessingPipeline(): Promise<void> {
    const testName = 'Event Processing Pipeline';
    const startTime = Date.now();

    try {
      logger.info('🚀 Testing Event Processing Pipeline...');

      // Test event publishing through the trading system
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

  private displayResults(): void {
    logger.info('\n📊 Real Infrastructure Test Results Summary:');
    logger.info('=' .repeat(80));

    let totalTests = this.results.length;
    let passedTests = this.results.filter(r => r.success).length;
    let failedTests = totalTests - passedTests;

    this.results.forEach(result => {
      const status = result.success ? '✅ PASS' : '❌ FAIL';
      const duration = `${result.duration}ms`;
      
      logger.info(`${status} ${result.name.padEnd(30)} (${duration})`);
      
      if (result.metrics) {
        if (result.metrics.throughput) {
          logger.info(`     Throughput: ${result.metrics.throughput.toFixed(0)} ops/sec`);
        }
        if (result.metrics.latency) {
          logger.info(`     Latency: ${result.metrics.latency.toFixed(2)}ms`);
        }
        if (result.metrics.errorRate !== undefined) {
          logger.info(`     Error Rate: ${result.metrics.errorRate.toFixed(1)}%`);
        }
      }
      
      if (!result.success && result.error) {
        logger.error(`     Error: ${result.error}`);
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
      logger.info('\n📈 Performance Summary:');
      
      const performanceResults = this.results.filter(r => r.metrics);
      performanceResults.forEach(result => {
        if (result.metrics?.throughput) {
          logger.info(`  ${result.name}: ${result.metrics.throughput.toFixed(0)} ops/sec`);
        }
      });
    }
  }

  public async cleanup(): Promise<void> {
    try {
      await questdbService.shutdown();
      await redisStreamsService.shutdown();
      await questdbConnection.disconnect();
      await redisConnection.disconnect();
      logger.info('🧹 Cleanup completed');
    } catch (error) {
      logger.error('❌ Cleanup failed:', error);
    }
  }
}

// Main execution
async function main() {
  const tester = new RealInfrastructureTester();
  
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
  const tester = new RealInfrastructureTester();
  await tester.cleanup();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  logger.info('🛑 Received SIGTERM, cleaning up...');
  const tester = new RealInfrastructureTester();
  await tester.cleanup();
  process.exit(0);
});

// Run tests if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { RealInfrastructureTester };
