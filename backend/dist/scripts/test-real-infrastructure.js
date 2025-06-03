#!/usr/bin/env node
"use strict";
/**
 * Real Infrastructure Test Script
 * Comprehensive testing with actual Redis and QuestDB instances
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RealInfrastructureTester = void 0;
const questdb_1 = require("../config/questdb");
const questdbService_1 = require("../services/questdbService");
const redis_1 = require("../config/redis");
const redisStreamsService_1 = require("../services/redisStreamsService");
const eventDrivenTradingSystem_1 = require("../services/eventDrivenTradingSystem");
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
class RealInfrastructureTester {
    constructor() {
        this.results = [];
    }
    async runAllTests() {
        logger_1.logger.info('🧪 Starting comprehensive real infrastructure tests...');
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
    async testQuestDBConnection() {
        const testName = 'QuestDB Connection';
        const startTime = Date.now();
        try {
            logger_1.logger.info('📊 Testing QuestDB Connection...');
            // Test connection
            await questdb_1.questdbConnection.connect();
            const healthCheck = await questdb_1.questdbConnection.healthCheck();
            if (!healthCheck) {
                throw new Error('QuestDB health check failed');
            }
            const stats = questdb_1.questdbConnection.getStats();
            this.results.push({
                name: testName,
                success: true,
                duration: Date.now() - startTime,
                details: {
                    isConnected: stats.isConnected,
                    config: stats.clientOptions,
                },
            });
            logger_1.logger.info(`✅ ${testName} passed`);
        }
        catch (error) {
            this.results.push({
                name: testName,
                success: false,
                duration: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            });
            logger_1.logger.error(`❌ ${testName} failed:`, error);
        }
    }
    async testQuestDBOperations() {
        const testName = 'QuestDB Operations';
        const startTime = Date.now();
        try {
            logger_1.logger.info('📊 Testing QuestDB Operations...');
            // Initialize service
            await questdbService_1.questdbService.initialize();
            // Test metric insertion
            await questdbService_1.questdbService.insertMetric({
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
            await questdbService_1.questdbService.insertMetrics(metrics);
            // Test trading signal insertion
            await questdbService_1.questdbService.insertTradingSignal({
                timestamp: new Date(),
                id: (0, events_1.createEventId)(),
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
            const queryResult = await questdbService_1.questdbService.executeQuery('SELECT count() FROM metrics');
            const tableStats = await questdbService_1.questdbService.getTableStats('metrics');
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
            logger_1.logger.info(`✅ ${testName} passed`);
        }
        catch (error) {
            this.results.push({
                name: testName,
                success: false,
                duration: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            });
            logger_1.logger.error(`❌ ${testName} failed:`, error);
        }
    }
    async testQuestDBPerformance() {
        const testName = 'QuestDB Performance';
        const startTime = Date.now();
        try {
            logger_1.logger.info('📊 Testing QuestDB Performance...');
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
                    await questdbService_1.questdbService.insertMetrics(metrics);
                    totalInserted += batchSize;
                }
                catch (error) {
                    totalErrors++;
                    logger_1.logger.error(`Batch ${batch} failed:`, error);
                }
                const batchDuration = Date.now() - batchStartTime;
                logger_1.logger.info(`Batch ${batch + 1}/${batches}: ${batchSize} records in ${batchDuration}ms (${(batchSize / (batchDuration / 1000)).toFixed(0)} records/sec)`);
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
            logger_1.logger.info(`✅ ${testName} completed: ${totalInserted} records, ${throughput.toFixed(0)} records/sec`);
        }
        catch (error) {
            this.results.push({
                name: testName,
                success: false,
                duration: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            });
            logger_1.logger.error(`❌ ${testName} failed:`, error);
        }
    }
    async testRedisConnection() {
        const testName = 'Redis Connection';
        const startTime = Date.now();
        try {
            logger_1.logger.info('🔌 Testing Redis Connection...');
            // Validate environment
            const envValidation = (0, redis_1.validateRedisEnvironment)();
            if (!envValidation.valid) {
                throw new Error(`Environment validation failed: ${envValidation.errors.join(', ')}`);
            }
            // Test connection
            await redis_1.redisConnection.connect();
            const healthCheck = await redis_1.redisConnection.healthCheck();
            if (!healthCheck) {
                throw new Error('Redis health check failed');
            }
            const stats = redis_1.redisConnection.getStats();
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
            logger_1.logger.info(`✅ ${testName} passed`);
        }
        catch (error) {
            this.results.push({
                name: testName,
                success: false,
                duration: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            });
            logger_1.logger.error(`❌ ${testName} failed:`, error);
        }
    }
    async testRedisStreams() {
        const testName = 'Redis Streams Operations';
        const startTime = Date.now();
        try {
            logger_1.logger.info('📡 Testing Redis Streams Operations...');
            // Initialize service
            await redisStreamsService_1.redisStreamsService.initialize();
            // Test health check
            const healthCheck = await redisStreamsService_1.redisStreamsService.healthCheck();
            if (!healthCheck) {
                throw new Error('Redis Streams health check failed');
            }
            // Test event publishing
            const testEvent = {
                id: (0, events_1.createEventId)(),
                type: 'MARKET_DATA_RECEIVED',
                timestamp: Date.now(),
                version: '1.0',
                source: 'test-system',
                correlationId: (0, events_1.createCorrelationId)(),
                data: {
                    symbol: 'BTCUSD',
                    exchange: 'test-exchange',
                    price: 45000,
                    volume: 1.5,
                    timestamp: Date.now(),
                },
            };
            const messageId = await redisStreamsService_1.redisStreamsService.publishEvent('market-data-stream', testEvent);
            // Test batch publishing
            const batchEvents = Array.from({ length: 10 }, (_, i) => ({
                id: (0, events_1.createEventId)(),
                type: 'MARKET_DATA_RECEIVED',
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
            const batchMessageIds = await redisStreamsService_1.redisStreamsService.publishEvents('market-data-stream', batchEvents);
            // Test reading events
            const readEvents = await redisStreamsService_1.redisStreamsService.readEvents('market-data-stream', {
                groupName: 'test-group',
                consumerName: 'test-consumer',
                count: 5,
            });
            // Get service stats
            const stats = await redisStreamsService_1.redisStreamsService.getStats();
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
            logger_1.logger.info(`✅ ${testName} passed`);
        }
        catch (error) {
            this.results.push({
                name: testName,
                success: false,
                duration: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            });
            logger_1.logger.error(`❌ ${testName} failed:`, error);
        }
    }
    async testRedisStreamsPerformance() {
        const testName = 'Redis Streams Performance';
        const startTime = Date.now();
        try {
            logger_1.logger.info('📡 Testing Redis Streams Performance...');
            const eventCount = 1000;
            const events = Array.from({ length: eventCount }, (_, i) => ({
                id: (0, events_1.createEventId)(),
                type: 'MARKET_DATA_RECEIVED',
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
            const messageIds = await redisStreamsService_1.redisStreamsService.publishEvents('market-data-stream', events);
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
            logger_1.logger.info(`✅ ${testName} completed: ${eventCount} events, ${publishThroughput.toFixed(0)} events/sec`);
        }
        catch (error) {
            this.results.push({
                name: testName,
                success: false,
                duration: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            });
            logger_1.logger.error(`❌ ${testName} failed:`, error);
        }
    }
    async testEndToEndIntegration() {
        const testName = 'End-to-End Integration';
        const startTime = Date.now();
        try {
            logger_1.logger.info('🔗 Testing End-to-End Integration...');
            // Test data flow: Market Data Event -> Redis Streams -> QuestDB
            const marketEvent = {
                id: (0, events_1.createEventId)(),
                type: 'MARKET_DATA_RECEIVED',
                timestamp: Date.now(),
                version: '1.0',
                source: 'integration-test',
                correlationId: (0, events_1.createCorrelationId)(),
                data: {
                    symbol: 'BTCUSD',
                    exchange: 'test-exchange',
                    price: 45000,
                    volume: 1.5,
                    timestamp: Date.now(),
                },
            };
            // Publish to Redis Streams
            const eventMessageId = await redisStreamsService_1.redisStreamsService.publishEvent('market-data-stream', marketEvent);
            // Store corresponding metric in QuestDB
            await questdbService_1.questdbService.insertMetric({
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
            const questdbStats = await questdbService_1.questdbService.getTableStats('metrics');
            const redisStats = await redisStreamsService_1.redisStreamsService.getStats();
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
            logger_1.logger.info(`✅ ${testName} passed`);
        }
        catch (error) {
            this.results.push({
                name: testName,
                success: false,
                duration: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            });
            logger_1.logger.error(`❌ ${testName} failed:`, error);
        }
    }
    async testEventProcessingPipeline() {
        const testName = 'Event Processing Pipeline';
        const startTime = Date.now();
        try {
            logger_1.logger.info('🚀 Testing Event Processing Pipeline...');
            // Test event publishing through the trading system
            const marketDataMessageId = await eventDrivenTradingSystem_1.eventDrivenTradingSystem.publishMarketDataEvent('BTCUSD', 45000, 1.5, 'test-exchange');
            const signalMessageId = await eventDrivenTradingSystem_1.eventDrivenTradingSystem.publishTradingSignalEvent({
                signalId: (0, events_1.createEventId)(),
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
            const systemStats = eventDrivenTradingSystem_1.eventDrivenTradingSystem.getStats();
            const systemConfig = eventDrivenTradingSystem_1.eventDrivenTradingSystem.getConfig();
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
            logger_1.logger.info(`✅ ${testName} passed`);
        }
        catch (error) {
            this.results.push({
                name: testName,
                success: false,
                duration: Date.now() - startTime,
                error: error instanceof Error ? error.message : 'Unknown error',
            });
            logger_1.logger.error(`❌ ${testName} failed:`, error);
        }
    }
    displayResults() {
        logger_1.logger.info('\n📊 Real Infrastructure Test Results Summary:');
        logger_1.logger.info('='.repeat(80));
        let totalTests = this.results.length;
        let passedTests = this.results.filter(r => r.success).length;
        let failedTests = totalTests - passedTests;
        this.results.forEach(result => {
            const status = result.success ? '✅ PASS' : '❌ FAIL';
            const duration = `${result.duration}ms`;
            logger_1.logger.info(`${status} ${result.name.padEnd(30)} (${duration})`);
            if (result.metrics) {
                if (result.metrics.throughput) {
                    logger_1.logger.info(`     Throughput: ${result.metrics.throughput.toFixed(0)} ops/sec`);
                }
                if (result.metrics.latency) {
                    logger_1.logger.info(`     Latency: ${result.metrics.latency.toFixed(2)}ms`);
                }
                if (result.metrics.errorRate !== undefined) {
                    logger_1.logger.info(`     Error Rate: ${result.metrics.errorRate.toFixed(1)}%`);
                }
            }
            if (!result.success && result.error) {
                logger_1.logger.error(`     Error: ${result.error}`);
            }
        });
        logger_1.logger.info('='.repeat(80));
        logger_1.logger.info(`Total Tests: ${totalTests}`);
        logger_1.logger.info(`Passed: ${passedTests}`);
        logger_1.logger.info(`Failed: ${failedTests}`);
        logger_1.logger.info(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
        if (failedTests > 0) {
            logger_1.logger.error(`\n❌ ${failedTests} test(s) failed. Please check the errors above.`);
            process.exit(1);
        }
        else {
            logger_1.logger.info(`\n🎉 All tests passed successfully!`);
            logger_1.logger.info('\n📈 Performance Summary:');
            const performanceResults = this.results.filter(r => r.metrics);
            performanceResults.forEach(result => {
                if (result.metrics?.throughput) {
                    logger_1.logger.info(`  ${result.name}: ${result.metrics.throughput.toFixed(0)} ops/sec`);
                }
            });
        }
    }
    async cleanup() {
        try {
            await questdbService_1.questdbService.shutdown();
            await redisStreamsService_1.redisStreamsService.shutdown();
            await questdb_1.questdbConnection.disconnect();
            await redis_1.redisConnection.disconnect();
            logger_1.logger.info('🧹 Cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Cleanup failed:', error);
        }
    }
}
exports.RealInfrastructureTester = RealInfrastructureTester;
// Main execution
async function main() {
    const tester = new RealInfrastructureTester();
    try {
        await tester.runAllTests();
    }
    catch (error) {
        logger_1.logger.error('💥 Test execution failed:', error);
        process.exit(1);
    }
    finally {
        await tester.cleanup();
    }
}
// Handle graceful shutdown
process.on('SIGINT', async () => {
    logger_1.logger.info('🛑 Received SIGINT, cleaning up...');
    const tester = new RealInfrastructureTester();
    await tester.cleanup();
    process.exit(0);
});
process.on('SIGTERM', async () => {
    logger_1.logger.info('🛑 Received SIGTERM, cleaning up...');
    const tester = new RealInfrastructureTester();
    await tester.cleanup();
    process.exit(0);
});
// Run tests if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=test-real-infrastructure.js.map