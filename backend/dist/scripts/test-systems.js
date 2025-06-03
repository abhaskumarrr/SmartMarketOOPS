#!/usr/bin/env node
"use strict";
/**
 * Test Systems Script
 * Comprehensive testing for QuestDB migration and Redis Streams
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SystemTester = void 0;
const mockQuestdbService_1 = require("../services/mockQuestdbService");
const mockRedisStreamsService_1 = require("../services/mockRedisStreamsService");
const redis_1 = require("../config/redis");
const logger_1 = require("../utils/logger");
const events_1 = require("../types/events");
class SystemTester {
    constructor() {
        this.results = [];
    }
    async runAllTests() {
        logger_1.logger.info('🧪 Starting comprehensive system tests...');
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
    async testQuestDBMock() {
        const testName = 'QuestDB Mock Service';
        const startTime = Date.now();
        try {
            logger_1.logger.info('📊 Testing QuestDB Mock Service...');
            // Initialize service
            await mockQuestdbService_1.mockQuestdbService.initialize();
            // Test metric insertion
            await mockQuestdbService_1.mockQuestdbService.insertMetric({
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
            await mockQuestdbService_1.mockQuestdbService.insertMetrics(metrics);
            // Test trading signal insertion
            await mockQuestdbService_1.mockQuestdbService.insertTradingSignal({
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
            const queryResult = await mockQuestdbService_1.mockQuestdbService.executeQuery('SELECT count() FROM metrics');
            // Get statistics
            const stats = mockQuestdbService_1.mockQuestdbService.getStats();
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
    async testRedisStreamsMock() {
        const testName = 'Redis Streams Mock Service';
        const startTime = Date.now();
        try {
            logger_1.logger.info('📡 Testing Redis Streams Mock Service...');
            // Initialize service
            await mockRedisStreamsService_1.mockRedisStreamsService.initialize();
            // Test health check
            const healthCheck = await mockRedisStreamsService_1.mockRedisStreamsService.healthCheck();
            if (!healthCheck) {
                throw new Error('Redis Streams Mock health check failed');
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
            const messageId = await mockRedisStreamsService_1.mockRedisStreamsService.publishEvent('market-data-stream', testEvent);
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
            const batchMessageIds = await mockRedisStreamsService_1.mockRedisStreamsService.publishEvents('market-data-stream', batchEvents);
            // Test reading events
            const readEvents = await mockRedisStreamsService_1.mockRedisStreamsService.readEvents('market-data-stream', {
                groupName: 'test-group',
                consumerName: 'test-consumer',
                count: 5,
            });
            // Get service stats
            const stats = await mockRedisStreamsService_1.mockRedisStreamsService.getStats();
            const mockStats = mockRedisStreamsService_1.mockRedisStreamsService.getMockStats();
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
    async testEventDrivenSystem() {
        const testName = 'Event-Driven Trading System';
        const startTime = Date.now();
        try {
            logger_1.logger.info('🚀 Testing Event-Driven Trading System...');
            // Test system health before starting
            const initialHealth = eventDrivenTradingSystem.getHealthStatus();
            // Test event publishing without starting the full system
            const marketDataMessageId = await eventDrivenTradingSystem.publishMarketDataEvent('BTCUSD', 45000, 1.5, 'test-exchange');
            const signalMessageId = await eventDrivenTradingSystem.publishTradingSignalEvent({
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
    async testIntegration() {
        const testName = 'System Integration';
        const startTime = Date.now();
        try {
            logger_1.logger.info('🔗 Testing System Integration...');
            // Test data flow: Market Data -> QuestDB
            const marketData = {
                timestamp: new Date(),
                name: 'btc_price',
                value: 45000,
                tags: { symbol: 'BTCUSD', exchange: 'test' },
            };
            await mockQuestdbService_1.mockQuestdbService.insertMetric(marketData);
            // Test event flow: Market Data Event -> Redis Streams
            const marketEvent = {
                id: (0, events_1.createEventId)(),
                type: 'MARKET_DATA_RECEIVED',
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
            const questdbStats = mockQuestdbService_1.mockQuestdbService.getStats();
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
        logger_1.logger.info('\n📊 Test Results Summary:');
        logger_1.logger.info('='.repeat(80));
        let totalTests = this.results.length;
        let passedTests = this.results.filter(r => r.success).length;
        let failedTests = totalTests - passedTests;
        this.results.forEach(result => {
            const status = result.success ? '✅ PASS' : '❌ FAIL';
            const duration = `${result.duration}ms`;
            logger_1.logger.info(`${status} ${result.name.padEnd(30)} (${duration})`);
            if (!result.success && result.error) {
                logger_1.logger.error(`     Error: ${result.error}`);
            }
            if (result.details) {
                logger_1.logger.info(`     Details: ${JSON.stringify(result.details, null, 2)}`);
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
        }
    }
    async cleanup() {
        try {
            await mockQuestdbService_1.mockQuestdbService.shutdown();
            await redisStreamsService.shutdown();
            await redis_1.redisConnection.disconnect();
            logger_1.logger.info('🧹 Cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Cleanup failed:', error);
        }
    }
}
exports.SystemTester = SystemTester;
// Main execution
async function main() {
    const tester = new SystemTester();
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
    const tester = new SystemTester();
    await tester.cleanup();
    process.exit(0);
});
process.on('SIGTERM', async () => {
    logger_1.logger.info('🛑 Received SIGTERM, cleaning up...');
    const tester = new SystemTester();
    await tester.cleanup();
    process.exit(0);
});
// Run tests if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=test-systems.js.map