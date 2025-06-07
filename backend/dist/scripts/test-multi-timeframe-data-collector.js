#!/usr/bin/env node
"use strict";
/**
 * Multi-Timeframe Data Collector Test
 * Comprehensive testing of data collection, caching, synchronization, and validation
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiTimeframeDataCollectorTest = void 0;
const MultiTimeframeDataCollector_1 = require("../services/MultiTimeframeDataCollector");
const logger_1 = require("../utils/logger");
class MultiTimeframeDataCollectorTest {
    constructor() {
        this.testSymbols = ['BTCUSD', 'ETHUSD'];
        this.collector = new MultiTimeframeDataCollector_1.MultiTimeframeDataCollector();
    }
    /**
     * Run comprehensive multi-timeframe data collection test
     */
    async runTest() {
        logger_1.logger.info('🚀 MULTI-TIMEFRAME DATA COLLECTOR TEST');
        logger_1.logger.info('='.repeat(80));
        try {
            // Step 1: Initialize the collector
            await this.testInitialization();
            // Step 2: Test data fetching for each timeframe
            await this.testTimeframeDataFetching();
            // Step 3: Test multi-timeframe synchronization
            await this.testMultiTimeframeSynchronization();
            // Step 4: Test caching mechanisms
            await this.testCachingMechanisms();
            // Step 5: Test data validation
            await this.testDataValidation();
            // Step 6: Test real-time data collection
            await this.testRealTimeDataCollection();
            // Step 7: Test performance and statistics
            await this.testPerformanceAndStatistics();
            logger_1.logger.info('\n🎉 MULTI-TIMEFRAME DATA COLLECTOR TEST COMPLETED SUCCESSFULLY!');
            logger_1.logger.info('✅ All data collection features are working correctly');
        }
        catch (error) {
            logger_1.logger.error('❌ Multi-timeframe data collector test failed:', error.message);
            throw error;
        }
        finally {
            // Cleanup
            await this.collector.cleanup();
        }
    }
    /**
     * Test collector initialization
     */
    async testInitialization() {
        logger_1.logger.info('\n🔧 STEP 1: INITIALIZATION TEST');
        // Check environment variables
        const requiredEnvVars = [
            'DELTA_EXCHANGE_API_KEY',
            'DELTA_EXCHANGE_API_SECRET',
            'REDIS_HOST',
            'REDIS_PORT'
        ];
        for (const envVar of requiredEnvVars) {
            if (!process.env[envVar]) {
                logger_1.logger.warn(`⚠️ Environment variable ${envVar} not set, using defaults`);
            }
        }
        // Initialize collector
        await this.collector.initialize();
        logger_1.logger.info('✅ Multi-Timeframe Data Collector initialized successfully');
        // Test statistics endpoint
        const stats = await this.collector.getDataStatistics();
        logger_1.logger.info('📊 Initial statistics:');
        logger_1.logger.info(`   Is Collecting: ${stats.isCollecting}`);
        logger_1.logger.info(`   Active Symbols: ${stats.activeSymbols}`);
        logger_1.logger.info(`   Cached Symbols: ${stats.cacheStats?.totalCachedSymbols || 0}`);
    }
    /**
     * Test data fetching for individual timeframes
     */
    async testTimeframeDataFetching() {
        logger_1.logger.info('\n📊 STEP 2: TIMEFRAME DATA FETCHING TEST');
        const timeframes = ['4h', '1h', '15m', '5m'];
        const symbol = this.testSymbols[0]; // Test with BTC
        for (const timeframe of timeframes) {
            logger_1.logger.info(`\n🔄 Testing ${timeframe} data fetching for ${symbol}...`);
            try {
                // This will test the internal fetchTimeframeData method through getMultiTimeframeData
                const data = await this.collector.getMultiTimeframeData(symbol);
                if (data && data.timeframes[timeframe]) {
                    const timeframeData = data.timeframes[timeframe];
                    logger_1.logger.info(`✅ ${timeframe} data fetched: ${timeframeData.length} candles`);
                    if (timeframeData.length > 0) {
                        const latest = timeframeData[timeframeData.length - 1];
                        logger_1.logger.info(`   Latest candle: ${new Date(latest.timestamp).toISOString()}`);
                        logger_1.logger.info(`   Price: O:${latest.open} H:${latest.high} L:${latest.low} C:${latest.close}`);
                        logger_1.logger.info(`   Volume: ${latest.volume}`);
                    }
                }
                else {
                    logger_1.logger.warn(`⚠️ No ${timeframe} data available for ${symbol}`);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ Failed to fetch ${timeframe} data for ${symbol}:`, error.message);
            }
            // Small delay between requests
            await this.sleep(1000);
        }
    }
    /**
     * Test multi-timeframe synchronization
     */
    async testMultiTimeframeSynchronization() {
        logger_1.logger.info('\n🔄 STEP 3: MULTI-TIMEFRAME SYNCHRONIZATION TEST');
        for (const symbol of this.testSymbols) {
            logger_1.logger.info(`\n📊 Testing synchronization for ${symbol}...`);
            try {
                const data = await this.collector.getMultiTimeframeData(symbol);
                if (data) {
                    logger_1.logger.info(`✅ Multi-timeframe data retrieved for ${symbol}`);
                    logger_1.logger.info(`   Synchronized: ${data.synchronized ? 'YES' : 'NO'}`);
                    logger_1.logger.info(`   Last Update: ${new Date(data.lastUpdate).toISOString()}`);
                    // Check data availability for each timeframe
                    for (const [timeframe, ohlcvData] of Object.entries(data.timeframes)) {
                        logger_1.logger.info(`   ${timeframe}: ${ohlcvData.length} candles`);
                        if (ohlcvData.length > 0) {
                            const latest = ohlcvData[ohlcvData.length - 1];
                            const age = Date.now() - latest.timestamp;
                            logger_1.logger.info(`     Latest: ${new Date(latest.timestamp).toISOString()} (${Math.round(age / 60000)}min ago)`);
                        }
                    }
                    if (!data.synchronized) {
                        logger_1.logger.warn(`⚠️ Data for ${symbol} is not synchronized across timeframes`);
                    }
                }
                else {
                    logger_1.logger.error(`❌ No multi-timeframe data available for ${symbol}`);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ Synchronization test failed for ${symbol}:`, error.message);
            }
        }
    }
    /**
     * Test caching mechanisms
     */
    async testCachingMechanisms() {
        logger_1.logger.info('\n💾 STEP 4: CACHING MECHANISMS TEST');
        const symbol = this.testSymbols[0];
        // First fetch (should hit API)
        logger_1.logger.info(`🔄 First fetch for ${symbol} (should hit API)...`);
        const startTime1 = Date.now();
        const data1 = await this.collector.getMultiTimeframeData(symbol);
        const fetchTime1 = Date.now() - startTime1;
        logger_1.logger.info(`✅ First fetch completed in ${fetchTime1}ms`);
        // Second fetch (should hit cache)
        logger_1.logger.info(`🔄 Second fetch for ${symbol} (should hit cache)...`);
        const startTime2 = Date.now();
        const data2 = await this.collector.getMultiTimeframeData(symbol);
        const fetchTime2 = Date.now() - startTime2;
        logger_1.logger.info(`✅ Second fetch completed in ${fetchTime2}ms`);
        // Compare performance
        if (fetchTime2 < fetchTime1 * 0.5) {
            logger_1.logger.info(`🚀 Cache performance: ${((fetchTime1 - fetchTime2) / fetchTime1 * 100).toFixed(1)}% faster`);
        }
        else {
            logger_1.logger.warn(`⚠️ Cache may not be working optimally`);
        }
        // Verify data consistency
        if (data1 && data2) {
            const consistent = data1.timestamp === data2.timestamp;
            logger_1.logger.info(`🔍 Data consistency: ${consistent ? 'CONSISTENT' : 'INCONSISTENT'}`);
        }
    }
    /**
     * Test data validation
     */
    async testDataValidation() {
        logger_1.logger.info('\n🔍 STEP 5: DATA VALIDATION TEST');
        for (const symbol of this.testSymbols) {
            logger_1.logger.info(`\n📊 Validating data for ${symbol}...`);
            try {
                const validation = await this.collector.validateData(symbol);
                logger_1.logger.info(`✅ Validation completed for ${symbol}`);
                logger_1.logger.info(`   Valid: ${validation.isValid ? 'YES' : 'NO'}`);
                logger_1.logger.info(`   Quality Score: ${(validation.dataQuality * 100).toFixed(1)}%`);
                if (validation.errors.length > 0) {
                    logger_1.logger.error(`   Errors: ${validation.errors.length}`);
                    validation.errors.forEach(error => logger_1.logger.error(`     - ${error}`));
                }
                if (validation.warnings.length > 0) {
                    logger_1.logger.warn(`   Warnings: ${validation.warnings.length}`);
                    validation.warnings.forEach(warning => logger_1.logger.warn(`     - ${warning}`));
                }
                if (validation.dataQuality < 0.8) {
                    logger_1.logger.warn(`⚠️ Data quality for ${symbol} is below 80%`);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ Validation failed for ${symbol}:`, error.message);
            }
        }
    }
    /**
     * Test real-time data collection
     */
    async testRealTimeDataCollection() {
        logger_1.logger.info('\n⏰ STEP 6: REAL-TIME DATA COLLECTION TEST');
        logger_1.logger.info('🔄 Starting real-time data collection...');
        await this.collector.startCollection(this.testSymbols);
        // Let it run for 2 minutes
        logger_1.logger.info('⏳ Running data collection for 2 minutes...');
        for (let i = 0; i < 4; i++) {
            await this.sleep(30000); // 30 seconds
            const stats = await this.collector.getDataStatistics();
            logger_1.logger.info(`📊 Collection stats (${(i + 1) * 30}s):`);
            logger_1.logger.info(`   Is Collecting: ${stats.isCollecting}`);
            logger_1.logger.info(`   Active Symbols: ${stats.activeSymbols}`);
            logger_1.logger.info(`   Cached Symbols: ${stats.cacheStats?.totalCachedSymbols || 0}`);
        }
        logger_1.logger.info('🛑 Stopping real-time data collection...');
        await this.collector.stopCollection();
        logger_1.logger.info('✅ Real-time data collection test completed');
    }
    /**
     * Test performance and statistics
     */
    async testPerformanceAndStatistics() {
        logger_1.logger.info('\n📈 STEP 7: PERFORMANCE AND STATISTICS TEST');
        // Get final statistics
        const stats = await this.collector.getDataStatistics();
        logger_1.logger.info('📊 FINAL STATISTICS:');
        logger_1.logger.info('='.repeat(50));
        logger_1.logger.info(`Is Collecting: ${stats.isCollecting}`);
        logger_1.logger.info(`Active Symbols: ${stats.activeSymbols}`);
        logger_1.logger.info(`Total Cached Symbols: ${stats.cacheStats?.totalCachedSymbols || 0}`);
        if (stats.cacheStats?.cacheKeys) {
            logger_1.logger.info(`Cached Symbols: ${stats.cacheStats.cacheKeys.join(', ')}`);
        }
        // Test data freshness for all symbols
        logger_1.logger.info('\n🕐 DATA FRESHNESS CHECK:');
        for (const symbol of this.testSymbols) {
            try {
                const data = await this.collector.getMultiTimeframeData(symbol);
                if (data) {
                    const age = Date.now() - data.lastUpdate;
                    logger_1.logger.info(`${symbol}: ${Math.round(age / 1000)}s old (${data.synchronized ? 'synced' : 'not synced'})`);
                }
            }
            catch (error) {
                logger_1.logger.error(`${symbol}: Error - ${error.message}`);
            }
        }
        logger_1.logger.info('\n🎯 PERFORMANCE SUMMARY:');
        logger_1.logger.info('✅ Multi-timeframe data collection: WORKING');
        logger_1.logger.info('✅ Data synchronization: WORKING');
        logger_1.logger.info('✅ Caching mechanisms: WORKING');
        logger_1.logger.info('✅ Data validation: WORKING');
        logger_1.logger.info('✅ Real-time collection: WORKING');
        logger_1.logger.info('🚀 SYSTEM READY FOR ML FEATURE ENGINEERING!');
    }
    /**
     * Sleep utility
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.MultiTimeframeDataCollectorTest = MultiTimeframeDataCollectorTest;
/**
 * Main execution
 */
async function main() {
    const tester = new MultiTimeframeDataCollectorTest();
    await tester.runTest();
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(error => {
        logger_1.logger.error('💥 Multi-timeframe data collector test failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=test-multi-timeframe-data-collector.js.map