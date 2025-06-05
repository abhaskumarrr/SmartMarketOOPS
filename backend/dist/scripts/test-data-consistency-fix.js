"use strict";
/**
 * Simple test script to verify data consistency fixes
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.testDataConsistencyFix = testDataConsistencyFix;
const logger_1 = require("../utils/logger");
const marketDataProvider_1 = require("../services/marketDataProvider");
const tradingEnvironment_1 = require("../config/tradingEnvironment");
async function testDataConsistencyFix() {
    try {
        logger_1.logger.info('🔍 Testing data consistency fixes...');
        // 1. Check environment configuration
        logger_1.logger.info('📋 Environment Configuration:');
        logger_1.logger.info(`  Mode: ${tradingEnvironment_1.environmentConfig.mode}`);
        logger_1.logger.info(`  Data Source: ${tradingEnvironment_1.environmentConfig.dataSource}`);
        logger_1.logger.info(`  Allow Mock Data: ${tradingEnvironment_1.environmentConfig.allowMockData}`);
        logger_1.logger.info(`  Enforce Data Consistency: ${tradingEnvironment_1.environmentConfig.enforceDataConsistency}`);
        logger_1.logger.info(`  Delta Testnet: ${tradingEnvironment_1.environmentConfig.deltaExchange.testnet}`);
        // 2. Check market data service provider
        logger_1.logger.info('📊 Market Data Service:');
        const providerInfo = marketDataProvider_1.marketDataService.getCurrentProviderInfo();
        logger_1.logger.info(`  Current Provider: ${providerInfo.name}`);
        logger_1.logger.info(`  Is Live: ${providerInfo.isLive}`);
        logger_1.logger.info(`  Is Mock: ${providerInfo.isMock}`);
        // 3. Test data source validation
        logger_1.logger.info('🔒 Testing data source validation...');
        try {
            // This should work - setting to a live provider
            marketDataProvider_1.marketDataService.setDefaultProvider('delta-exchange');
            logger_1.logger.info('✅ Successfully set Delta Exchange as provider');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to set Delta Exchange provider:', error);
        }
        // 4. Test mock data prevention (if in production mode)
        if (tradingEnvironment_1.environmentConfig.mode === 'production' || !tradingEnvironment_1.environmentConfig.allowMockData) {
            logger_1.logger.info('🚫 Testing mock data prevention...');
            try {
                marketDataProvider_1.marketDataService.setDefaultProvider('enhanced-mock');
                logger_1.logger.error('❌ CRITICAL: Mock data was allowed in production mode!');
            }
            catch (error) {
                logger_1.logger.info('✅ Mock data correctly prevented:', error instanceof Error ? error.message : error);
            }
        }
        // 5. Test live data mode enforcement
        logger_1.logger.info('🔄 Testing live data mode enforcement...');
        marketDataProvider_1.marketDataService.enforceLiveDataMode();
        const finalProviderInfo = marketDataProvider_1.marketDataService.getCurrentProviderInfo();
        if (finalProviderInfo.isLive && !finalProviderInfo.isMock) {
            logger_1.logger.info('✅ Live data mode successfully enforced');
        }
        else {
            logger_1.logger.error('❌ Live data mode enforcement failed');
        }
        // 6. Test data fetching consistency
        logger_1.logger.info('📈 Testing data fetching...');
        try {
            const testRequest = {
                symbol: 'BTCUSD',
                timeframe: '1h',
                startDate: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
                endDate: new Date(),
            };
            const response = await marketDataProvider_1.marketDataService.fetchHistoricalData(testRequest);
            if (response.count > 0) {
                logger_1.logger.info(`✅ Successfully fetched ${response.count} data points from ${finalProviderInfo.name}`);
                logger_1.logger.info(`   Latest price: $${response.data[response.data.length - 1]?.close || 'N/A'}`);
            }
            else {
                logger_1.logger.warn('⚠️ No data points returned');
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Data fetching failed:', error);
        }
        // 7. Summary
        logger_1.logger.info('📋 Data Consistency Test Summary:');
        logger_1.logger.info(`  ✅ Environment: ${tradingEnvironment_1.environmentConfig.mode}`);
        logger_1.logger.info(`  ✅ Provider: ${finalProviderInfo.name}`);
        logger_1.logger.info(`  ✅ Live Data: ${finalProviderInfo.isLive}`);
        logger_1.logger.info(`  ✅ Mock Prevention: ${!finalProviderInfo.isMock}`);
        if (finalProviderInfo.isLive && !finalProviderInfo.isMock) {
            logger_1.logger.info('🎉 Data consistency fixes are working correctly!');
            logger_1.logger.info('🔒 Trading system will use consistent live data sources');
        }
        else {
            logger_1.logger.error('❌ Data consistency issues detected - review configuration');
        }
    }
    catch (error) {
        logger_1.logger.error('❌ Data consistency test failed:', error);
        throw error;
    }
}
// Run the test
if (require.main === module) {
    testDataConsistencyFix()
        .then(() => {
        logger_1.logger.info('✅ Data consistency test completed');
        process.exit(0);
    })
        .catch(error => {
        logger_1.logger.error('❌ Data consistency test failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=test-data-consistency-fix.js.map