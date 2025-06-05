/**
 * Simple test script to verify data consistency fixes
 */

import { logger } from '../utils/logger';
import { marketDataService } from '../services/marketDataProvider';
import { environmentConfig } from '../config/tradingEnvironment';

async function testDataConsistencyFix(): Promise<void> {
  try {
    logger.info('🔍 Testing data consistency fixes...');
    
    // 1. Check environment configuration
    logger.info('📋 Environment Configuration:');
    logger.info(`  Mode: ${environmentConfig.mode}`);
    logger.info(`  Data Source: ${environmentConfig.dataSource}`);
    logger.info(`  Allow Mock Data: ${environmentConfig.allowMockData}`);
    logger.info(`  Enforce Data Consistency: ${environmentConfig.enforceDataConsistency}`);
    logger.info(`  Delta Testnet: ${environmentConfig.deltaExchange.testnet}`);
    
    // 2. Check market data service provider
    logger.info('📊 Market Data Service:');
    const providerInfo = marketDataService.getCurrentProviderInfo();
    logger.info(`  Current Provider: ${providerInfo.name}`);
    logger.info(`  Is Live: ${providerInfo.isLive}`);
    logger.info(`  Is Mock: ${providerInfo.isMock}`);
    
    // 3. Test data source validation
    logger.info('🔒 Testing data source validation...');
    
    try {
      // This should work - setting to a live provider
      marketDataService.setDefaultProvider('delta-exchange');
      logger.info('✅ Successfully set Delta Exchange as provider');
    } catch (error) {
      logger.error('❌ Failed to set Delta Exchange provider:', error);
    }
    
    // 4. Test mock data prevention (if in production mode)
    if (environmentConfig.mode === 'production' || !environmentConfig.allowMockData) {
      logger.info('🚫 Testing mock data prevention...');
      
      try {
        marketDataService.setDefaultProvider('enhanced-mock');
        logger.error('❌ CRITICAL: Mock data was allowed in production mode!');
      } catch (error) {
        logger.info('✅ Mock data correctly prevented:', error instanceof Error ? error.message : error);
      }
    }
    
    // 5. Test live data mode enforcement
    logger.info('🔄 Testing live data mode enforcement...');
    marketDataService.enforceLiveDataMode();
    
    const finalProviderInfo = marketDataService.getCurrentProviderInfo();
    if (finalProviderInfo.isLive && !finalProviderInfo.isMock) {
      logger.info('✅ Live data mode successfully enforced');
    } else {
      logger.error('❌ Live data mode enforcement failed');
    }
    
    // 6. Test data fetching consistency
    logger.info('📈 Testing data fetching...');
    
    try {
      const testRequest = {
        symbol: 'BTCUSD',
        timeframe: '1h',
        startDate: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
        endDate: new Date(),
      };
      
      const response = await marketDataService.fetchHistoricalData(testRequest);
      
      if (response.count > 0) {
        logger.info(`✅ Successfully fetched ${response.count} data points from ${finalProviderInfo.name}`);
        logger.info(`   Latest price: $${response.data[response.data.length - 1]?.close || 'N/A'}`);
      } else {
        logger.warn('⚠️ No data points returned');
      }
      
    } catch (error) {
      logger.error('❌ Data fetching failed:', error);
    }
    
    // 7. Summary
    logger.info('📋 Data Consistency Test Summary:');
    logger.info(`  ✅ Environment: ${environmentConfig.mode}`);
    logger.info(`  ✅ Provider: ${finalProviderInfo.name}`);
    logger.info(`  ✅ Live Data: ${finalProviderInfo.isLive}`);
    logger.info(`  ✅ Mock Prevention: ${!finalProviderInfo.isMock}`);
    
    if (finalProviderInfo.isLive && !finalProviderInfo.isMock) {
      logger.info('🎉 Data consistency fixes are working correctly!');
      logger.info('🔒 Trading system will use consistent live data sources');
    } else {
      logger.error('❌ Data consistency issues detected - review configuration');
    }
    
  } catch (error) {
    logger.error('❌ Data consistency test failed:', error);
    throw error;
  }
}

// Run the test
if (require.main === module) {
  testDataConsistencyFix()
    .then(() => {
      logger.info('✅ Data consistency test completed');
      process.exit(0);
    })
    .catch(error => {
      logger.error('❌ Data consistency test failed:', error);
      process.exit(1);
    });
}

export { testDataConsistencyFix };
