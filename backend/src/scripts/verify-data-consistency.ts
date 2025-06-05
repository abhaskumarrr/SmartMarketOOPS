/**
 * Data Consistency Verification Script
 * Verifies that all trading operations use consistent live data sources
 */

import { logger } from '../utils/logger';
import { marketDataService } from '../services/marketDataProvider';
import { DeltaExchangeUnified } from '../services/DeltaExchangeUnified';
import { DeltaTradingBot, BotConfig } from '../services/DeltaTradingBot';

interface DataConsistencyReport {
  timestamp: string;
  marketDataProvider: {
    name: string;
    isLive: boolean;
    isMock: boolean;
  };
  deltaExchangeService: {
    isReady: boolean;
    testnetMode: boolean;
  };
  consistencyChecks: {
    dataSourceUnified: boolean;
    noMockDataInProduction: boolean;
    deltaServiceReady: boolean;
    priceDataConsistent: boolean;
  };
  testResults: {
    marketDataFetch: boolean;
    deltaExchangeDataFetch: boolean;
    priceComparison: {
      marketDataPrice: number;
      deltaExchangePrice: number;
      difference: number;
      withinTolerance: boolean;
    };
  };
  recommendations: string[];
  overallStatus: 'SAFE' | 'WARNING' | 'CRITICAL';
}

async function verifyDataConsistency(): Promise<DataConsistencyReport> {
  logger.info('🔍 Starting data consistency verification...');
  
  const report: DataConsistencyReport = {
    timestamp: new Date().toISOString(),
    marketDataProvider: {
      name: '',
      isLive: false,
      isMock: false,
    },
    deltaExchangeService: {
      isReady: false,
      testnetMode: false,
    },
    consistencyChecks: {
      dataSourceUnified: false,
      noMockDataInProduction: false,
      deltaServiceReady: false,
      priceDataConsistent: false,
    },
    testResults: {
      marketDataFetch: false,
      deltaExchangeDataFetch: false,
      priceComparison: {
        marketDataPrice: 0,
        deltaExchangePrice: 0,
        difference: 0,
        withinTolerance: false,
      },
    },
    recommendations: [],
    overallStatus: 'CRITICAL',
  };

  try {
    // 1. Check MarketDataService provider
    logger.info('📊 Checking MarketDataService provider...');
    const providerInfo = marketDataService.getCurrentProviderInfo();
    report.marketDataProvider = providerInfo;
    
    logger.info(`Current provider: ${providerInfo.name}, Live: ${providerInfo.isLive}, Mock: ${providerInfo.isMock}`);

    // 2. Check Delta Exchange service
    logger.info('🔗 Checking Delta Exchange service...');
    const deltaCredentials = {
      apiKey: process.env.DELTA_API_KEY || '',
      apiSecret: process.env.DELTA_API_SECRET || '',
      testnet: true
    };

    const deltaService = new DeltaExchangeUnified(deltaCredentials);

    // Wait a moment for initialization
    await new Promise(resolve => setTimeout(resolve, 2000));

    report.deltaExchangeService.isReady = deltaService.isReady();

    // 3. Perform consistency checks
    logger.info('✅ Performing consistency checks...');
    
    // Check if using live data
    report.consistencyChecks.noMockDataInProduction = !providerInfo.isMock;
    
    // Check if data source is unified
    report.consistencyChecks.dataSourceUnified = providerInfo.name === 'delta-exchange';
    
    // Check if Delta service is ready
    report.consistencyChecks.deltaServiceReady = report.deltaExchangeService.isReady;

    // 4. Test actual data fetching
    logger.info('📈 Testing data fetching from both sources...');
    
    const testSymbol = 'BTCUSD';
    
    try {
      // Test MarketDataService
      const marketDataRequest = {
        symbol: testSymbol,
        timeframe: '1h',
        startDate: new Date(Date.now() - 24 * 60 * 60 * 1000), // 24 hours ago
        endDate: new Date(),
      };
      
      const marketDataResponse = await marketDataService.fetchHistoricalData(marketDataRequest);
      report.testResults.marketDataFetch = marketDataResponse.count > 0;
      
      if (marketDataResponse.data.length > 0) {
        report.testResults.priceComparison.marketDataPrice = marketDataResponse.data[marketDataResponse.data.length - 1].close;
      }
      
    } catch (error) {
      logger.error('❌ MarketDataService fetch failed:', error);
      report.testResults.marketDataFetch = false;
    }

    try {
      // Test Delta Exchange direct API
      const deltaMarketData = await deltaService.getMarketData(testSymbol);
      report.testResults.deltaExchangeDataFetch = true;
      report.testResults.priceComparison.deltaExchangePrice = parseFloat(
        deltaMarketData.mark_price || deltaMarketData.last_price || '0'
      );
      
    } catch (error) {
      logger.error('❌ Delta Exchange fetch failed:', error);
      report.testResults.deltaExchangeDataFetch = false;
    }

    // 5. Compare prices for consistency
    if (report.testResults.marketDataFetch && report.testResults.deltaExchangeDataFetch) {
      const priceDiff = Math.abs(
        report.testResults.priceComparison.marketDataPrice - 
        report.testResults.priceComparison.deltaExchangePrice
      );
      
      report.testResults.priceComparison.difference = priceDiff;
      
      // Allow 5% tolerance for price differences (due to timing and data source variations)
      const tolerance = report.testResults.priceComparison.deltaExchangePrice * 0.05;
      report.testResults.priceComparison.withinTolerance = priceDiff <= tolerance;
      report.consistencyChecks.priceDataConsistent = report.testResults.priceComparison.withinTolerance;
    }

    // 6. Generate recommendations
    if (providerInfo.isMock) {
      report.recommendations.push('🚨 CRITICAL: Switch from mock data provider to Delta Exchange for live trading');
    }
    
    if (!report.consistencyChecks.dataSourceUnified) {
      report.recommendations.push('⚠️ WARNING: Use Delta Exchange as the unified data source for all operations');
    }
    
    if (!report.deltaExchangeService.isReady) {
      report.recommendations.push('🔧 REQUIRED: Initialize Delta Exchange service before trading');
    }
    
    if (!report.consistencyChecks.priceDataConsistent) {
      report.recommendations.push('📊 WARNING: Price data inconsistency detected - verify data sources');
    }

    // 7. Determine overall status
    if (providerInfo.isMock) {
      report.overallStatus = 'CRITICAL';
    } else if (!report.consistencyChecks.dataSourceUnified || !report.consistencyChecks.priceDataConsistent) {
      report.overallStatus = 'WARNING';
    } else {
      report.overallStatus = 'SAFE';
    }

    return report;

  } catch (error) {
    logger.error('❌ Data consistency verification failed:', error);
    report.recommendations.push(`🚨 CRITICAL: Verification failed - ${error}`);
    report.overallStatus = 'CRITICAL';
    return report;
  }
}

async function testTradingBotDataConsistency(): Promise<void> {
  logger.info('🤖 Testing trading bot data consistency...');
  
  try {
    // Create a test bot configuration
    const testConfig: BotConfig = {
      id: 'test-consistency-bot',
      name: 'Data Consistency Test Bot',
      symbol: 'BTCUSD',
      strategy: 'momentum',
      capital: 1000,
      leverage: 3,
      riskPerTrade: 2,
      maxPositions: 1,
      stopLoss: 2,
      takeProfit: 4,
      enabled: false, // Don't actually trade
      testnet: true,
    };

    const deltaCredentials = {
      apiKey: process.env.DELTA_API_KEY || '',
      apiSecret: process.env.DELTA_API_SECRET || '',
      testnet: true
    };

    const deltaService = new DeltaExchangeUnified(deltaCredentials);
    const testBot = new DeltaTradingBot(testConfig, deltaService);
    
    // This should trigger data source validation
    logger.info('🔍 Testing bot configuration validation...');
    
    try {
      // This will call validateConfig() which includes our data consistency checks
      await testBot.start();
      logger.info('✅ Bot started successfully with consistent data sources');
      await testBot.stop();
      
    } catch (error) {
      if (error instanceof Error && error.message.includes('SAFETY VIOLATION')) {
        logger.info('✅ Data consistency validation working - bot correctly rejected mock data');
      } else {
        logger.error('❌ Unexpected bot validation error:', error);
        throw error;
      }
    }
    
  } catch (error) {
    logger.error('❌ Trading bot consistency test failed:', error);
    throw error;
  }
}

async function main(): Promise<void> {
  try {
    logger.info('🚀 Starting comprehensive data consistency verification...');
    
    // Run data consistency verification
    const report = await verifyDataConsistency();
    
    // Log detailed report
    logger.info('📋 Data Consistency Report:');
    logger.info(`Status: ${report.overallStatus}`);
    logger.info(`Provider: ${report.marketDataProvider.name} (Live: ${report.marketDataProvider.isLive})`);
    logger.info(`Delta Service Ready: ${report.deltaExchangeService.isReady}`);
    logger.info(`Price Consistency: ${report.consistencyChecks.priceDataConsistent}`);
    
    if (report.recommendations.length > 0) {
      logger.info('📝 Recommendations:');
      report.recommendations.forEach(rec => logger.info(`  ${rec}`));
    }

    // Test trading bot data consistency
    await testTradingBotDataConsistency();
    
    // Final status
    if (report.overallStatus === 'SAFE') {
      logger.info('🎉 Data consistency verification PASSED - System is safe for live trading');
    } else {
      logger.warn(`⚠️ Data consistency verification ${report.overallStatus} - Review recommendations before live trading`);
    }
    
  } catch (error) {
    logger.error('❌ Data consistency verification failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(error => {
    logger.error('❌ Script execution failed:', error);
    process.exit(1);
  });
}

export { verifyDataConsistency, testTradingBotDataConsistency };
