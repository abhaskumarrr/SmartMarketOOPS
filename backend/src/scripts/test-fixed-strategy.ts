#!/usr/bin/env node

/**
 * Test Fixed Strategy Implementation
 * Validates that all fixes from root cause analysis improve performance
 */

import { createFixedIntelligentTradingSystem } from '../services/fixedIntelligentTradingSystem';
import { createIntelligentTradingSystem } from '../services/intelligentTradingSystem';
import { marketDataService } from '../services/marketDataProvider';
import { technicalAnalysis } from '../utils/technicalAnalysis';
import { PortfolioManager } from '../services/portfolioManager';
import { PerformanceAnalytics } from '../utils/performanceAnalytics';
import { BacktestConfig, EnhancedMarketData } from '../types/marketData';
import { logger } from '../utils/logger';

class FixedStrategyTester {
  private marketData: EnhancedMarketData[] = [];
  private baseConfig: BacktestConfig;

  constructor() {
    this.baseConfig = this.createBaseConfig();
  }

  /**
   * Run comprehensive comparison between original and fixed strategies
   */
  public async runComparison(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🔧 Testing Fixed Strategy vs Original Strategy...');

      // Load market data
      await this.loadMarketData();

      // Test original strategy
      logger.info('\n📊 Testing ORIGINAL Strategy...');
      const originalResults = await this.testStrategy(
        createIntelligentTradingSystem(),
        'Original_Strategy'
      );

      // Test fixed strategy
      logger.info('\n🔧 Testing FIXED Strategy...');
      const fixedResults = await this.testStrategy(
        createFixedIntelligentTradingSystem(),
        'Fixed_Strategy'
      );

      // Compare results
      this.compareResults(originalResults, fixedResults, startTime);

      logger.info('🎉 Fixed strategy testing completed successfully!');

    } catch (error) {
      logger.error('❌ Fixed strategy testing failed:', error);
      throw error;
    }
  }

  /**
   * Test a specific strategy
   */
  private async testStrategy(strategy: any, strategyName: string): Promise<any> {
    strategy.initialize(this.baseConfig);
    const portfolioManager = new PortfolioManager(this.baseConfig);
    
    let signalCount = 0;
    let validSignals = 0;
    let tradeCount = 0;
    let signalDetails: any[] = [];

    logger.info(`🎯 Testing ${strategyName}...`);

    // Process all market data
    for (let i = 0; i < this.marketData.length; i++) {
      const currentCandle = this.marketData[i];

      portfolioManager.updatePositions(
        currentCandle.symbol, 
        currentCandle.close, 
        currentCandle.timestamp
      );

      const closedTrades = portfolioManager.checkStopLossAndTakeProfit(
        currentCandle.symbol,
        currentCandle.close,
        currentCandle.timestamp
      );

      tradeCount += closedTrades.length;

      // Generate signal
      const signal = strategy.generateSignal(this.marketData, i);
      
      if (signal) {
        signalCount++;
        
        signalDetails.push({
          index: i,
          timestamp: currentCandle.timestamp,
          price: currentCandle.close,
          signal: signal.type,
          confidence: signal.confidence,
          reason: signal.reason,
        });

        if (signal.confidence > 0) {
          validSignals++;
          
          const trade = portfolioManager.executeTrade(
            signal, 
            currentCandle.close, 
            currentCandle.timestamp
          );

          if (trade) {
            trade.strategy = strategyName;
            tradeCount++;
          }
        }
      }

      // Progress logging
      if (i % 100 === 0) {
        const progress = ((i / this.marketData.length) * 100).toFixed(1);
        logger.debug(`   Progress: ${progress}% (${signalCount} signals, ${tradeCount} trades)`);
      }
    }

    // Calculate performance
    const trades = portfolioManager.getTrades();
    const portfolioHistory = portfolioManager.getPortfolioHistory();
    const performance = PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, this.baseConfig);

    return {
      strategyName,
      performance,
      trades,
      portfolioHistory,
      signalCount,
      validSignals,
      tradeCount,
      signalDetails,
      dataPoints: this.marketData.length,
    };
  }

  /**
   * Compare results between strategies
   */
  private compareResults(originalResults: any, fixedResults: any, startTime: number): void {
    const duration = (Date.now() - startTime) / 1000;

    logger.info('\n' + '🔧 FIXED STRATEGY COMPARISON RESULTS'.padStart(60, '='));
    logger.info('=' .repeat(120));

    // Execution Summary
    logger.info('⚡ EXECUTION SUMMARY:');
    logger.info(`   Analysis Duration: ${duration.toFixed(2)} seconds`);
    logger.info(`   Data Points Processed: ${originalResults.dataPoints.toLocaleString()}`);
    logger.info(`   Processing Speed: ${(originalResults.dataPoints / duration).toFixed(0)} points/sec`);

    // Signal Generation Comparison
    logger.info('\n📊 SIGNAL GENERATION COMPARISON:');
    logger.info('   Metric                    Original      Fixed         Improvement');
    logger.info('   ' + '-'.repeat(70));
    
    const signalImprovement = ((fixedResults.signalCount - originalResults.signalCount) / Math.max(originalResults.signalCount, 1) * 100);
    const validSignalImprovement = ((fixedResults.validSignals - originalResults.validSignals) / Math.max(originalResults.validSignals, 1) * 100);
    const tradeImprovement = ((fixedResults.tradeCount - originalResults.tradeCount) / Math.max(originalResults.tradeCount, 1) * 100);
    
    logger.info(`   Total Signals             ${originalResults.signalCount.toString().padStart(8)} ${fixedResults.signalCount.toString().padStart(12)} ${signalImprovement > 0 ? '+' : ''}${signalImprovement.toFixed(0)}%`);
    logger.info(`   Valid Signals             ${originalResults.validSignals.toString().padStart(8)} ${fixedResults.validSignals.toString().padStart(12)} ${validSignalImprovement > 0 ? '+' : ''}${validSignalImprovement.toFixed(0)}%`);
    logger.info(`   Trades Executed           ${originalResults.tradeCount.toString().padStart(8)} ${fixedResults.tradeCount.toString().padStart(12)} ${tradeImprovement > 0 ? '+' : ''}${tradeImprovement.toFixed(0)}%`);

    // Performance Comparison
    logger.info('\n💰 PERFORMANCE COMPARISON:');
    logger.info('   Metric                    Original      Fixed         Improvement');
    logger.info('   ' + '-'.repeat(70));
    
    const metrics = [
      { name: 'Total Return (%)', orig: originalResults.performance.totalReturnPercent, fixed: fixedResults.performance.totalReturnPercent },
      { name: 'Sharpe Ratio', orig: originalResults.performance.sharpeRatio, fixed: fixedResults.performance.sharpeRatio },
      { name: 'Max Drawdown (%)', orig: originalResults.performance.maxDrawdownPercent, fixed: fixedResults.performance.maxDrawdownPercent },
      { name: 'Win Rate (%)', orig: originalResults.performance.winRate, fixed: fixedResults.performance.winRate },
      { name: 'Profit Factor', orig: originalResults.performance.profitFactor, fixed: fixedResults.performance.profitFactor },
    ];

    metrics.forEach(metric => {
      const improvement = metric.fixed - metric.orig;
      const improvementStr = improvement > 0 ? `+${improvement.toFixed(2)}` : improvement.toFixed(2);
      const indicator = improvement > 0 ? '📈' : improvement < 0 ? '📉' : '➡️';
      
      logger.info(`   ${metric.name.padEnd(25)} ${metric.orig.toFixed(2).padStart(8)} ${metric.fixed.toFixed(2).padStart(12)} ${indicator} ${improvementStr.padStart(8)}`);
    });

    // Fix Validation
    logger.info('\n🔧 FIX VALIDATION:');
    
    // Fix #1: Lower thresholds
    if (fixedResults.signalCount > originalResults.signalCount) {
      logger.info(`   ✅ Fix #1 - Lower Thresholds: ${fixedResults.signalCount - originalResults.signalCount} more signals generated`);
    } else {
      logger.info(`   ❌ Fix #1 - Lower Thresholds: No improvement in signal generation`);
    }
    
    // Fix #2: Enhanced AI models
    if (fixedResults.validSignals > originalResults.validSignals) {
      logger.info(`   ✅ Fix #2 - Enhanced AI Models: ${fixedResults.validSignals - originalResults.validSignals} more valid signals`);
    } else {
      logger.info(`   ❌ Fix #2 - Enhanced AI Models: No improvement in signal quality`);
    }
    
    // Fix #3: Sideways market trading
    if (fixedResults.tradeCount > originalResults.tradeCount) {
      logger.info(`   ✅ Fix #3 - Sideways Trading: ${fixedResults.tradeCount - originalResults.tradeCount} more trades executed`);
    } else {
      logger.info(`   ❌ Fix #3 - Sideways Trading: No improvement in trade execution`);
    }
    
    // Fix #4: Risk management
    const returnImprovement = fixedResults.performance.totalReturnPercent - originalResults.performance.totalReturnPercent;
    if (returnImprovement > 0) {
      logger.info(`   ✅ Fix #4 - Risk Management: ${returnImprovement.toFixed(2)}% better returns`);
    } else {
      logger.info(`   ❌ Fix #4 - Risk Management: ${Math.abs(returnImprovement).toFixed(2)}% worse returns`);
    }

    // Signal Details Analysis
    if (fixedResults.signalDetails.length > 0) {
      logger.info('\n🎯 FIXED STRATEGY SIGNAL ANALYSIS:');
      logger.info(`   First 5 Signals from Fixed Strategy:`);
      fixedResults.signalDetails.slice(0, 5).forEach((detail: any, idx: number) => {
        const date = new Date(detail.timestamp).toISOString().split('T')[0];
        logger.info(`     ${idx + 1}. ${detail.signal} at $${detail.price.toFixed(0)} (${detail.confidence.toFixed(1)}%) - ${date}`);
      });
    }

    // Trade Analysis
    if (fixedResults.trades.length > 0) {
      logger.info('\n💼 FIXED STRATEGY TRADE ANALYSIS:');
      logger.info(`   Trades Executed by Fixed Strategy:`);
      fixedResults.trades.forEach((trade: any, idx: number) => {
        const duration = (trade.duration / (1000 * 60 * 60)).toFixed(1);
        logger.info(`     ${idx + 1}. ${trade.side}: Entry $${trade.entryPrice.toFixed(0)} → Exit $${trade.exitPrice.toFixed(0)} = ${trade.pnlPercent.toFixed(2)}% (${duration}h)`);
      });
    }

    // Overall Assessment
    logger.info('\n⭐ OVERALL ASSESSMENT:');
    
    let improvementScore = 0;
    if (fixedResults.signalCount > originalResults.signalCount) improvementScore++;
    if (fixedResults.validSignals > originalResults.validSignals) improvementScore++;
    if (fixedResults.tradeCount > originalResults.tradeCount) improvementScore++;
    if (fixedResults.performance.totalReturnPercent > originalResults.performance.totalReturnPercent) improvementScore++;
    if (fixedResults.performance.sharpeRatio > originalResults.performance.sharpeRatio) improvementScore++;
    
    if (improvementScore >= 4) {
      logger.info('   🌟 EXCELLENT - Fixed strategy shows significant improvements across all metrics');
    } else if (improvementScore >= 3) {
      logger.info('   ✅ GOOD - Fixed strategy shows improvements in most areas');
    } else if (improvementScore >= 2) {
      logger.info('   ⚠️ MODERATE - Fixed strategy shows some improvements');
    } else {
      logger.info('   ❌ POOR - Fixed strategy needs further optimization');
    }

    // Recommendations
    logger.info('\n💡 RECOMMENDATIONS:');
    if (fixedResults.signalCount > originalResults.signalCount * 2) {
      logger.info('   🚀 Signal generation significantly improved - ready for live testing');
    }
    if (fixedResults.performance.totalReturnPercent > 5) {
      logger.info('   📈 Strong positive returns achieved - validate with different market conditions');
    }
    if (fixedResults.tradeCount > 10) {
      logger.info('   💼 Good trade frequency - strategy is actively trading');
    }
    
    logger.info('   🔧 Continue monitoring and optimizing based on live market conditions');
    logger.info('   📊 Consider testing on different timeframes and market regimes');

    logger.info('=' .repeat(120));
  }

  // Helper methods
  private async loadMarketData(): Promise<void> {
    const response = await marketDataService.fetchHistoricalData({
      symbol: this.baseConfig.symbol,
      timeframe: this.baseConfig.timeframe,
      startDate: this.baseConfig.startDate,
      endDate: this.baseConfig.endDate,
      exchange: 'enhanced-mock',
    });

    const closes = response.data.map(d => d.close);
    const volumes = response.data.map(d => d.volume);
    const highs = response.data.map(d => d.high);
    const lows = response.data.map(d => d.low);

    const sma20 = technicalAnalysis.calculateSMA(closes, 20);
    const sma50 = technicalAnalysis.calculateSMA(closes, 50);
    const ema12 = technicalAnalysis.calculateEMA(closes, 12);
    const ema26 = technicalAnalysis.calculateEMA(closes, 26);
    const rsi = technicalAnalysis.calculateRSI(closes, 14);
    const macd = technicalAnalysis.calculateMACD(closes, 12, 26, 9);
    const bollinger = technicalAnalysis.calculateBollingerBands(closes, 20, 2);
    const volumeSMA = technicalAnalysis.calculateSMA(volumes, 20);

    this.marketData = response.data.map((point, index) => ({
      ...point,
      indicators: {
        sma_20: sma20[index],
        sma_50: sma50[index],
        ema_12: ema12[index],
        ema_26: ema26[index],
        rsi: rsi[index],
        macd: macd.macd[index],
        macd_signal: macd.signal[index],
        macd_histogram: macd.histogram[index],
        bollinger_upper: bollinger.upper[index],
        bollinger_middle: bollinger.middle[index],
        bollinger_lower: bollinger.lower[index],
        volume_sma: volumeSMA[index],
      },
    }));
  }

  private createBaseConfig(): BacktestConfig {
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - (30 * 24 * 60 * 60 * 1000));

    return {
      symbol: 'BTCUSD',
      timeframe: '1h',
      startDate,
      endDate,
      initialCapital: 2000,
      leverage: 3,
      riskPerTrade: 2,
      commission: 0.1,
      slippage: 0.05,
      strategy: 'Fixed_Strategy_Test',
      parameters: {},
    };
  }
}

/**
 * Main execution function
 */
async function main() {
  const tester = new FixedStrategyTester();
  
  try {
    await tester.runComparison();
  } catch (error) {
    logger.error('💥 Fixed strategy testing failed:', error);
    process.exit(1);
  }
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { FixedStrategyTester };
