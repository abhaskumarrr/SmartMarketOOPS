#!/usr/bin/env node

/**
 * Strategy Comparison Script
 * Compares original MA Crossover vs Enhanced Trend Strategy
 */

import { marketDataService } from '../services/marketDataProvider';
import { technicalAnalysis } from '../utils/technicalAnalysis';
import { PortfolioManager } from '../services/portfolioManager';
import { PerformanceAnalytics } from '../utils/performanceAnalytics';
import { createMACrossoverStrategy } from '../strategies/movingAverageCrossover';
import { createEnhancedTrendStrategy } from '../strategies/enhancedTrendStrategy';
import { BacktestConfig, EnhancedMarketData } from '../types/marketData';
import { logger } from '../utils/logger';
import { redisStreamsService } from '../services/redisStreamsService';

class StrategyComparison {
  /**
   * Run strategy comparison
   */
  public async runComparison(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🔬 Starting strategy comparison analysis...');

      // Initialize infrastructure
      await this.initializeInfrastructure();
      
      // Create configurations
      const config = this.createBacktestConfig();
      
      // Load market data once
      const marketData = await this.loadMarketData(config);
      
      // Test original strategy
      logger.info('\n📊 Testing Original MA Crossover Strategy...');
      const originalResults = await this.testStrategy(
        config, 
        createMACrossoverStrategy(), 
        marketData,
        'Original MA Crossover'
      );
      
      // Test enhanced strategy
      logger.info('\n🚀 Testing Enhanced Trend Strategy...');
      const enhancedResults = await this.testStrategy(
        config, 
        createEnhancedTrendStrategy(), 
        marketData,
        'Enhanced Trend'
      );
      
      // Compare results
      this.compareStrategies(originalResults, enhancedResults, startTime);

      logger.info('🎉 Strategy comparison completed successfully!');

    } catch (error) {
      logger.error('❌ Strategy comparison failed:', error);
      throw error;
    }
  }

  private async initializeInfrastructure(): Promise<void> {
    try {
      await redisStreamsService.initialize();
      logger.info('✅ Redis Streams initialized');
    } catch (error) {
      logger.warn('⚠️ Redis Streams initialization failed, continuing without it');
    }
  }

  private createBacktestConfig(): BacktestConfig {
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
      strategy: 'Comparison',
      parameters: {},
    };
  }

  private async loadMarketData(config: BacktestConfig): Promise<EnhancedMarketData[]> {
    logger.info('📊 Loading market data for comparison...');

    const response = await marketDataService.fetchHistoricalData({
      symbol: config.symbol,
      timeframe: config.timeframe,
      startDate: config.startDate,
      endDate: config.endDate,
      exchange: 'enhanced-mock',
    });

    // Enhance with technical indicators
    const closes = response.data.map(d => d.close);
    const volumes = response.data.map(d => d.volume);
    const highs = response.data.map(d => d.high);
    const lows = response.data.map(d => d.low);

    // Calculate comprehensive indicators
    const sma20 = technicalAnalysis.calculateSMA(closes, 20);
    const sma50 = technicalAnalysis.calculateSMA(closes, 50);
    const ema12 = technicalAnalysis.calculateEMA(closes, 12);
    const ema26 = technicalAnalysis.calculateEMA(closes, 26);
    const rsi = technicalAnalysis.calculateRSI(closes, 14);
    const macd = technicalAnalysis.calculateMACD(closes, 12, 26, 9);
    const bollinger = technicalAnalysis.calculateBollingerBands(closes, 20, 2);
    const volumeSMA = technicalAnalysis.calculateSMA(volumes, 20);

    const enhancedData: EnhancedMarketData[] = response.data.map((point, index) => ({
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

    logger.info(`✅ Enhanced ${response.data.length} data points with comprehensive indicators`);
    return enhancedData;
  }

  private async testStrategy(
    config: BacktestConfig, 
    strategy: any, 
    marketData: EnhancedMarketData[],
    strategyName: string
  ): Promise<any> {
    strategy.initialize(config);
    const portfolioManager = new PortfolioManager(config);
    
    let signalCount = 0;
    let tradeCount = 0;
    let validSignals = 0;

    for (let i = 0; i < marketData.length; i++) {
      const currentCandle = marketData[i];

      // Update portfolio
      portfolioManager.updatePositions(
        currentCandle.symbol, 
        currentCandle.close, 
        currentCandle.timestamp
      );

      // Check stop loss/take profit
      const closedTrades = portfolioManager.checkStopLossAndTakeProfit(
        currentCandle.symbol,
        currentCandle.close,
        currentCandle.timestamp
      );

      tradeCount += closedTrades.length;

      // Generate signal
      const signal = strategy.generateSignal(marketData, i);
      
      if (signal) {
        signalCount++;
        
        if (signal.confidence > 0) {
          validSignals++;
          
          // Execute trade
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

      // Portfolio snapshots
      if (i % 24 === 0 || signal) {
        portfolioManager.createSnapshot(currentCandle.timestamp);
      }
    }

    // Calculate performance
    const trades = portfolioManager.getTrades();
    const portfolioHistory = portfolioManager.getPortfolioHistory();
    const performance = PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, config);

    return {
      strategyName,
      config,
      performance,
      trades,
      portfolioHistory,
      dataPoints: marketData.length,
      signalCount,
      validSignals,
      tradeCount,
      strategy,
    };
  }

  private compareStrategies(originalResults: any, enhancedResults: any, startTime: number): void {
    const duration = (Date.now() - startTime) / 1000;

    logger.info('\n' + '🔬 STRATEGY COMPARISON ANALYSIS'.padStart(50, '='));
    logger.info('=' .repeat(100));

    // Execution Summary
    logger.info('⚡ EXECUTION SUMMARY:');
    logger.info(`   Analysis Duration: ${duration.toFixed(2)} seconds`);
    logger.info(`   Data Points: ${originalResults.dataPoints.toLocaleString()}`);
    logger.info(`   Processing Speed: ${(originalResults.dataPoints / duration).toFixed(0)} points/sec`);

    // Signal Generation Comparison
    logger.info('\n📊 SIGNAL GENERATION:');
    logger.info(`   Original Strategy:`);
    logger.info(`     Total Signals: ${originalResults.signalCount}`);
    logger.info(`     Valid Signals: ${originalResults.validSignals}`);
    logger.info(`     Signal Quality: ${((originalResults.validSignals / Math.max(originalResults.signalCount, 1)) * 100).toFixed(1)}%`);
    
    logger.info(`   Enhanced Strategy:`);
    logger.info(`     Total Signals: ${enhancedResults.signalCount}`);
    logger.info(`     Valid Signals: ${enhancedResults.validSignals}`);
    logger.info(`     Signal Quality: ${((enhancedResults.validSignals / Math.max(enhancedResults.signalCount, 1)) * 100).toFixed(1)}%`);

    // Performance Comparison
    logger.info('\n💰 PERFORMANCE COMPARISON:');
    
    const metrics = [
      { name: 'Total Return (%)', orig: originalResults.performance.totalReturnPercent, enh: enhancedResults.performance.totalReturnPercent },
      { name: 'Annualized Return (%)', orig: originalResults.performance.annualizedReturn, enh: enhancedResults.performance.annualizedReturn },
      { name: 'Sharpe Ratio', orig: originalResults.performance.sharpeRatio, enh: enhancedResults.performance.sharpeRatio },
      { name: 'Max Drawdown (%)', orig: originalResults.performance.maxDrawdownPercent, enh: enhancedResults.performance.maxDrawdownPercent },
      { name: 'Win Rate (%)', orig: originalResults.performance.winRate, enh: enhancedResults.performance.winRate },
      { name: 'Profit Factor', orig: originalResults.performance.profitFactor, enh: enhancedResults.performance.profitFactor },
      { name: 'Total Trades', orig: originalResults.performance.totalTrades, enh: enhancedResults.performance.totalTrades },
    ];

    logger.info('   Metric                    Original      Enhanced      Improvement');
    logger.info('   ' + '-'.repeat(65));
    
    metrics.forEach(metric => {
      const improvement = metric.enh - metric.orig;
      const improvementStr = improvement > 0 ? `+${improvement.toFixed(2)}` : improvement.toFixed(2);
      const indicator = improvement > 0 ? '📈' : improvement < 0 ? '📉' : '➡️';
      
      logger.info(`   ${metric.name.padEnd(25)} ${metric.orig.toFixed(2).padStart(8)} ${metric.enh.toFixed(2).padStart(12)} ${indicator} ${improvementStr.padStart(8)}`);
    });

    // Strategy Analysis
    logger.info('\n🎯 STRATEGY ANALYSIS:');
    
    // Original Strategy Issues
    logger.info('   Original MA Crossover Issues:');
    if (originalResults.performance.totalReturnPercent < 0) {
      logger.info('     🔴 Negative returns - lagging indicators problem');
    }
    if (originalResults.performance.winRate < 40) {
      logger.info('     🔴 Low win rate - poor signal timing');
    }
    if (originalResults.performance.maxDrawdownPercent > 30) {
      logger.info('     🔴 High drawdown - inadequate risk management');
    }
    if (originalResults.performance.totalTrades > 10) {
      logger.info('     🔴 Overtrading - whipsaw in sideways markets');
    }

    // Enhanced Strategy Improvements
    logger.info('   Enhanced Strategy Improvements:');
    if (enhancedResults.performance.totalReturnPercent > originalResults.performance.totalReturnPercent) {
      logger.info('     ✅ Better returns through trend analysis');
    }
    if (enhancedResults.performance.winRate > originalResults.performance.winRate) {
      logger.info('     ✅ Higher win rate with better signal timing');
    }
    if (enhancedResults.performance.maxDrawdownPercent < originalResults.performance.maxDrawdownPercent) {
      logger.info('     ✅ Lower drawdown with dynamic risk management');
    }
    if (enhancedResults.validSignals < originalResults.validSignals) {
      logger.info('     ✅ Fewer but higher quality signals (anti-whipsaw)');
    }

    // Overall Assessment
    logger.info('\n⭐ OVERALL ASSESSMENT:');
    const improvementScore = this.calculateImprovementScore(originalResults.performance, enhancedResults.performance);
    
    if (improvementScore >= 3) {
      logger.info('   🌟 SIGNIFICANT IMPROVEMENT - Enhanced strategy is substantially better');
    } else if (improvementScore >= 1) {
      logger.info('   ✅ MODERATE IMPROVEMENT - Enhanced strategy shows promise');
    } else if (improvementScore >= -1) {
      logger.info('   ⚠️ MIXED RESULTS - Some improvements, some regressions');
    } else {
      logger.info('   ❌ POOR RESULTS - Enhanced strategy needs further work');
    }

    // Recommendations
    logger.info('\n💡 RECOMMENDATIONS:');
    if (enhancedResults.performance.totalReturnPercent > 0) {
      logger.info('   ✅ Enhanced strategy shows positive returns - consider live testing');
    }
    if (enhancedResults.performance.winRate > 50) {
      logger.info('   ✅ Good win rate - strategy has edge in current market conditions');
    }
    if (enhancedResults.performance.sharpeRatio > 1) {
      logger.info('   ✅ Good risk-adjusted returns - acceptable risk profile');
    }
    
    logger.info('   📊 Consider testing on different market periods and timeframes');
    logger.info('   🔧 Fine-tune parameters based on market regime analysis');
    logger.info('   📈 Monitor performance in live paper trading before real deployment');

    logger.info('=' .repeat(100));
  }

  private calculateImprovementScore(original: any, enhanced: any): number {
    let score = 0;
    
    // Return improvement
    if (enhanced.totalReturnPercent > original.totalReturnPercent + 10) score += 2;
    else if (enhanced.totalReturnPercent > original.totalReturnPercent) score += 1;
    else if (enhanced.totalReturnPercent < original.totalReturnPercent - 10) score -= 2;
    else score -= 1;
    
    // Sharpe ratio improvement
    if (enhanced.sharpeRatio > original.sharpeRatio + 0.5) score += 1;
    else if (enhanced.sharpeRatio < original.sharpeRatio - 0.5) score -= 1;
    
    // Drawdown improvement
    if (enhanced.maxDrawdownPercent < original.maxDrawdownPercent - 5) score += 1;
    else if (enhanced.maxDrawdownPercent > original.maxDrawdownPercent + 5) score -= 1;
    
    // Win rate improvement
    if (enhanced.winRate > original.winRate + 10) score += 1;
    else if (enhanced.winRate < original.winRate - 10) score -= 1;
    
    return score;
  }

  public async cleanup(): Promise<void> {
    try {
      await redisStreamsService.shutdown();
      logger.info('🧹 Cleanup completed');
    } catch (error) {
      logger.error('❌ Cleanup failed:', error);
    }
  }
}

/**
 * Main execution function
 */
async function main() {
  const comparison = new StrategyComparison();
  
  try {
    await comparison.runComparison();
  } catch (error) {
    logger.error('💥 Strategy comparison failed:', error);
    process.exit(1);
  } finally {
    await comparison.cleanup();
  }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  logger.info('🛑 Received SIGINT, cleaning up...');
  const comparison = new StrategyComparison();
  await comparison.cleanup();
  process.exit(0);
});

// Run if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { StrategyComparison };
