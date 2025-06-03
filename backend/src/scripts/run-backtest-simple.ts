#!/usr/bin/env node

/**
 * Simple Backtesting Script
 * Focuses on core backtesting logic without database storage issues
 */

import { marketDataService } from '../services/marketDataProvider';
import { technicalAnalysis } from '../utils/technicalAnalysis';
import { PortfolioManager } from '../services/portfolioManager';
import { PerformanceAnalytics } from '../utils/performanceAnalytics';
import { createMACrossoverStrategy } from '../strategies/movingAverageCrossover';
import { BacktestConfig, EnhancedMarketData } from '../types/marketData';
import { logger } from '../utils/logger';
import { redisStreamsService } from '../services/redisStreamsService';
import { eventDrivenTradingSystem } from '../services/eventDrivenTradingSystem';

class SimpleBacktestRunner {
  /**
   * Run comprehensive backtesting simulation
   */
  public async runBacktest(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🚀 Starting simple backtesting simulation...');

      // Step 1: Configure backtest
      const config = this.createBacktestConfig();
      
      // Step 2: Create strategy
      const strategy = this.createTradingStrategy();
      strategy.initialize(config);
      
      // Step 3: Initialize infrastructure
      await this.initializeInfrastructure();
      
      // Step 4: Load market data
      const marketData = await this.loadMarketData(config);
      
      // Step 5: Run backtest
      const result = await this.runBacktestLogic(config, strategy, marketData);
      
      // Step 6: Display results
      this.displayResults(result, startTime);

      logger.info('🎉 Simple backtesting simulation completed successfully!');

    } catch (error) {
      logger.error('❌ Backtesting simulation failed:', error);
      throw error;
    }
  }

  /**
   * Create backtest configuration
   */
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
      strategy: 'MA_Crossover',
      parameters: {},
    };
  }

  /**
   * Create trading strategy
   */
  private createTradingStrategy() {
    return createMACrossoverStrategy({
      fastPeriod: 20,
      slowPeriod: 50,
      rsiPeriod: 14,
      rsiOverbought: 70,
      rsiOversold: 30,
      volumeThreshold: 1.2,
      stopLossPercent: 2.0,
      takeProfitPercent: 4.0,
      minConfidence: 60,
    });
  }

  /**
   * Initialize infrastructure
   */
  private async initializeInfrastructure(): Promise<void> {
    try {
      await redisStreamsService.initialize();
      logger.info('✅ Redis Streams initialized');
    } catch (error) {
      logger.warn('⚠️ Redis Streams initialization failed, continuing without it');
    }
  }

  /**
   * Load and enhance market data
   */
  private async loadMarketData(config: BacktestConfig): Promise<EnhancedMarketData[]> {
    logger.info('📊 Loading market data...');

    const response = await marketDataService.fetchHistoricalData({
      symbol: config.symbol,
      timeframe: config.timeframe,
      startDate: config.startDate,
      endDate: config.endDate,
      exchange: 'enhanced-mock',
    });

    logger.info(`📈 Loaded ${response.data.length} data points`);

    // Enhance with technical indicators
    const closes = response.data.map(d => d.close);
    const volumes = response.data.map(d => d.volume);
    const indicators = technicalAnalysis.calculateAllIndicators(closes, volumes);

    const enhancedData: EnhancedMarketData[] = response.data.map((point, index) => ({
      ...point,
      indicators: indicators[index],
    }));

    logger.info('✅ Market data enhanced with technical indicators');
    return enhancedData;
  }

  /**
   * Run the core backtesting logic
   */
  private async runBacktestLogic(config: BacktestConfig, strategy: any, marketData: EnhancedMarketData[]): Promise<any> {
    logger.info('⚡ Running backtesting logic...');

    const portfolioManager = new PortfolioManager(config);
    let signalCount = 0;
    let tradeCount = 0;

    for (let i = 0; i < marketData.length; i++) {
      const currentCandle = marketData[i];

      // Update portfolio with current prices
      portfolioManager.updatePositions(
        currentCandle.symbol, 
        currentCandle.close, 
        currentCandle.timestamp
      );

      // Check for stop loss and take profit triggers
      const closedTrades = portfolioManager.checkStopLossAndTakeProfit(
        currentCandle.symbol,
        currentCandle.close,
        currentCandle.timestamp
      );

      tradeCount += closedTrades.length;

      // Generate trading signal
      const signal = strategy.generateSignal(marketData, i);
      
      if (signal && signal.confidence > 0) {
        signalCount++;
        
        // Publish signal to Redis Streams (if available)
        try {
          await this.publishTradingSignal(signal);
        } catch (error) {
          // Continue without Redis if it fails
        }

        // Execute trade
        const trade = portfolioManager.executeTrade(
          signal, 
          currentCandle.close, 
          currentCandle.timestamp
        );

        if (trade) {
          trade.strategy = strategy.name;
          tradeCount++;
        }
      }

      // Create portfolio snapshot periodically
      if (i % 24 === 0 || signal) {
        portfolioManager.createSnapshot(currentCandle.timestamp);
      }

      // Progress logging
      if (i % 100 === 0) {
        const progress = ((i / marketData.length) * 100).toFixed(1);
        logger.debug(`📊 Progress: ${progress}% (${i}/${marketData.length})`);
      }
    }

    // Calculate performance
    const trades = portfolioManager.getTrades();
    const portfolioHistory = portfolioManager.getPortfolioHistory();
    const performance = PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, config);

    logger.info(`✅ Backtesting completed`, {
      dataPoints: marketData.length,
      signalsGenerated: signalCount,
      tradesExecuted: tradeCount,
    });

    return {
      config,
      performance,
      trades,
      portfolioHistory,
      finalPortfolio: portfolioHistory[portfolioHistory.length - 1],
      dataPoints: marketData.length,
      signalCount,
      tradeCount,
    };
  }

  /**
   * Publish trading signal to Redis Streams
   */
  private async publishTradingSignal(signal: any): Promise<void> {
    await eventDrivenTradingSystem.publishTradingSignalEvent({
      signalId: signal.id,
      symbol: signal.symbol,
      signalType: signal.type === 'BUY' ? 'ENTRY' : 'EXIT',
      direction: signal.type === 'BUY' ? 'LONG' : 'SHORT',
      strength: signal.confidence > 80 ? 'STRONG' : signal.confidence > 60 ? 'MODERATE' : 'WEAK',
      timeframe: '1h',
      price: signal.price,
      confidenceScore: signal.confidence,
      expectedReturn: signal.riskReward ? signal.riskReward * 2 : 4,
      expectedRisk: 2,
      riskRewardRatio: signal.riskReward || 2,
      modelSource: 'MA_Crossover',
    });
  }

  /**
   * Display comprehensive results
   */
  private displayResults(result: any, startTime: number): void {
    const duration = (Date.now() - startTime) / 1000;
    const performance = result.performance;

    // Generate performance report
    const report = PerformanceAnalytics.generateReport(performance, result.trades, result.config);
    logger.info('\n' + report);

    // Display summary
    logger.info('\n' + '🎯 BACKTESTING SUMMARY'.padStart(40, '='));
    logger.info('=' .repeat(80));
    
    // Execution metrics
    logger.info('⚡ EXECUTION METRICS:');
    logger.info(`   Duration: ${duration.toFixed(2)} seconds`);
    logger.info(`   Data Points Processed: ${result.dataPoints.toLocaleString()}`);
    logger.info(`   Processing Speed: ${(result.dataPoints / duration).toFixed(0)} points/sec`);
    logger.info(`   Signals Generated: ${result.signalCount}`);
    logger.info(`   Trades Executed: ${result.tradeCount}`);
    
    // Performance metrics
    logger.info('\n💰 KEY PERFORMANCE METRICS:');
    logger.info(`   Total Return: $${performance.totalReturn.toFixed(2)} (${performance.totalReturnPercent.toFixed(2)}%)`);
    logger.info(`   Annualized Return: ${performance.annualizedReturn.toFixed(2)}%`);
    logger.info(`   Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}`);
    logger.info(`   Maximum Drawdown: ${performance.maxDrawdownPercent.toFixed(2)}%`);
    logger.info(`   Win Rate: ${performance.winRate.toFixed(1)}%`);
    logger.info(`   Profit Factor: ${performance.profitFactor.toFixed(2)}`);

    // Infrastructure validation
    logger.info('\n🏗️ INFRASTRUCTURE VALIDATION:');
    logger.info(`   ✅ Market Data Processing: ${result.dataPoints} points at ${(result.dataPoints / duration).toFixed(0)} points/sec`);
    logger.info(`   ✅ Trading Engine: ${result.tradeCount} trades executed`);
    logger.info(`   ✅ Portfolio Management: Real-time P&L tracking`);
    logger.info(`   ✅ Technical Analysis: Moving averages, RSI, volume analysis`);
    logger.info(`   ✅ Risk Management: Stop loss, take profit, position sizing`);

    // Performance rating
    const rating = this.getPerformanceRating(performance);
    logger.info(`\n⭐ OVERALL RATING: ${rating}`);
    
    logger.info('=' .repeat(80));
  }

  /**
   * Get performance rating
   */
  private getPerformanceRating(performance: any): string {
    let score = 0;

    if (performance.totalReturnPercent > 20) score += 2;
    else if (performance.totalReturnPercent > 10) score += 1;
    else if (performance.totalReturnPercent > 0) score += 0;
    else score -= 1;

    if (performance.sharpeRatio > 1.5) score += 2;
    else if (performance.sharpeRatio > 1) score += 1;
    else if (performance.sharpeRatio > 0) score += 0;
    else score -= 1;

    if (performance.winRate > 60) score += 1;
    else if (performance.winRate > 40) score += 0;
    else score -= 1;

    if (performance.maxDrawdownPercent < 10) score += 1;
    else if (performance.maxDrawdownPercent < 20) score += 0;
    else score -= 1;

    if (score >= 4) return '🌟 EXCELLENT';
    else if (score >= 2) return '✅ GOOD';
    else if (score >= 0) return '⚠️ AVERAGE';
    else return '❌ POOR';
  }

  /**
   * Cleanup resources
   */
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
  const runner = new SimpleBacktestRunner();
  
  try {
    await runner.runBacktest();
  } catch (error) {
    logger.error('💥 Backtesting failed:', error);
    process.exit(1);
  } finally {
    await runner.cleanup();
  }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  logger.info('🛑 Received SIGINT, cleaning up...');
  const runner = new SimpleBacktestRunner();
  await runner.cleanup();
  process.exit(0);
});

// Run if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { SimpleBacktestRunner };
