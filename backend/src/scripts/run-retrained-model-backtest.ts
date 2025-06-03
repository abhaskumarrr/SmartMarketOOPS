#!/usr/bin/env node

/**
 * Retrained Model Comprehensive Backtest
 * Tests the retrained AI models against the original models with multi-timeframe analysis
 */

import { createRetrainedAITradingSystem } from '../services/retrainedAITradingSystem';
import { createMultiTimeframeAITradingSystem } from '../services/multiTimeframeAITradingSystem';
import { createMultiTimeframeBacktester } from '../services/multiTimeframeBacktester';
import { marketDataService } from '../services/marketDataProvider';
import { PortfolioManager } from '../services/portfolioManager';
import { PerformanceAnalytics } from '../utils/performanceAnalytics';
import { BacktestConfig } from '../types/marketData';
import { Timeframe } from '../services/multiTimeframeDataProvider';
import { logger } from '../utils/logger';

interface ModelComparisonResult {
  originalModel: {
    performance: any;
    trades: any[];
    signals: number;
  };
  retrainedModel: {
    performance: any;
    trades: any[];
    signals: number;
  };
  improvement: {
    returnImprovement: number;
    sharpeImprovement: number;
    drawdownImprovement: number;
    winRateImprovement: number;
    signalImprovement: number;
  };
  config: BacktestConfig;
}

class RetrainedModelBacktestRunner {
  private backtester = createMultiTimeframeBacktester();

  /**
   * Run comprehensive backtest comparing original vs retrained models
   */
  public async runComparativeBacktest(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🧠 Starting Retrained Model Comprehensive Backtest...');

      // Step 1: Test configurations
      const testConfigs = this.createTestConfigurations();

      // Step 2: Run comparative tests
      const comparisonResults: ModelComparisonResult[] = [];

      for (const config of testConfigs) {
        logger.info(`\n🔬 Testing configuration: ${config.strategy}`);
        const result = await this.runModelComparison(config);
        comparisonResults.push(result);
      }

      // Step 3: Analyze overall results
      this.analyzeOverallResults(comparisonResults, startTime);

      // Step 4: Generate recommendations
      this.generateRecommendations(comparisonResults);

      const duration = (Date.now() - startTime) / 1000;
      logger.info(`🎉 Retrained model backtest completed in ${duration.toFixed(2)} seconds`);

    } catch (error) {
      logger.error('❌ Retrained model backtest failed:', error);
      throw error;
    }
  }

  /**
   * Create test configurations for comparison
   */
  private createTestConfigurations(): BacktestConfig[] {
    const baseEndDate = new Date();
    const baseStartDate = new Date(baseEndDate.getTime() - (14 * 24 * 60 * 60 * 1000)); // 14 days

    return [
      {
        symbol: 'BTCUSD',
        timeframe: '1h',
        startDate: baseStartDate,
        endDate: baseEndDate,
        initialCapital: 10000,
        leverage: 2,
        riskPerTrade: 2,
        commission: 0.1,
        slippage: 0.05,
        strategy: 'Conservative_Comparison',
        parameters: {},
      },
      {
        symbol: 'BTCUSD',
        timeframe: '1h',
        startDate: baseStartDate,
        endDate: baseEndDate,
        initialCapital: 20000,
        leverage: 3,
        riskPerTrade: 3,
        commission: 0.1,
        slippage: 0.05,
        strategy: 'Aggressive_Comparison',
        parameters: {},
      },
      {
        symbol: 'BTCUSD',
        timeframe: '4h',
        startDate: baseStartDate,
        endDate: baseEndDate,
        initialCapital: 15000,
        leverage: 2.5,
        riskPerTrade: 2.5,
        commission: 0.1,
        slippage: 0.05,
        strategy: 'Swing_Trading_Comparison',
        parameters: {},
      },
    ];
  }

  /**
   * Run model comparison for a specific configuration
   */
  private async runModelComparison(config: BacktestConfig): Promise<ModelComparisonResult> {
    logger.info(`📊 Running model comparison for ${config.strategy}...`);

    // Load market data
    const marketData = await this.loadMarketData(config);

    // Test original model
    logger.info('🔄 Testing original multi-timeframe AI model...');
    const originalResults = await this.testModel(
      createMultiTimeframeAITradingSystem(),
      marketData,
      config,
      'Original_Multi_TF_AI'
    );

    // Test retrained model
    logger.info('🧠 Testing retrained AI model...');
    const retrainedResults = await this.testModel(
      await this.createAndInitializeRetrainedModel(config),
      marketData,
      config,
      'Retrained_AI'
    );

    // Calculate improvements
    const improvement = this.calculateImprovements(originalResults, retrainedResults);

    return {
      originalModel: originalResults,
      retrainedModel: retrainedResults,
      improvement,
      config,
    };
  }

  /**
   * Load market data for testing
   */
  private async loadMarketData(config: BacktestConfig): Promise<any[]> {
    try {
      const response = await marketDataService.fetchHistoricalData({
        symbol: config.symbol,
        timeframe: config.timeframe,
        startDate: config.startDate,
        endDate: config.endDate,
        exchange: 'binance',
      }, 'binance');

      return this.enhanceMarketData(response.data);
    } catch (error) {
      logger.warn('⚠️ Failed to load real data, using enhanced mock data');
      
      const response = await marketDataService.fetchHistoricalData({
        symbol: config.symbol,
        timeframe: config.timeframe,
        startDate: config.startDate,
        endDate: config.endDate,
        exchange: 'enhanced-mock',
      }, 'enhanced-mock');

      return this.enhanceMarketData(response.data);
    }
  }

  /**
   * Enhance market data with technical indicators
   */
  private enhanceMarketData(data: any[]): any[] {
    // Add basic technical indicators for testing
    return data.map((point, index) => ({
      ...point,
      indicators: {
        rsi: 30 + Math.random() * 40,
        ema_12: point.close * (0.98 + Math.random() * 0.04),
        ema_26: point.close * (0.97 + Math.random() * 0.06),
        macd: (Math.random() - 0.5) * 100,
        volume_sma: point.volume * (0.8 + Math.random() * 0.4),
        bollinger_upper: point.close * 1.02,
        bollinger_lower: point.close * 0.98,
      },
    }));
  }

  /**
   * Create and initialize retrained model
   */
  private async createAndInitializeRetrainedModel(config: BacktestConfig): Promise<any> {
    const retrainedSystem = createRetrainedAITradingSystem();
    await retrainedSystem.initialize(config);
    return retrainedSystem;
  }

  /**
   * Test a specific model
   */
  private async testModel(
    strategy: any,
    marketData: any[],
    config: BacktestConfig,
    strategyName: string
  ): Promise<any> {
    
    if (strategy.initialize) {
      strategy.initialize(config);
    }

    const portfolioManager = new PortfolioManager(config);
    let signalCount = 0;
    let tradeCount = 0;

    // Run backtest
    for (let i = 0; i < marketData.length; i++) {
      const currentCandle = marketData[i];

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

      const signal = strategy.generateSignal(marketData, i);

      if (signal) {
        signalCount++;
        
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

      if (i % 24 === 0 || signal) {
        portfolioManager.createSnapshot(currentCandle.timestamp);
      }
    }

    const trades = portfolioManager.getTrades();
    const portfolioHistory = portfolioManager.getPortfolioHistory();
    const performance = PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, config);

    return {
      performance,
      trades,
      signals: signalCount,
    };
  }

  /**
   * Calculate improvements between models
   */
  private calculateImprovements(originalResults: any, retrainedResults: any): any {
    const returnImprovement = retrainedResults.performance.totalReturnPercent - originalResults.performance.totalReturnPercent;
    const sharpeImprovement = retrainedResults.performance.sharpeRatio - originalResults.performance.sharpeRatio;
    const drawdownImprovement = originalResults.performance.maxDrawdownPercent - retrainedResults.performance.maxDrawdownPercent;
    const winRateImprovement = retrainedResults.performance.winRate - originalResults.performance.winRate;
    const signalImprovement = retrainedResults.signals - originalResults.signals;

    return {
      returnImprovement,
      sharpeImprovement,
      drawdownImprovement,
      winRateImprovement,
      signalImprovement,
    };
  }

  /**
   * Analyze overall results across all configurations
   */
  private analyzeOverallResults(results: ModelComparisonResult[], startTime: number): void {
    const duration = (Date.now() - startTime) / 1000;

    logger.info('\n' + '🧠 RETRAINED MODEL BACKTEST RESULTS'.padStart(60, '='));
    logger.info('=' .repeat(120));

    // Overall Summary
    logger.info('📊 OVERALL COMPARISON SUMMARY:');
    logger.info(`   Test Duration: ${duration.toFixed(2)} seconds`);
    logger.info(`   Configurations Tested: ${results.length}`);
    logger.info(`   Data Source: Real Binance data (with fallback)`);

    // Performance Comparison Table
    logger.info('\n📈 PERFORMANCE COMPARISON:');
    logger.info('   Configuration           | Original Return | Retrained Return | Improvement | Original Sharpe | Retrained Sharpe | Sharpe Δ');
    logger.info('   ' + '-'.repeat(110));

    results.forEach(result => {
      const config = result.config.strategy.padEnd(23);
      const origReturn = result.originalModel.performance.totalReturnPercent.toFixed(2).padStart(15);
      const retrainedReturn = result.retrainedModel.performance.totalReturnPercent.toFixed(2).padStart(16);
      const returnImpr = (result.improvement.returnImprovement > 0 ? '+' : '') + result.improvement.returnImprovement.toFixed(2).padStart(11);
      const origSharpe = result.originalModel.performance.sharpeRatio.toFixed(3).padStart(15);
      const retrainedSharpe = result.retrainedModel.performance.sharpeRatio.toFixed(3).padStart(16);
      const sharpeImpr = (result.improvement.sharpeImprovement > 0 ? '+' : '') + result.improvement.sharpeImprovement.toFixed(3).padStart(8);

      logger.info(`   ${config} | ${origReturn}% | ${retrainedReturn}% | ${returnImpr}% | ${origSharpe} | ${retrainedSharpe} | ${sharpeImpr}`);
    });

    // Detailed Analysis
    logger.info('\n🔍 DETAILED ANALYSIS:');

    const avgReturnImprovement = results.reduce((sum, r) => sum + r.improvement.returnImprovement, 0) / results.length;
    const avgSharpeImprovement = results.reduce((sum, r) => sum + r.improvement.sharpeImprovement, 0) / results.length;
    const avgDrawdownImprovement = results.reduce((sum, r) => sum + r.improvement.drawdownImprovement, 0) / results.length;
    const avgSignalImprovement = results.reduce((sum, r) => sum + r.improvement.signalImprovement, 0) / results.length;

    logger.info(`   Average Return Improvement: ${avgReturnImprovement > 0 ? '+' : ''}${avgReturnImprovement.toFixed(2)}%`);
    logger.info(`   Average Sharpe Improvement: ${avgSharpeImprovement > 0 ? '+' : ''}${avgSharpeImprovement.toFixed(3)}`);
    logger.info(`   Average Drawdown Improvement: ${avgDrawdownImprovement > 0 ? '+' : ''}${avgDrawdownImprovement.toFixed(2)}%`);
    logger.info(`   Average Signal Improvement: ${avgSignalImprovement > 0 ? '+' : ''}${avgSignalImprovement.toFixed(0)} signals`);

    // Win/Loss Analysis
    const improvementWins = results.filter(r => r.improvement.returnImprovement > 0).length;
    const improvementLosses = results.length - improvementWins;

    logger.info(`\n🏆 IMPROVEMENT ANALYSIS:`);
    logger.info(`   Configurations with Better Returns: ${improvementWins}/${results.length} (${(improvementWins / results.length * 100).toFixed(1)}%)`);
    logger.info(`   Configurations with Worse Returns: ${improvementLosses}/${results.length} (${(improvementLosses / results.length * 100).toFixed(1)}%)`);

    // Best Performing Configuration
    const bestConfig = results.reduce((best, current) => 
      current.improvement.returnImprovement > best.improvement.returnImprovement ? current : best
    );

    logger.info(`\n🌟 BEST PERFORMING CONFIGURATION:`);
    logger.info(`   Configuration: ${bestConfig.config.strategy}`);
    logger.info(`   Return Improvement: ${bestConfig.improvement.returnImprovement > 0 ? '+' : ''}${bestConfig.improvement.returnImprovement.toFixed(2)}%`);
    logger.info(`   Sharpe Improvement: ${bestConfig.improvement.sharpeImprovement > 0 ? '+' : ''}${bestConfig.improvement.sharpeImprovement.toFixed(3)}`);
    logger.info(`   Signal Improvement: ${bestConfig.improvement.signalImprovement > 0 ? '+' : ''}${bestConfig.improvement.signalImprovement} signals`);

    // Model Effectiveness Assessment
    logger.info(`\n⭐ MODEL EFFECTIVENESS ASSESSMENT:`);
    if (avgReturnImprovement > 2) {
      logger.info(`   🎉 EXCELLENT: Retrained models show significant improvement (${avgReturnImprovement.toFixed(2)}% avg return boost)`);
    } else if (avgReturnImprovement > 0.5) {
      logger.info(`   ✅ GOOD: Retrained models show meaningful improvement (${avgReturnImprovement.toFixed(2)}% avg return boost)`);
    } else if (avgReturnImprovement > -0.5) {
      logger.info(`   ⚠️ NEUTRAL: Retrained models show mixed results (${avgReturnImprovement.toFixed(2)}% avg return change)`);
    } else {
      logger.info(`   ❌ POOR: Retrained models underperform original models (${avgReturnImprovement.toFixed(2)}% avg return loss)`);
    }

    logger.info('=' .repeat(120));
  }

  /**
   * Generate recommendations based on results
   */
  private generateRecommendations(results: ModelComparisonResult[]): void {
    logger.info('\n💡 RECOMMENDATIONS:');

    const avgImprovement = results.reduce((sum, r) => sum + r.improvement.returnImprovement, 0) / results.length;

    if (avgImprovement > 1) {
      logger.info('   🚀 DEPLOY RETRAINED MODELS: Significant performance improvement detected');
      logger.info('   📊 Monitor performance closely in live trading');
      logger.info('   🔄 Consider retraining models monthly with new data');
    } else if (avgImprovement > 0) {
      logger.info('   ✅ GRADUAL DEPLOYMENT: Modest improvement detected');
      logger.info('   🧪 Test with smaller position sizes initially');
      logger.info('   📈 Continue monitoring and optimization');
    } else {
      logger.info('   ⚠️ FURTHER OPTIMIZATION NEEDED: Limited or negative improvement');
      logger.info('   🔧 Review training data quality and feature engineering');
      logger.info('   📚 Consider additional model architectures');
    }

    logger.info('   🎯 Next Steps:');
    logger.info('     1. Validate results with longer backtesting periods');
    logger.info('     2. Test on different market conditions and volatility regimes');
    logger.info('     3. Implement ensemble methods combining best models');
    logger.info('     4. Set up automated model retraining pipeline');
    logger.info('     5. Begin paper trading with retrained models');
  }
}

/**
 * Main execution function
 */
async function main() {
  const runner = new RetrainedModelBacktestRunner();
  
  try {
    await runner.runComparativeBacktest();
  } catch (error) {
    logger.error('💥 Retrained model backtest failed:', error);
    process.exit(1);
  }
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { RetrainedModelBacktestRunner };
