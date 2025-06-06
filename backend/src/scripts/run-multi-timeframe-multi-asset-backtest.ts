#!/usr/bin/env node

/**
 * Multi-Timeframe Multi-Asset Comprehensive Backtesting Script
 * Executes comprehensive backtests combining multi-timeframe analysis with multi-asset portfolio optimization
 */

import { 
  createMultiTimeframeMultiAssetBacktester,
  MultiTimeframeMultiAssetBacktestResult
} from '../services/multiTimeframeMultiAssetBacktester';
import { 
  createMultiTimeframeMultiAssetConfigManager,
  BacktestScenario
} from '../services/multiTimeframeMultiAssetConfigManager';
import { logger } from '../utils/logger';

class MultiTimeframeMultiAssetBacktestRunner {
  private backtester = createMultiTimeframeMultiAssetBacktester();
  private configManager = createMultiTimeframeMultiAssetConfigManager();

  /**
   * Run comprehensive multi-timeframe multi-asset backtesting suite
   */
  public async runComprehensiveBacktest(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🔄 Starting Comprehensive Multi-Timeframe Multi-Asset Backtest Suite...');
      logger.info('📊 Features: Multi-timeframe analysis + Multi-asset portfolios + Hierarchical decisions');

      // Step 1: Define test scenarios
      const testScenarios = this.defineTestScenarios();

      // Step 2: Execute backtests for each scenario
      const results: MultiTimeframeMultiAssetBacktestResult[] = [];

      for (const scenario of testScenarios) {
        logger.info(`\n🔬 Testing scenario: ${scenario}`);
        const result = await this.runSingleScenario(scenario);
        if (result) {
          results.push(result);
        }
      }

      // Step 3: Generate comprehensive analysis
      this.generateComprehensiveAnalysis(results, startTime);

      // Step 4: Generate recommendations
      this.generateStrategicRecommendations(results);

      const duration = (Date.now() - startTime) / 1000;
      logger.info(`🎉 Comprehensive multi-timeframe multi-asset backtest completed in ${duration.toFixed(2)} seconds`);

    } catch (error) {
      logger.error('❌ Comprehensive backtest failed:', error);
      throw error;
    }
  }

  /**
   * Define test scenarios for comprehensive backtesting
   */
  private defineTestScenarios(): BacktestScenario[] {
    return [
      // Single Asset Scenarios
      'CONSERVATIVE_SINGLE_ASSET',
      'AGGRESSIVE_SINGLE_ASSET',
      
      // Multi-Asset Portfolio Scenarios
      'CONSERVATIVE_MULTI_ASSET',
      'AGGRESSIVE_MULTI_ASSET',
      'BALANCED_PORTFOLIO',
      
      // Specialized Strategy Scenarios
      'HIGH_FREQUENCY_SCALPING',
      'SWING_TRADING',
      'TREND_FOLLOWING',
      'MEAN_REVERSION',
      'CORRELATION_ARBITRAGE',
    ];
  }

  /**
   * Run backtest for a single scenario
   */
  private async runSingleScenario(scenario: BacktestScenario): Promise<MultiTimeframeMultiAssetBacktestResult | null> {
    try {
      // Create configuration for this scenario
      const endDate = new Date();
      const startDate = new Date(endDate.getTime() - (14 * 24 * 60 * 60 * 1000)); // 14 days

      const config = this.configManager.createBacktestConfig(scenario, startDate, endDate);
      
      if (!config) {
        logger.error(`❌ Failed to create config for scenario: ${scenario}`);
        return null;
      }

      // Validate configuration
      const scenarioConfig = this.configManager.getScenarioConfig(scenario);
      if (scenarioConfig) {
        const validation = this.configManager.validateScenarioConfig(scenarioConfig);
        if (!validation.valid) {
          logger.error(`❌ Invalid configuration for ${scenario}:`, validation.errors);
          return null;
        }
      }

      logger.info(`📊 Running ${scenario} backtest...`, {
        assets: config.assetConfigs.map(c => c.asset),
        timeframes: config.assetConfigs.flatMap(c => c.timeframes),
        primaryTimeframe: config.primaryTimeframe,
        portfolioMode: config.portfolioMode,
        capital: config.initialCapital,
      });

      // Execute backtest
      const result = await this.backtester.runBacktest(config);

      logger.info(`✅ ${scenario} completed`, {
        totalReturn: `${result.overallPerformance.totalReturn.toFixed(2)}%`,
        sharpeRatio: result.overallPerformance.sharpeRatio.toFixed(3),
        totalTrades: result.overallPerformance.totalTrades,
        duration: `${result.duration.toFixed(2)}s`,
      });

      return result;

    } catch (error) {
      logger.error(`❌ Failed to run scenario ${scenario}:`, error);
      return null;
    }
  }

  /**
   * Generate comprehensive analysis of all results
   */
  private generateComprehensiveAnalysis(
    results: MultiTimeframeMultiAssetBacktestResult[],
    startTime: number
  ): void {
    const totalDuration = (Date.now() - startTime) / 1000;

    logger.info('\n' + '🔄 MULTI-TIMEFRAME MULTI-ASSET BACKTEST RESULTS'.padStart(80, '='));
    logger.info('=' .repeat(160));

    // Overall Summary
    logger.info('📊 COMPREHENSIVE BACKTEST SUMMARY:');
    logger.info(`   Total Test Duration: ${totalDuration.toFixed(2)} seconds (${(totalDuration / 60).toFixed(1)} minutes)`);
    logger.info(`   Scenarios Tested: ${results.length}`);
    logger.info(`   Features: Multi-timeframe analysis + Multi-asset portfolios + Hierarchical decisions`);
    logger.info(`   Data Source: Real Binance data with enhanced fallback`);

    // Performance Summary Table
    logger.info('\n📈 SCENARIO PERFORMANCE SUMMARY:');
    logger.info('   Scenario                    | Return  | Sharpe | Drawdown | Win Rate | Trades | Assets | Timeframes');
    logger.info('   ' + '-'.repeat(130));

    results.forEach(result => {
      const scenario = result.config.strategy.padEnd(27);
      const totalReturn = result.overallPerformance.totalReturn.toFixed(2).padStart(7);
      const sharpeRatio = result.overallPerformance.sharpeRatio.toFixed(3).padStart(6);
      const maxDrawdown = result.overallPerformance.maxDrawdown.toFixed(2).padStart(8);
      const winRate = result.overallPerformance.winRate.toFixed(1).padStart(8);
      const totalTrades = result.overallPerformance.totalTrades.toString().padStart(6);
      const assetCount = result.config.assetConfigs.length.toString().padStart(6);
      const timeframeCount = new Set(result.config.assetConfigs.flatMap(c => c.timeframes)).size.toString().padStart(10);

      logger.info(`   ${scenario} | ${totalReturn}% | ${sharpeRatio} | ${maxDrawdown}% | ${winRate}% | ${totalTrades} | ${assetCount} | ${timeframeCount}`);
    });

    // Multi-Timeframe Analysis
    logger.info('\n🕐 MULTI-TIMEFRAME ANALYSIS:');
    this.analyzeTimeframePerformance(results);

    // Multi-Asset Analysis
    logger.info('\n🪙 MULTI-ASSET ANALYSIS:');
    this.analyzeMultiAssetPerformance(results);

    // Hierarchical Decision Analysis
    logger.info('\n🎯 HIERARCHICAL DECISION ANALYSIS:');
    this.analyzeHierarchicalDecisions(results);

    // Cross-Asset Correlation Analysis
    logger.info('\n🔗 CROSS-ASSET CORRELATION ANALYSIS:');
    this.analyzeCrossAssetCorrelations(results);

    // Best Performing Configurations
    logger.info('\n🌟 BEST PERFORMING CONFIGURATIONS:');
    this.analyzeBestPerformers(results);

    logger.info('=' .repeat(160));
  }

  /**
   * Analyze timeframe performance across scenarios
   */
  private analyzeTimeframePerformance(results: MultiTimeframeMultiAssetBacktestResult[]): void {
    const timeframeStats: { [timeframe: string]: { returns: number[]; trades: number[] } } = {};

    results.forEach(result => {
      result.assetTimeframePerformance.forEach(assetPerf => {
        assetPerf.timeframePerformance.forEach(tfPerf => {
          if (!timeframeStats[tfPerf.timeframe]) {
            timeframeStats[tfPerf.timeframe] = { returns: [], trades: [] };
          }
          timeframeStats[tfPerf.timeframe].returns.push(tfPerf.totalReturn);
          timeframeStats[tfPerf.timeframe].trades.push(tfPerf.totalTrades);
        });
      });
    });

    logger.info('   Timeframe | Avg Return | Best Return | Avg Trades | Total Scenarios');
    logger.info('   ' + '-'.repeat(70));

    Object.entries(timeframeStats).forEach(([timeframe, stats]) => {
      const avgReturn = stats.returns.reduce((sum, ret) => sum + ret, 0) / stats.returns.length;
      const bestReturn = Math.max(...stats.returns);
      const avgTrades = stats.trades.reduce((sum, trades) => sum + trades, 0) / stats.trades.length;
      const scenarios = stats.returns.length;

      logger.info(`   ${timeframe.padEnd(9)} | ${avgReturn.toFixed(2).padStart(10)}% | ${bestReturn.toFixed(2).padStart(11)}% | ${avgTrades.toFixed(0).padStart(10)} | ${scenarios.toString().padStart(15)}`);
    });

    // Identify best performing timeframe
    const bestTimeframe = Object.entries(timeframeStats).reduce((best, [timeframe, stats]) => {
      const avgReturn = stats.returns.reduce((sum, ret) => sum + ret, 0) / stats.returns.length;
      return avgReturn > best.avgReturn ? { timeframe, avgReturn } : best;
    }, { timeframe: '', avgReturn: -Infinity });

    logger.info(`\n   🏆 Best Performing Timeframe: ${bestTimeframe.timeframe} (${bestTimeframe.avgReturn.toFixed(2)}% avg return)`);
  }

  /**
   * Analyze multi-asset performance
   */
  private analyzeMultiAssetPerformance(results: MultiTimeframeMultiAssetBacktestResult[]): void {
    const singleAssetResults = results.filter(r => r.config.portfolioMode === 'SINGLE_ASSET');
    const multiAssetResults = results.filter(r => r.config.portfolioMode === 'MULTI_ASSET');
    const dynamicResults = results.filter(r => r.config.portfolioMode === 'DYNAMIC');

    const avgSingleAssetReturn = singleAssetResults.reduce((sum, r) => sum + r.overallPerformance.totalReturn, 0) / Math.max(singleAssetResults.length, 1);
    const avgMultiAssetReturn = multiAssetResults.reduce((sum, r) => sum + r.overallPerformance.totalReturn, 0) / Math.max(multiAssetResults.length, 1);
    const avgDynamicReturn = dynamicResults.reduce((sum, r) => sum + r.overallPerformance.totalReturn, 0) / Math.max(dynamicResults.length, 1);

    logger.info(`   Single Asset Average Return: ${avgSingleAssetReturn.toFixed(2)}% (${singleAssetResults.length} scenarios)`);
    logger.info(`   Multi-Asset Average Return: ${avgMultiAssetReturn.toFixed(2)}% (${multiAssetResults.length} scenarios)`);
    logger.info(`   Dynamic Portfolio Average Return: ${avgDynamicReturn.toFixed(2)}% (${dynamicResults.length} scenarios)`);

    // Calculate diversification benefit
    const diversificationBenefit = avgMultiAssetReturn - avgSingleAssetReturn;
    logger.info(`\n   📊 Diversification Benefit: ${diversificationBenefit > 0 ? '+' : ''}${diversificationBenefit.toFixed(2)}%`);

    if (diversificationBenefit > 2) {
      logger.info('   🎉 EXCELLENT: Multi-asset approach shows significant benefits');
    } else if (diversificationBenefit > 0) {
      logger.info('   ✅ GOOD: Multi-asset approach shows positive benefits');
    } else {
      logger.info('   ⚠️ NEUTRAL: Single asset strategies may be preferred');
    }
  }

  /**
   * Analyze hierarchical decision effectiveness
   */
  private analyzeHierarchicalDecisions(results: MultiTimeframeMultiAssetBacktestResult[]): void {
    const hierarchicalStats = results.map(r => r.hierarchicalAnalysis);
    
    const avgConsensusAccuracy = hierarchicalStats.reduce((sum, h) => sum + h.timeframeConsensusAccuracy, 0) / hierarchicalStats.length;
    const avgHigherTimeframeWinRate = hierarchicalStats.reduce((sum, h) => sum + h.higherTimeframeWinRate, 0) / hierarchicalStats.length;
    const avgConflictResolution = hierarchicalStats.reduce((sum, h) => sum + h.conflictResolutionSuccess, 0) / hierarchicalStats.length;

    logger.info(`   Average Timeframe Consensus Accuracy: ${avgConsensusAccuracy.toFixed(1)}%`);
    logger.info(`   Average Higher Timeframe Win Rate: ${avgHigherTimeframeWinRate.toFixed(1)}%`);
    logger.info(`   Average Conflict Resolution Success: ${avgConflictResolution.toFixed(1)}%`);

    if (avgConsensusAccuracy > 75) {
      logger.info('   🎯 EXCELLENT: Hierarchical decision making is highly effective');
    } else if (avgConsensusAccuracy > 65) {
      logger.info('   ✅ GOOD: Hierarchical decision making shows solid performance');
    } else {
      logger.info('   ⚠️ MODERATE: Hierarchical decision making needs optimization');
    }
  }

  /**
   * Analyze cross-asset correlations
   */
  private analyzeCrossAssetCorrelations(results: MultiTimeframeMultiAssetBacktestResult[]): void {
    const crossAssetStats = results.map(r => r.crossAssetAnalysis);
    
    const avgCorrelationBenefit = crossAssetStats.reduce((sum, c) => sum + c.correlationBenefit, 0) / crossAssetStats.length;
    const avgDiversificationRatio = crossAssetStats.reduce((sum, c) => sum + c.diversificationRatio, 0) / crossAssetStats.length;
    const avgPortfolioOptimization = crossAssetStats.reduce((sum, c) => sum + c.portfolioOptimizationGain, 0) / crossAssetStats.length;

    logger.info(`   Average Correlation Benefit: ${avgCorrelationBenefit.toFixed(1)}%`);
    logger.info(`   Average Diversification Ratio: ${avgDiversificationRatio.toFixed(2)}`);
    logger.info(`   Average Portfolio Optimization Gain: ${avgPortfolioOptimization.toFixed(1)}%`);

    if (avgCorrelationBenefit > 15) {
      logger.info('   🔗 EXCELLENT: Strong correlation benefits detected');
    } else if (avgCorrelationBenefit > 8) {
      logger.info('   ✅ GOOD: Moderate correlation benefits achieved');
    } else {
      logger.info('   ⚠️ LIMITED: Correlation benefits are minimal');
    }
  }

  /**
   * Analyze best performing configurations
   */
  private analyzeBestPerformers(results: MultiTimeframeMultiAssetBacktestResult[]): void {
    // Best overall return
    const bestReturn = results.reduce((best, current) => 
      current.overallPerformance.totalReturn > best.overallPerformance.totalReturn ? current : best
    );

    // Best Sharpe ratio
    const bestSharpe = results.reduce((best, current) => 
      current.overallPerformance.sharpeRatio > best.overallPerformance.sharpeRatio ? current : best
    );

    // Best win rate
    const bestWinRate = results.reduce((best, current) => 
      current.overallPerformance.winRate > best.overallPerformance.winRate ? current : best
    );

    logger.info(`   🏆 Best Total Return: ${bestReturn.config.strategy} (${bestReturn.overallPerformance.totalReturn.toFixed(2)}%)`);
    logger.info(`   📊 Best Sharpe Ratio: ${bestSharpe.config.strategy} (${bestSharpe.overallPerformance.sharpeRatio.toFixed(3)})`);
    logger.info(`   🎯 Best Win Rate: ${bestWinRate.config.strategy} (${bestWinRate.overallPerformance.winRate.toFixed(1)}%)`);

    // Analyze best configuration characteristics
    logger.info('\n   🔍 Best Configuration Analysis:');
    logger.info(`   Primary Timeframe: ${bestReturn.config.primaryTimeframe}`);
    logger.info(`   Portfolio Mode: ${bestReturn.config.portfolioMode}`);
    logger.info(`   Assets: ${bestReturn.config.assetConfigs.map(c => c.asset).join(', ')}`);
    logger.info(`   Rebalance Frequency: ${bestReturn.config.rebalanceFrequency}`);
  }

  /**
   * Generate strategic recommendations
   */
  private generateStrategicRecommendations(results: MultiTimeframeMultiAssetBacktestResult[]): void {
    logger.info('\n💡 STRATEGIC RECOMMENDATIONS:');

    const avgReturn = results.reduce((sum, r) => sum + r.overallPerformance.totalReturn, 0) / results.length;
    const avgSharpe = results.reduce((sum, r) => sum + r.overallPerformance.sharpeRatio, 0) / results.length;

    // Overall assessment
    if (avgReturn > 5 && avgSharpe > 1.5) {
      logger.info('   🚀 EXCELLENT PERFORMANCE: Deploy multi-timeframe multi-asset strategies');
      logger.info('   📊 High returns with good risk-adjusted performance detected');
    } else if (avgReturn > 2 && avgSharpe > 1.0) {
      logger.info('   ✅ GOOD PERFORMANCE: Multi-timeframe multi-asset approach is viable');
      logger.info('   🔧 Consider optimization for better risk-adjusted returns');
    } else {
      logger.info('   ⚠️ MODERATE PERFORMANCE: Further optimization needed');
      logger.info('   🔍 Review strategy parameters and market conditions');
    }

    // Specific recommendations
    logger.info('\n   🎯 IMPLEMENTATION RECOMMENDATIONS:');
    
    const multiAssetResults = results.filter(r => r.config.portfolioMode !== 'SINGLE_ASSET');
    const avgMultiAssetReturn = multiAssetResults.reduce((sum, r) => sum + r.overallPerformance.totalReturn, 0) / Math.max(multiAssetResults.length, 1);
    
    if (avgMultiAssetReturn > avgReturn) {
      logger.info('     1. 🪙 Prioritize multi-asset portfolio strategies');
      logger.info('     2. 📊 Implement dynamic asset allocation');
      logger.info('     3. 🔄 Use correlation-based rebalancing');
    }

    const hierarchicalResults = results.filter(r => r.hierarchicalAnalysis.timeframeConsensusAccuracy > 70);
    if (hierarchicalResults.length > results.length * 0.6) {
      logger.info('     4. 🎯 Deploy hierarchical timeframe decision making');
      logger.info('     5. ⏰ Prioritize higher timeframe signals');
      logger.info('     6. 🔍 Use lower timeframes for entry optimization');
    }

    logger.info('\n   🚀 NEXT STEPS:');
    logger.info('     1. Implement real-time multi-timeframe data feeds');
    logger.info('     2. Deploy portfolio optimization algorithms');
    logger.info('     3. Set up hierarchical signal processing');
    logger.info('     4. Monitor cross-asset correlation changes');
    logger.info('     5. Implement adaptive timeframe weighting');
    logger.info('     6. Begin paper trading with best configurations');
  }
}

/**
 * Main execution function
 */
async function main() {
  const runner = new MultiTimeframeMultiAssetBacktestRunner();
  
  try {
    await runner.runComprehensiveBacktest();
  } catch (error) {
    logger.error('💥 Comprehensive multi-timeframe multi-asset backtest failed:', error);
    process.exit(1);
  }
}

// Run if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { MultiTimeframeMultiAssetBacktestRunner };
