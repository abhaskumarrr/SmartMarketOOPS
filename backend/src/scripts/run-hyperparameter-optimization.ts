#!/usr/bin/env node

/**
 * Comprehensive Hyperparameter Optimization Script
 * Systematically tests parameter combinations to maximize trading performance
 */

import { createHyperparameterOptimizer } from '../services/hyperparameterOptimizer';
import { createOptimizationAnalyzer } from '../services/optimizationAnalyzer';
import { logger } from '../utils/logger';
import { redisStreamsService } from '../services/redisStreamsService';
import fs from 'fs';
import path from 'path';

class HyperparameterOptimizationRunner {
  /**
   * Run comprehensive hyperparameter optimization
   */
  public async runOptimization(): Promise<void> {
    const startTime = Date.now();
    
    try {
      logger.info('🔬 Starting Comprehensive Hyperparameter Optimization...');
      logger.info('🎯 Target Metrics: Sharpe Ratio (Primary), Total Return (Secondary), Max Drawdown (Tertiary)');

      // Initialize infrastructure
      await this.initializeInfrastructure();

      // Create optimizer and analyzer
      const optimizer = createHyperparameterOptimizer();
      const analyzer = createOptimizationAnalyzer();

      // Configuration for optimization
      const numIterations = this.getOptimizationIterations();
      
      logger.info(`📊 Configuration:`, {
        iterations: numIterations,
        methodology: 'Grid Search + Random Search',
        dataset: '30-day BTCUSD hourly data',
        targetMetrics: ['Sharpe Ratio', 'Total Return %', 'Max Drawdown %'],
      });

      // Run optimization
      logger.info('🚀 Starting optimization process...');
      const results = await optimizer.runOptimization(numIterations);

      if (results.length === 0) {
        throw new Error('No valid optimization results generated');
      }

      // Analyze results
      logger.info('📊 Analyzing optimization results...');
      const summary = analyzer.analyzeResults(results);

      // Display comprehensive report
      analyzer.displayOptimizationReport(summary);

      // Save results to file
      await this.saveOptimizationResults(results, summary);

      // Display execution summary
      this.displayExecutionSummary(results, startTime);

      logger.info('🎉 Hyperparameter optimization completed successfully!');

    } catch (error) {
      logger.error('❌ Hyperparameter optimization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize infrastructure
   */
  private async initializeInfrastructure(): Promise<void> {
    try {
      await redisStreamsService.initialize();
      logger.info('✅ Redis Streams initialized for optimization');
    } catch (error) {
      logger.warn('⚠️ Redis Streams initialization failed, continuing without it');
    }
  }

  /**
   * Get number of optimization iterations based on environment
   */
  private getOptimizationIterations(): number {
    // Check for environment variable or command line argument
    const envIterations = process.env.OPTIMIZATION_ITERATIONS;
    const argIterations = process.argv.find(arg => arg.startsWith('--iterations='));
    
    if (argIterations) {
      return parseInt(argIterations.split('=')[1]) || 100;
    }
    
    if (envIterations) {
      return parseInt(envIterations) || 100;
    }

    // Default based on available time/resources
    return 100; // Comprehensive optimization
  }

  /**
   * Save optimization results to files
   */
  private async saveOptimizationResults(results: any[], summary: any): Promise<void> {
    try {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const resultsDir = path.join(process.cwd(), 'optimization_results');
      
      // Create results directory if it doesn't exist
      if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
      }

      // Save detailed results
      const detailedResultsPath = path.join(resultsDir, `optimization_results_${timestamp}.json`);
      fs.writeFileSync(detailedResultsPath, JSON.stringify(results, null, 2));

      // Save summary
      const summaryPath = path.join(resultsDir, `optimization_summary_${timestamp}.json`);
      fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));

      // Save top 5 configurations in CSV format
      const csvPath = path.join(resultsDir, `top_configurations_${timestamp}.csv`);
      this.saveTop5AsCSV(summary.top5Configurations, csvPath);

      // Save best configuration parameters
      const bestConfigPath = path.join(resultsDir, `best_configuration_${timestamp}.json`);
      fs.writeFileSync(bestConfigPath, JSON.stringify(summary.bestConfiguration.config, null, 2));

      logger.info('💾 Optimization results saved:', {
        detailedResults: detailedResultsPath,
        summary: summaryPath,
        topConfigurations: csvPath,
        bestConfiguration: bestConfigPath,
      });

    } catch (error) {
      logger.error('❌ Failed to save optimization results:', error);
    }
  }

  /**
   * Save top 5 configurations as CSV
   */
  private saveTop5AsCSV(top5: any[], csvPath: string): void {
    const headers = [
      'Rank',
      'Config_ID',
      'Sharpe_Ratio',
      'Total_Return_Percent',
      'Max_Drawdown_Percent',
      'Win_Rate',
      'Total_Trades',
      'Profit_Factor',
      'Score',
      'Min_Confidence',
      'Model_Consensus',
      'Decision_Cooldown',
      'Risk_Per_Trade',
      'Stop_Loss_Percent',
      'Take_Profit_Multiplier',
      'Position_Size_Multiplier',
      'Trend_Threshold',
      'Volatility_Threshold'
    ];

    const rows = top5.map((result, index) => [
      index + 1,
      result.config.id,
      result.performance.sharpeRatio.toFixed(4),
      result.performance.totalReturnPercent.toFixed(2),
      result.performance.maxDrawdownPercent.toFixed(2),
      result.performance.winRate.toFixed(1),
      result.performance.totalTrades,
      result.performance.profitFactor.toFixed(2),
      result.score.toFixed(2),
      result.config.minConfidence,
      result.config.modelConsensus,
      result.config.decisionCooldown,
      result.config.riskPerTrade,
      result.config.stopLossPercent,
      result.config.takeProfitMultiplier,
      result.config.positionSizeMultiplier,
      result.config.trendThreshold,
      result.config.volatilityThreshold
    ]);

    const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
    fs.writeFileSync(csvPath, csvContent);
  }

  /**
   * Display execution summary
   */
  private displayExecutionSummary(results: any[], startTime: number): void {
    const duration = (Date.now() - startTime) / 1000;
    const best = results[0];
    const worst = results[results.length - 1];

    logger.info('\n' + '⚡ OPTIMIZATION EXECUTION SUMMARY'.padStart(50, '='));
    logger.info('=' .repeat(80));

    logger.info('🔬 EXECUTION METRICS:');
    logger.info(`   Total Duration: ${duration.toFixed(1)} seconds (${(duration / 60).toFixed(1)} minutes)`);
    logger.info(`   Configurations Tested: ${results.length}`);
    logger.info(`   Average Time per Configuration: ${(duration / results.length).toFixed(2)} seconds`);
    logger.info(`   Optimization Speed: ${(results.length / duration * 60).toFixed(1)} configs/minute`);

    logger.info('\n📊 PERFORMANCE RANGE:');
    logger.info(`   Sharpe Ratio Range: ${worst.performance.sharpeRatio.toFixed(3)} to ${best.performance.sharpeRatio.toFixed(3)}`);
    logger.info(`   Return Range: ${worst.performance.totalReturnPercent.toFixed(1)}% to ${best.performance.totalReturnPercent.toFixed(1)}%`);
    logger.info(`   Drawdown Range: ${best.performance.maxDrawdownPercent.toFixed(1)}% to ${worst.performance.maxDrawdownPercent.toFixed(1)}%`);

    logger.info('\n🎯 OPTIMIZATION SUCCESS:');
    const improvementVsBaseline = ((best.performance.sharpeRatio - 0) * 100).toFixed(1);
    logger.info(`   Best Sharpe Ratio: ${best.performance.sharpeRatio.toFixed(3)} (${improvementVsBaseline}% improvement vs baseline)`);
    logger.info(`   Best Configuration: ${best.config.id}`);
    logger.info(`   Optimization Score: ${best.score.toFixed(2)}/100`);

    if (best.performance.sharpeRatio > 1) {
      logger.info('   ✅ Excellent risk-adjusted returns achieved');
    }
    if (best.performance.totalReturnPercent > 10) {
      logger.info('   ✅ Strong absolute returns achieved');
    }
    if (best.performance.maxDrawdownPercent < 10) {
      logger.info('   ✅ Good risk control maintained');
    }

    logger.info('\n🚀 NEXT STEPS:');
    logger.info('   1. Validate best configuration on out-of-sample data');
    logger.info('   2. Test on different market conditions and time periods');
    logger.info('   3. Consider ensemble approach with top configurations');
    logger.info('   4. Implement walk-forward optimization for robustness');
    logger.info('   5. Begin paper trading with optimized parameters');

    logger.info('=' .repeat(80));
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
  const runner = new HyperparameterOptimizationRunner();
  
  try {
    await runner.runOptimization();
  } catch (error) {
    logger.error('💥 Hyperparameter optimization failed:', error);
    process.exit(1);
  } finally {
    await runner.cleanup();
  }
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  logger.info('🛑 Received SIGINT, cleaning up...');
  const runner = new HyperparameterOptimizationRunner();
  await runner.cleanup();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  logger.info('🛑 Received SIGTERM, cleaning up...');
  const runner = new HyperparameterOptimizationRunner();
  await runner.cleanup();
  process.exit(0);
});

// Run if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { HyperparameterOptimizationRunner };
