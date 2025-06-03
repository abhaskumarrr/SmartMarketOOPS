/**
 * Optimization Results Analyzer
 * Analyzes hyperparameter optimization results and provides insights
 */

import { OptimizationResult, ParameterConfig } from './hyperparameterOptimizer';
import { logger } from '../utils/logger';

export interface ParameterImpactAnalysis {
  parameter: string;
  correlation: number;
  significance: number;
  optimalRange: [number, number];
  impact: 'HIGH' | 'MEDIUM' | 'LOW';
}

export interface OptimizationSummary {
  totalConfigurations: number;
  bestConfiguration: OptimizationResult;
  top5Configurations: OptimizationResult[];
  parameterImpacts: ParameterImpactAnalysis[];
  performanceDistribution: {
    sharpeRatio: { min: number; max: number; mean: number; std: number };
    totalReturn: { min: number; max: number; mean: number; std: number };
    maxDrawdown: { min: number; max: number; mean: number; std: number };
  };
  insights: string[];
  recommendations: string[];
}

export class OptimizationAnalyzer {
  /**
   * Analyze optimization results and provide comprehensive insights
   */
  public analyzeResults(results: OptimizationResult[]): OptimizationSummary {
    logger.info('📊 Analyzing optimization results...', {
      totalResults: results.length,
    });

    const top5 = results.slice(0, 5);
    const best = results[0];

    const parameterImpacts = this.analyzeParameterImpacts(results);
    const performanceDistribution = this.calculatePerformanceDistribution(results);
    const insights = this.generateInsights(results, parameterImpacts);
    const recommendations = this.generateRecommendations(best, parameterImpacts);

    return {
      totalConfigurations: results.length,
      bestConfiguration: best,
      top5Configurations: top5,
      parameterImpacts,
      performanceDistribution,
      insights,
      recommendations,
    };
  }

  /**
   * Display comprehensive optimization report
   */
  public displayOptimizationReport(summary: OptimizationSummary): void {
    logger.info('\n' + '🔬 HYPERPARAMETER OPTIMIZATION REPORT'.padStart(60, '='));
    logger.info('=' .repeat(120));

    // Overview
    this.displayOverview(summary);

    // Top 5 Configurations
    this.displayTop5Configurations(summary.top5Configurations);

    // Parameter Impact Analysis
    this.displayParameterImpacts(summary.parameterImpacts);

    // Performance Distribution
    this.displayPerformanceDistribution(summary.performanceDistribution);

    // Best Configuration Details
    this.displayBestConfigurationDetails(summary.bestConfiguration);

    // Trade Analysis for Best Configuration
    this.displayTradeAnalysis(summary.bestConfiguration);

    // Insights and Recommendations
    this.displayInsightsAndRecommendations(summary.insights, summary.recommendations);

    logger.info('=' .repeat(120));
  }

  private displayOverview(summary: OptimizationSummary): void {
    const best = summary.bestConfiguration;
    
    logger.info('📊 OPTIMIZATION OVERVIEW:');
    logger.info(`   Total Configurations Tested: ${summary.totalConfigurations}`);
    logger.info(`   Best Sharpe Ratio: ${best.performance.sharpeRatio.toFixed(3)}`);
    logger.info(`   Best Total Return: ${best.performance.totalReturnPercent.toFixed(2)}%`);
    logger.info(`   Best Max Drawdown: ${best.performance.maxDrawdownPercent.toFixed(2)}%`);
    logger.info(`   Best Win Rate: ${best.performance.winRate.toFixed(1)}%`);
    logger.info(`   Best Configuration ID: ${best.config.id}`);
    logger.info(`   Optimization Score: ${best.score.toFixed(2)}/100`);
  }

  private displayTop5Configurations(top5: OptimizationResult[]): void {
    logger.info('\n🏆 TOP 5 CONFIGURATIONS:');
    logger.info('   Rank | Config ID      | Sharpe | Return% | Drawdown% | Win Rate% | Trades | Score');
    logger.info('   ' + '-'.repeat(85));

    top5.forEach((result, index) => {
      const p = result.performance;
      logger.info(`   ${(index + 1).toString().padStart(4)} | ${result.config.id.padEnd(14)} | ${p.sharpeRatio.toFixed(2).padStart(6)} | ${p.totalReturnPercent.toFixed(1).padStart(7)} | ${p.maxDrawdownPercent.toFixed(1).padStart(9)} | ${p.winRate.toFixed(1).padStart(9)} | ${p.totalTrades.toString().padStart(6)} | ${result.score.toFixed(1)}`);
    });
  }

  private displayParameterImpacts(impacts: ParameterImpactAnalysis[]): void {
    logger.info('\n📈 PARAMETER IMPACT ANALYSIS:');
    logger.info('   Parameter              | Impact | Correlation | Optimal Range        | Significance');
    logger.info('   ' + '-'.repeat(85));

    impacts.forEach(impact => {
      const optimalStr = `${impact.optimalRange[0]} - ${impact.optimalRange[1]}`;
      logger.info(`   ${impact.parameter.padEnd(22)} | ${impact.impact.padEnd(6)} | ${impact.correlation.toFixed(3).padStart(11)} | ${optimalStr.padEnd(20)} | ${impact.significance.toFixed(3)}`);
    });
  }

  private displayPerformanceDistribution(distribution: any): void {
    logger.info('\n📊 PERFORMANCE DISTRIBUTION:');
    
    logger.info('   Sharpe Ratio:');
    logger.info(`     Min: ${distribution.sharpeRatio.min.toFixed(3)}, Max: ${distribution.sharpeRatio.max.toFixed(3)}, Mean: ${distribution.sharpeRatio.mean.toFixed(3)}, Std: ${distribution.sharpeRatio.std.toFixed(3)}`);
    
    logger.info('   Total Return (%):');
    logger.info(`     Min: ${distribution.totalReturn.min.toFixed(2)}, Max: ${distribution.totalReturn.max.toFixed(2)}, Mean: ${distribution.totalReturn.mean.toFixed(2)}, Std: ${distribution.totalReturn.std.toFixed(2)}`);
    
    logger.info('   Max Drawdown (%):');
    logger.info(`     Min: ${distribution.maxDrawdown.min.toFixed(2)}, Max: ${distribution.maxDrawdown.max.toFixed(2)}, Mean: ${distribution.maxDrawdown.mean.toFixed(2)}, Std: ${distribution.maxDrawdown.std.toFixed(2)}`);
  }

  private displayBestConfigurationDetails(best: OptimizationResult): void {
    logger.info('\n🎯 BEST CONFIGURATION DETAILS:');
    logger.info(`   Configuration ID: ${best.config.id}`);
    logger.info(`   Optimization Score: ${best.score.toFixed(2)}/100`);
    logger.info('');
    logger.info('   📋 Parameter Settings:');
    logger.info(`     Min Confidence: ${best.config.minConfidence}%`);
    logger.info(`     Model Consensus: ${best.config.modelConsensus}`);
    logger.info(`     Decision Cooldown: ${best.config.decisionCooldown} minutes`);
    logger.info(`     Risk Per Trade: ${best.config.riskPerTrade}%`);
    logger.info(`     Stop Loss: ${best.config.stopLossPercent}%`);
    logger.info(`     Take Profit Multiplier: ${best.config.takeProfitMultiplier}x`);
    logger.info(`     Position Size Multiplier: ${best.config.positionSizeMultiplier}`);
    logger.info(`     Trend Threshold: ${best.config.trendThreshold}`);
    logger.info(`     Volatility Threshold: ${best.config.volatilityThreshold}`);
    logger.info('');
    logger.info('   📊 Performance Metrics:');
    logger.info(`     Total Return: ${best.performance.totalReturnPercent.toFixed(2)}%`);
    logger.info(`     Sharpe Ratio: ${best.performance.sharpeRatio.toFixed(3)}`);
    logger.info(`     Calmar Ratio: ${best.performance.calmarRatio.toFixed(3)}`);
    logger.info(`     Maximum Drawdown: ${best.performance.maxDrawdownPercent.toFixed(2)}%`);
    logger.info(`     Win Rate: ${best.performance.winRate.toFixed(1)}%`);
    logger.info(`     Profit Factor: ${best.performance.profitFactor.toFixed(2)}`);
    logger.info(`     Total Trades: ${best.performance.totalTrades}`);
    logger.info(`     Average Win: $${best.performance.averageWin.toFixed(2)}`);
    logger.info(`     Average Loss: $${best.performance.averageLoss.toFixed(2)}`);
    logger.info(`     Volatility: ${best.performance.volatility.toFixed(2)}%`);
  }

  private displayTradeAnalysis(best: OptimizationResult): void {
    if (best.trades.length === 0) {
      logger.info('\n📋 TRADE ANALYSIS: No trades executed');
      return;
    }

    logger.info('\n📋 TRADE-BY-TRADE ANALYSIS (Best Configuration):');
    logger.info('   Trade | Side  | Entry Price | Exit Price | P&L      | P&L%    | Duration | Reason');
    logger.info('   ' + '-'.repeat(90));

    best.trades.forEach((trade, index) => {
      const duration = (trade.duration / (1000 * 60 * 60)).toFixed(1);
      const reason = trade.reason.length > 20 ? trade.reason.substring(0, 17) + '...' : trade.reason;
      
      logger.info(`   ${(index + 1).toString().padStart(5)} | ${trade.side.padEnd(5)} | ${trade.entryPrice.toFixed(2).padStart(11)} | ${trade.exitPrice.toFixed(2).padStart(10)} | ${trade.pnl.toFixed(2).padStart(8)} | ${trade.pnlPercent.toFixed(2).padStart(7)} | ${duration.padStart(8)}h | ${reason}`);
    });

    // Trade statistics
    const winningTrades = best.trades.filter(t => t.pnl > 0);
    const losingTrades = best.trades.filter(t => t.pnl <= 0);
    
    logger.info('\n   📊 Trade Statistics:');
    logger.info(`     Total Trades: ${best.trades.length}`);
    logger.info(`     Winning Trades: ${winningTrades.length} (${(winningTrades.length / best.trades.length * 100).toFixed(1)}%)`);
    logger.info(`     Losing Trades: ${losingTrades.length} (${(losingTrades.length / best.trades.length * 100).toFixed(1)}%)`);
    
    if (winningTrades.length > 0) {
      const avgWinDuration = winningTrades.reduce((sum, t) => sum + t.duration, 0) / winningTrades.length / (1000 * 60 * 60);
      logger.info(`     Average Winning Trade Duration: ${avgWinDuration.toFixed(1)} hours`);
    }
    
    if (losingTrades.length > 0) {
      const avgLossDuration = losingTrades.reduce((sum, t) => sum + t.duration, 0) / losingTrades.length / (1000 * 60 * 60);
      logger.info(`     Average Losing Trade Duration: ${avgLossDuration.toFixed(1)} hours`);
    }
  }

  private displayInsightsAndRecommendations(insights: string[], recommendations: string[]): void {
    logger.info('\n💡 KEY INSIGHTS:');
    insights.forEach((insight, index) => {
      logger.info(`   ${index + 1}. ${insight}`);
    });

    logger.info('\n🚀 RECOMMENDATIONS:');
    recommendations.forEach((recommendation, index) => {
      logger.info(`   ${index + 1}. ${recommendation}`);
    });
  }

  /**
   * Analyze parameter impacts on performance
   */
  private analyzeParameterImpacts(results: OptimizationResult[]): ParameterImpactAnalysis[] {
    const parameters = [
      'minConfidence',
      'modelConsensus', 
      'decisionCooldown',
      'riskPerTrade',
      'stopLossPercent',
      'takeProfitMultiplier',
      'positionSizeMultiplier',
      'trendThreshold',
      'volatilityThreshold'
    ];

    const impacts: ParameterImpactAnalysis[] = [];

    parameters.forEach(param => {
      const correlation = this.calculateCorrelation(results, param, 'sharpeRatio');
      const significance = Math.abs(correlation);
      const optimalRange = this.findOptimalRange(results, param);
      
      let impact: 'HIGH' | 'MEDIUM' | 'LOW' = 'LOW';
      if (significance > 0.3) impact = 'HIGH';
      else if (significance > 0.15) impact = 'MEDIUM';

      impacts.push({
        parameter: param,
        correlation,
        significance,
        optimalRange,
        impact,
      });
    });

    // Sort by significance (descending)
    return impacts.sort((a, b) => b.significance - a.significance);
  }

  /**
   * Calculate correlation between parameter and performance metric
   */
  private calculateCorrelation(results: OptimizationResult[], parameter: string, metric: string): number {
    const paramValues = results.map(r => (r.config as any)[parameter]);
    const metricValues = results.map(r => (r.performance as any)[metric]);

    const n = paramValues.length;
    const sumX = paramValues.reduce((a, b) => a + b, 0);
    const sumY = metricValues.reduce((a, b) => a + b, 0);
    const sumXY = paramValues.reduce((sum, x, i) => sum + x * metricValues[i], 0);
    const sumXX = paramValues.reduce((sum, x) => sum + x * x, 0);
    const sumYY = metricValues.reduce((sum, y) => sum + y * y, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Find optimal range for a parameter
   */
  private findOptimalRange(results: OptimizationResult[], parameter: string): [number, number] {
    // Get top 20% of results
    const topResults = results.slice(0, Math.ceil(results.length * 0.2));
    const paramValues = topResults.map(r => (r.config as any)[parameter]);
    
    const min = Math.min(...paramValues);
    const max = Math.max(...paramValues);
    
    return [min, max];
  }

  /**
   * Calculate performance distribution statistics
   */
  private calculatePerformanceDistribution(results: OptimizationResult[]): any {
    const metrics = ['sharpeRatio', 'totalReturnPercent', 'maxDrawdownPercent'];
    const distribution: any = {};

    metrics.forEach(metric => {
      const values = results.map(r => (r.performance as any)[metric]);
      
      distribution[metric.replace('Percent', '')] = {
        min: Math.min(...values),
        max: Math.max(...values),
        mean: values.reduce((a, b) => a + b, 0) / values.length,
        std: this.calculateStandardDeviation(values),
      };
    });

    return distribution;
  }

  /**
   * Calculate standard deviation
   */
  private calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  /**
   * Generate insights from optimization results
   */
  private generateInsights(results: OptimizationResult[], impacts: ParameterImpactAnalysis[]): string[] {
    const insights: string[] = [];
    const best = results[0];
    const worst = results[results.length - 1];

    // Performance range insights
    const sharpeRange = best.performance.sharpeRatio - worst.performance.sharpeRatio;
    insights.push(`Sharpe ratio varied by ${sharpeRange.toFixed(3)} across configurations (${worst.performance.sharpeRatio.toFixed(3)} to ${best.performance.sharpeRatio.toFixed(3)})`);

    const returnRange = best.performance.totalReturnPercent - worst.performance.totalReturnPercent;
    insights.push(`Total return varied by ${returnRange.toFixed(1)}% across configurations (${worst.performance.totalReturnPercent.toFixed(1)}% to ${best.performance.totalReturnPercent.toFixed(1)}%)`);

    // Parameter impact insights
    const highImpactParams = impacts.filter(i => i.impact === 'HIGH');
    if (highImpactParams.length > 0) {
      insights.push(`High-impact parameters: ${highImpactParams.map(p => p.parameter).join(', ')}`);
    }

    const strongestCorrelation = impacts[0];
    insights.push(`${strongestCorrelation.parameter} showed strongest correlation with performance (${strongestCorrelation.correlation.toFixed(3)})`);

    // Configuration insights
    if (best.performance.totalTrades > 0) {
      insights.push(`Best configuration executed ${best.performance.totalTrades} trades with ${best.performance.winRate.toFixed(1)}% win rate`);
    } else {
      insights.push(`Best configuration avoided trading, preserving capital in unfavorable market conditions`);
    }

    // Risk insights
    if (best.performance.maxDrawdownPercent < 5) {
      insights.push(`Excellent risk control achieved with maximum drawdown of only ${best.performance.maxDrawdownPercent.toFixed(2)}%`);
    }

    return insights;
  }

  /**
   * Generate recommendations based on optimization results
   */
  private generateRecommendations(best: OptimizationResult, impacts: ParameterImpactAnalysis[]): string[] {
    const recommendations: string[] = [];

    // Parameter-specific recommendations
    const highImpactParams = impacts.filter(i => i.impact === 'HIGH');
    highImpactParams.forEach(param => {
      recommendations.push(`Focus on optimizing ${param.parameter} within range ${param.optimalRange[0]} - ${param.optimalRange[1]} for maximum impact`);
    });

    // Performance-based recommendations
    if (best.performance.sharpeRatio > 1) {
      recommendations.push(`Excellent risk-adjusted returns achieved - consider live testing with small position sizes`);
    }

    if (best.performance.totalReturnPercent > 10) {
      recommendations.push(`Strong absolute returns - validate with out-of-sample data before deployment`);
    }

    if (best.performance.totalTrades < 5) {
      recommendations.push(`Low trade frequency - consider testing on longer time periods or multiple assets`);
    }

    // Configuration recommendations
    recommendations.push(`Use configuration ${best.config.id} as baseline for further optimization`);
    recommendations.push(`Test best parameters on different market regimes and time periods`);
    recommendations.push(`Consider ensemble approach combining top 3-5 configurations`);

    return recommendations;
  }
}

// Export factory function
export function createOptimizationAnalyzer(): OptimizationAnalyzer {
  return new OptimizationAnalyzer();
}
