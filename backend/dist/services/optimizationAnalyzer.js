"use strict";
/**
 * Optimization Results Analyzer
 * Analyzes hyperparameter optimization results and provides insights
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.OptimizationAnalyzer = void 0;
exports.createOptimizationAnalyzer = createOptimizationAnalyzer;
const logger_1 = require("../utils/logger");
class OptimizationAnalyzer {
    /**
     * Analyze optimization results and provide comprehensive insights
     */
    analyzeResults(results) {
        logger_1.logger.info('📊 Analyzing optimization results...', {
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
    displayOptimizationReport(summary) {
        logger_1.logger.info('\n' + '🔬 HYPERPARAMETER OPTIMIZATION REPORT'.padStart(60, '='));
        logger_1.logger.info('='.repeat(120));
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
        logger_1.logger.info('='.repeat(120));
    }
    displayOverview(summary) {
        const best = summary.bestConfiguration;
        logger_1.logger.info('📊 OPTIMIZATION OVERVIEW:');
        logger_1.logger.info(`   Total Configurations Tested: ${summary.totalConfigurations}`);
        logger_1.logger.info(`   Best Sharpe Ratio: ${best.performance.sharpeRatio.toFixed(3)}`);
        logger_1.logger.info(`   Best Total Return: ${best.performance.totalReturnPercent.toFixed(2)}%`);
        logger_1.logger.info(`   Best Max Drawdown: ${best.performance.maxDrawdownPercent.toFixed(2)}%`);
        logger_1.logger.info(`   Best Win Rate: ${best.performance.winRate.toFixed(1)}%`);
        logger_1.logger.info(`   Best Configuration ID: ${best.config.id}`);
        logger_1.logger.info(`   Optimization Score: ${best.score.toFixed(2)}/100`);
    }
    displayTop5Configurations(top5) {
        logger_1.logger.info('\n🏆 TOP 5 CONFIGURATIONS:');
        logger_1.logger.info('   Rank | Config ID      | Sharpe | Return% | Drawdown% | Win Rate% | Trades | Score');
        logger_1.logger.info('   ' + '-'.repeat(85));
        top5.forEach((result, index) => {
            const p = result.performance;
            logger_1.logger.info(`   ${(index + 1).toString().padStart(4)} | ${result.config.id.padEnd(14)} | ${p.sharpeRatio.toFixed(2).padStart(6)} | ${p.totalReturnPercent.toFixed(1).padStart(7)} | ${p.maxDrawdownPercent.toFixed(1).padStart(9)} | ${p.winRate.toFixed(1).padStart(9)} | ${p.totalTrades.toString().padStart(6)} | ${result.score.toFixed(1)}`);
        });
    }
    displayParameterImpacts(impacts) {
        logger_1.logger.info('\n📈 PARAMETER IMPACT ANALYSIS:');
        logger_1.logger.info('   Parameter              | Impact | Correlation | Optimal Range        | Significance');
        logger_1.logger.info('   ' + '-'.repeat(85));
        impacts.forEach(impact => {
            const optimalStr = `${impact.optimalRange[0]} - ${impact.optimalRange[1]}`;
            logger_1.logger.info(`   ${impact.parameter.padEnd(22)} | ${impact.impact.padEnd(6)} | ${impact.correlation.toFixed(3).padStart(11)} | ${optimalStr.padEnd(20)} | ${impact.significance.toFixed(3)}`);
        });
    }
    displayPerformanceDistribution(distribution) {
        logger_1.logger.info('\n📊 PERFORMANCE DISTRIBUTION:');
        logger_1.logger.info('   Sharpe Ratio:');
        logger_1.logger.info(`     Min: ${distribution.sharpeRatio.min.toFixed(3)}, Max: ${distribution.sharpeRatio.max.toFixed(3)}, Mean: ${distribution.sharpeRatio.mean.toFixed(3)}, Std: ${distribution.sharpeRatio.std.toFixed(3)}`);
        logger_1.logger.info('   Total Return (%):');
        logger_1.logger.info(`     Min: ${distribution.totalReturn.min.toFixed(2)}, Max: ${distribution.totalReturn.max.toFixed(2)}, Mean: ${distribution.totalReturn.mean.toFixed(2)}, Std: ${distribution.totalReturn.std.toFixed(2)}`);
        logger_1.logger.info('   Max Drawdown (%):');
        logger_1.logger.info(`     Min: ${distribution.maxDrawdown.min.toFixed(2)}, Max: ${distribution.maxDrawdown.max.toFixed(2)}, Mean: ${distribution.maxDrawdown.mean.toFixed(2)}, Std: ${distribution.maxDrawdown.std.toFixed(2)}`);
    }
    displayBestConfigurationDetails(best) {
        logger_1.logger.info('\n🎯 BEST CONFIGURATION DETAILS:');
        logger_1.logger.info(`   Configuration ID: ${best.config.id}`);
        logger_1.logger.info(`   Optimization Score: ${best.score.toFixed(2)}/100`);
        logger_1.logger.info('');
        logger_1.logger.info('   📋 Parameter Settings:');
        logger_1.logger.info(`     Min Confidence: ${best.config.minConfidence}%`);
        logger_1.logger.info(`     Model Consensus: ${best.config.modelConsensus}`);
        logger_1.logger.info(`     Decision Cooldown: ${best.config.decisionCooldown} minutes`);
        logger_1.logger.info(`     Risk Per Trade: ${best.config.riskPerTrade}%`);
        logger_1.logger.info(`     Stop Loss: ${best.config.stopLossPercent}%`);
        logger_1.logger.info(`     Take Profit Multiplier: ${best.config.takeProfitMultiplier}x`);
        logger_1.logger.info(`     Position Size Multiplier: ${best.config.positionSizeMultiplier}`);
        logger_1.logger.info(`     Trend Threshold: ${best.config.trendThreshold}`);
        logger_1.logger.info(`     Volatility Threshold: ${best.config.volatilityThreshold}`);
        logger_1.logger.info('');
        logger_1.logger.info('   📊 Performance Metrics:');
        logger_1.logger.info(`     Total Return: ${best.performance.totalReturnPercent.toFixed(2)}%`);
        logger_1.logger.info(`     Sharpe Ratio: ${best.performance.sharpeRatio.toFixed(3)}`);
        logger_1.logger.info(`     Calmar Ratio: ${best.performance.calmarRatio.toFixed(3)}`);
        logger_1.logger.info(`     Maximum Drawdown: ${best.performance.maxDrawdownPercent.toFixed(2)}%`);
        logger_1.logger.info(`     Win Rate: ${best.performance.winRate.toFixed(1)}%`);
        logger_1.logger.info(`     Profit Factor: ${best.performance.profitFactor.toFixed(2)}`);
        logger_1.logger.info(`     Total Trades: ${best.performance.totalTrades}`);
        logger_1.logger.info(`     Average Win: $${best.performance.averageWin.toFixed(2)}`);
        logger_1.logger.info(`     Average Loss: $${best.performance.averageLoss.toFixed(2)}`);
        logger_1.logger.info(`     Volatility: ${best.performance.volatility.toFixed(2)}%`);
    }
    displayTradeAnalysis(best) {
        if (best.trades.length === 0) {
            logger_1.logger.info('\n📋 TRADE ANALYSIS: No trades executed');
            return;
        }
        logger_1.logger.info('\n📋 TRADE-BY-TRADE ANALYSIS (Best Configuration):');
        logger_1.logger.info('   Trade | Side  | Entry Price | Exit Price | P&L      | P&L%    | Duration | Reason');
        logger_1.logger.info('   ' + '-'.repeat(90));
        best.trades.forEach((trade, index) => {
            const duration = (trade.duration / (1000 * 60 * 60)).toFixed(1);
            const reason = trade.reason.length > 20 ? trade.reason.substring(0, 17) + '...' : trade.reason;
            logger_1.logger.info(`   ${(index + 1).toString().padStart(5)} | ${trade.side.padEnd(5)} | ${trade.entryPrice.toFixed(2).padStart(11)} | ${trade.exitPrice.toFixed(2).padStart(10)} | ${trade.pnl.toFixed(2).padStart(8)} | ${trade.pnlPercent.toFixed(2).padStart(7)} | ${duration.padStart(8)}h | ${reason}`);
        });
        // Trade statistics
        const winningTrades = best.trades.filter(t => t.pnl > 0);
        const losingTrades = best.trades.filter(t => t.pnl <= 0);
        logger_1.logger.info('\n   📊 Trade Statistics:');
        logger_1.logger.info(`     Total Trades: ${best.trades.length}`);
        logger_1.logger.info(`     Winning Trades: ${winningTrades.length} (${(winningTrades.length / best.trades.length * 100).toFixed(1)}%)`);
        logger_1.logger.info(`     Losing Trades: ${losingTrades.length} (${(losingTrades.length / best.trades.length * 100).toFixed(1)}%)`);
        if (winningTrades.length > 0) {
            const avgWinDuration = winningTrades.reduce((sum, t) => sum + t.duration, 0) / winningTrades.length / (1000 * 60 * 60);
            logger_1.logger.info(`     Average Winning Trade Duration: ${avgWinDuration.toFixed(1)} hours`);
        }
        if (losingTrades.length > 0) {
            const avgLossDuration = losingTrades.reduce((sum, t) => sum + t.duration, 0) / losingTrades.length / (1000 * 60 * 60);
            logger_1.logger.info(`     Average Losing Trade Duration: ${avgLossDuration.toFixed(1)} hours`);
        }
    }
    displayInsightsAndRecommendations(insights, recommendations) {
        logger_1.logger.info('\n💡 KEY INSIGHTS:');
        insights.forEach((insight, index) => {
            logger_1.logger.info(`   ${index + 1}. ${insight}`);
        });
        logger_1.logger.info('\n🚀 RECOMMENDATIONS:');
        recommendations.forEach((recommendation, index) => {
            logger_1.logger.info(`   ${index + 1}. ${recommendation}`);
        });
    }
    /**
     * Analyze parameter impacts on performance
     */
    analyzeParameterImpacts(results) {
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
        const impacts = [];
        parameters.forEach(param => {
            const correlation = this.calculateCorrelation(results, param, 'sharpeRatio');
            const significance = Math.abs(correlation);
            const optimalRange = this.findOptimalRange(results, param);
            let impact = 'LOW';
            if (significance > 0.3)
                impact = 'HIGH';
            else if (significance > 0.15)
                impact = 'MEDIUM';
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
    calculateCorrelation(results, parameter, metric) {
        const paramValues = results.map(r => r.config[parameter]);
        const metricValues = results.map(r => r.performance[metric]);
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
    findOptimalRange(results, parameter) {
        // Get top 20% of results
        const topResults = results.slice(0, Math.ceil(results.length * 0.2));
        const paramValues = topResults.map(r => r.config[parameter]);
        const min = Math.min(...paramValues);
        const max = Math.max(...paramValues);
        return [min, max];
    }
    /**
     * Calculate performance distribution statistics
     */
    calculatePerformanceDistribution(results) {
        const metrics = ['sharpeRatio', 'totalReturnPercent', 'maxDrawdownPercent'];
        const distribution = {};
        metrics.forEach(metric => {
            const values = results.map(r => r.performance[metric]);
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
    calculateStandardDeviation(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        return Math.sqrt(variance);
    }
    /**
     * Generate insights from optimization results
     */
    generateInsights(results, impacts) {
        const insights = [];
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
        }
        else {
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
    generateRecommendations(best, impacts) {
        const recommendations = [];
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
exports.OptimizationAnalyzer = OptimizationAnalyzer;
// Export factory function
function createOptimizationAnalyzer() {
    return new OptimizationAnalyzer();
}
//# sourceMappingURL=optimizationAnalyzer.js.map