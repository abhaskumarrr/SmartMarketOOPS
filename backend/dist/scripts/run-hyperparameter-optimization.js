#!/usr/bin/env node
"use strict";
/**
 * Comprehensive Hyperparameter Optimization Script
 * Systematically tests parameter combinations to maximize trading performance
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.HyperparameterOptimizationRunner = void 0;
const hyperparameterOptimizer_1 = require("../services/hyperparameterOptimizer");
const optimizationAnalyzer_1 = require("../services/optimizationAnalyzer");
const logger_1 = require("../utils/logger");
const redisStreamsService_1 = require("../services/redisStreamsService");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
class HyperparameterOptimizationRunner {
    /**
     * Run comprehensive hyperparameter optimization
     */
    async runOptimization() {
        const startTime = Date.now();
        try {
            logger_1.logger.info('🔬 Starting Comprehensive Hyperparameter Optimization...');
            logger_1.logger.info('🎯 Target Metrics: Sharpe Ratio (Primary), Total Return (Secondary), Max Drawdown (Tertiary)');
            // Initialize infrastructure
            await this.initializeInfrastructure();
            // Create optimizer and analyzer
            const optimizer = (0, hyperparameterOptimizer_1.createHyperparameterOptimizer)();
            const analyzer = (0, optimizationAnalyzer_1.createOptimizationAnalyzer)();
            // Configuration for optimization
            const numIterations = this.getOptimizationIterations();
            logger_1.logger.info(`📊 Configuration:`, {
                iterations: numIterations,
                methodology: 'Grid Search + Random Search',
                dataset: '30-day BTCUSD hourly data',
                targetMetrics: ['Sharpe Ratio', 'Total Return %', 'Max Drawdown %'],
            });
            // Run optimization
            logger_1.logger.info('🚀 Starting optimization process...');
            const results = await optimizer.runOptimization(numIterations);
            if (results.length === 0) {
                throw new Error('No valid optimization results generated');
            }
            // Analyze results
            logger_1.logger.info('📊 Analyzing optimization results...');
            const summary = analyzer.analyzeResults(results);
            // Display comprehensive report
            analyzer.displayOptimizationReport(summary);
            // Save results to file
            await this.saveOptimizationResults(results, summary);
            // Display execution summary
            this.displayExecutionSummary(results, startTime);
            logger_1.logger.info('🎉 Hyperparameter optimization completed successfully!');
        }
        catch (error) {
            logger_1.logger.error('❌ Hyperparameter optimization failed:', error);
            throw error;
        }
    }
    /**
     * Initialize infrastructure
     */
    async initializeInfrastructure() {
        try {
            await redisStreamsService_1.redisStreamsService.initialize();
            logger_1.logger.info('✅ Redis Streams initialized for optimization');
        }
        catch (error) {
            logger_1.logger.warn('⚠️ Redis Streams initialization failed, continuing without it');
        }
    }
    /**
     * Get number of optimization iterations based on environment
     */
    getOptimizationIterations() {
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
    async saveOptimizationResults(results, summary) {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const resultsDir = path_1.default.join(process.cwd(), 'optimization_results');
            // Create results directory if it doesn't exist
            if (!fs_1.default.existsSync(resultsDir)) {
                fs_1.default.mkdirSync(resultsDir, { recursive: true });
            }
            // Save detailed results
            const detailedResultsPath = path_1.default.join(resultsDir, `optimization_results_${timestamp}.json`);
            fs_1.default.writeFileSync(detailedResultsPath, JSON.stringify(results, null, 2));
            // Save summary
            const summaryPath = path_1.default.join(resultsDir, `optimization_summary_${timestamp}.json`);
            fs_1.default.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
            // Save top 5 configurations in CSV format
            const csvPath = path_1.default.join(resultsDir, `top_configurations_${timestamp}.csv`);
            this.saveTop5AsCSV(summary.top5Configurations, csvPath);
            // Save best configuration parameters
            const bestConfigPath = path_1.default.join(resultsDir, `best_configuration_${timestamp}.json`);
            fs_1.default.writeFileSync(bestConfigPath, JSON.stringify(summary.bestConfiguration.config, null, 2));
            logger_1.logger.info('💾 Optimization results saved:', {
                detailedResults: detailedResultsPath,
                summary: summaryPath,
                topConfigurations: csvPath,
                bestConfiguration: bestConfigPath,
            });
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to save optimization results:', error);
        }
    }
    /**
     * Save top 5 configurations as CSV
     */
    saveTop5AsCSV(top5, csvPath) {
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
        fs_1.default.writeFileSync(csvPath, csvContent);
    }
    /**
     * Display execution summary
     */
    displayExecutionSummary(results, startTime) {
        const duration = (Date.now() - startTime) / 1000;
        const best = results[0];
        const worst = results[results.length - 1];
        logger_1.logger.info('\n' + '⚡ OPTIMIZATION EXECUTION SUMMARY'.padStart(50, '='));
        logger_1.logger.info('='.repeat(80));
        logger_1.logger.info('🔬 EXECUTION METRICS:');
        logger_1.logger.info(`   Total Duration: ${duration.toFixed(1)} seconds (${(duration / 60).toFixed(1)} minutes)`);
        logger_1.logger.info(`   Configurations Tested: ${results.length}`);
        logger_1.logger.info(`   Average Time per Configuration: ${(duration / results.length).toFixed(2)} seconds`);
        logger_1.logger.info(`   Optimization Speed: ${(results.length / duration * 60).toFixed(1)} configs/minute`);
        logger_1.logger.info('\n📊 PERFORMANCE RANGE:');
        logger_1.logger.info(`   Sharpe Ratio Range: ${worst.performance.sharpeRatio.toFixed(3)} to ${best.performance.sharpeRatio.toFixed(3)}`);
        logger_1.logger.info(`   Return Range: ${worst.performance.totalReturnPercent.toFixed(1)}% to ${best.performance.totalReturnPercent.toFixed(1)}%`);
        logger_1.logger.info(`   Drawdown Range: ${best.performance.maxDrawdownPercent.toFixed(1)}% to ${worst.performance.maxDrawdownPercent.toFixed(1)}%`);
        logger_1.logger.info('\n🎯 OPTIMIZATION SUCCESS:');
        const improvementVsBaseline = ((best.performance.sharpeRatio - 0) * 100).toFixed(1);
        logger_1.logger.info(`   Best Sharpe Ratio: ${best.performance.sharpeRatio.toFixed(3)} (${improvementVsBaseline}% improvement vs baseline)`);
        logger_1.logger.info(`   Best Configuration: ${best.config.id}`);
        logger_1.logger.info(`   Optimization Score: ${best.score.toFixed(2)}/100`);
        if (best.performance.sharpeRatio > 1) {
            logger_1.logger.info('   ✅ Excellent risk-adjusted returns achieved');
        }
        if (best.performance.totalReturnPercent > 10) {
            logger_1.logger.info('   ✅ Strong absolute returns achieved');
        }
        if (best.performance.maxDrawdownPercent < 10) {
            logger_1.logger.info('   ✅ Good risk control maintained');
        }
        logger_1.logger.info('\n🚀 NEXT STEPS:');
        logger_1.logger.info('   1. Validate best configuration on out-of-sample data');
        logger_1.logger.info('   2. Test on different market conditions and time periods');
        logger_1.logger.info('   3. Consider ensemble approach with top configurations');
        logger_1.logger.info('   4. Implement walk-forward optimization for robustness');
        logger_1.logger.info('   5. Begin paper trading with optimized parameters');
        logger_1.logger.info('='.repeat(80));
    }
    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            await redisStreamsService_1.redisStreamsService.shutdown();
            logger_1.logger.info('🧹 Cleanup completed');
        }
        catch (error) {
            logger_1.logger.error('❌ Cleanup failed:', error);
        }
    }
}
exports.HyperparameterOptimizationRunner = HyperparameterOptimizationRunner;
/**
 * Main execution function
 */
async function main() {
    const runner = new HyperparameterOptimizationRunner();
    try {
        await runner.runOptimization();
    }
    catch (error) {
        logger_1.logger.error('💥 Hyperparameter optimization failed:', error);
        process.exit(1);
    }
    finally {
        await runner.cleanup();
    }
}
// Handle graceful shutdown
process.on('SIGINT', async () => {
    logger_1.logger.info('🛑 Received SIGINT, cleaning up...');
    const runner = new HyperparameterOptimizationRunner();
    await runner.cleanup();
    process.exit(0);
});
process.on('SIGTERM', async () => {
    logger_1.logger.info('🛑 Received SIGTERM, cleaning up...');
    const runner = new HyperparameterOptimizationRunner();
    await runner.cleanup();
    process.exit(0);
});
// Run if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=run-hyperparameter-optimization.js.map