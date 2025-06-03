#!/usr/bin/env node
"use strict";
/**
 * Retrained Model Comprehensive Backtest
 * Tests the retrained AI models against the original models with multi-timeframe analysis
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RetrainedModelBacktestRunner = void 0;
const retrainedAITradingSystem_1 = require("../services/retrainedAITradingSystem");
const multiTimeframeAITradingSystem_1 = require("../services/multiTimeframeAITradingSystem");
const multiTimeframeBacktester_1 = require("../services/multiTimeframeBacktester");
const marketDataProvider_1 = require("../services/marketDataProvider");
const portfolioManager_1 = require("../services/portfolioManager");
const performanceAnalytics_1 = require("../utils/performanceAnalytics");
const logger_1 = require("../utils/logger");
class RetrainedModelBacktestRunner {
    constructor() {
        this.backtester = (0, multiTimeframeBacktester_1.createMultiTimeframeBacktester)();
    }
    /**
     * Run comprehensive backtest comparing original vs retrained models
     */
    async runComparativeBacktest() {
        const startTime = Date.now();
        try {
            logger_1.logger.info('🧠 Starting Retrained Model Comprehensive Backtest...');
            // Step 1: Test configurations
            const testConfigs = this.createTestConfigurations();
            // Step 2: Run comparative tests
            const comparisonResults = [];
            for (const config of testConfigs) {
                logger_1.logger.info(`\n🔬 Testing configuration: ${config.strategy}`);
                const result = await this.runModelComparison(config);
                comparisonResults.push(result);
            }
            // Step 3: Analyze overall results
            this.analyzeOverallResults(comparisonResults, startTime);
            // Step 4: Generate recommendations
            this.generateRecommendations(comparisonResults);
            const duration = (Date.now() - startTime) / 1000;
            logger_1.logger.info(`🎉 Retrained model backtest completed in ${duration.toFixed(2)} seconds`);
        }
        catch (error) {
            logger_1.logger.error('❌ Retrained model backtest failed:', error);
            throw error;
        }
    }
    /**
     * Create test configurations for comparison
     */
    createTestConfigurations() {
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
    async runModelComparison(config) {
        logger_1.logger.info(`📊 Running model comparison for ${config.strategy}...`);
        // Load market data
        const marketData = await this.loadMarketData(config);
        // Test original model
        logger_1.logger.info('🔄 Testing original multi-timeframe AI model...');
        const originalResults = await this.testModel((0, multiTimeframeAITradingSystem_1.createMultiTimeframeAITradingSystem)(), marketData, config, 'Original_Multi_TF_AI');
        // Test retrained model
        logger_1.logger.info('🧠 Testing retrained AI model...');
        const retrainedResults = await this.testModel(await this.createAndInitializeRetrainedModel(config), marketData, config, 'Retrained_AI');
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
    async loadMarketData(config) {
        try {
            const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
                symbol: config.symbol,
                timeframe: config.timeframe,
                startDate: config.startDate,
                endDate: config.endDate,
                exchange: 'binance',
            }, 'binance');
            return this.enhanceMarketData(response.data);
        }
        catch (error) {
            logger_1.logger.warn('⚠️ Failed to load real data, using enhanced mock data');
            const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
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
    enhanceMarketData(data) {
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
    async createAndInitializeRetrainedModel(config) {
        const retrainedSystem = (0, retrainedAITradingSystem_1.createRetrainedAITradingSystem)();
        await retrainedSystem.initialize(config);
        return retrainedSystem;
    }
    /**
     * Test a specific model
     */
    async testModel(strategy, marketData, config, strategyName) {
        if (strategy.initialize) {
            strategy.initialize(config);
        }
        const portfolioManager = new portfolioManager_1.PortfolioManager(config);
        let signalCount = 0;
        let tradeCount = 0;
        // Run backtest
        for (let i = 0; i < marketData.length; i++) {
            const currentCandle = marketData[i];
            portfolioManager.updatePositions(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            const closedTrades = portfolioManager.checkStopLossAndTakeProfit(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            tradeCount += closedTrades.length;
            const signal = strategy.generateSignal(marketData, i);
            if (signal) {
                signalCount++;
                const trade = portfolioManager.executeTrade(signal, currentCandle.close, currentCandle.timestamp);
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
        const performance = performanceAnalytics_1.PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, config);
        return {
            performance,
            trades,
            signals: signalCount,
        };
    }
    /**
     * Calculate improvements between models
     */
    calculateImprovements(originalResults, retrainedResults) {
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
    analyzeOverallResults(results, startTime) {
        const duration = (Date.now() - startTime) / 1000;
        logger_1.logger.info('\n' + '🧠 RETRAINED MODEL BACKTEST RESULTS'.padStart(60, '='));
        logger_1.logger.info('='.repeat(120));
        // Overall Summary
        logger_1.logger.info('📊 OVERALL COMPARISON SUMMARY:');
        logger_1.logger.info(`   Test Duration: ${duration.toFixed(2)} seconds`);
        logger_1.logger.info(`   Configurations Tested: ${results.length}`);
        logger_1.logger.info(`   Data Source: Real Binance data (with fallback)`);
        // Performance Comparison Table
        logger_1.logger.info('\n📈 PERFORMANCE COMPARISON:');
        logger_1.logger.info('   Configuration           | Original Return | Retrained Return | Improvement | Original Sharpe | Retrained Sharpe | Sharpe Δ');
        logger_1.logger.info('   ' + '-'.repeat(110));
        results.forEach(result => {
            const config = result.config.strategy.padEnd(23);
            const origReturn = result.originalModel.performance.totalReturnPercent.toFixed(2).padStart(15);
            const retrainedReturn = result.retrainedModel.performance.totalReturnPercent.toFixed(2).padStart(16);
            const returnImpr = (result.improvement.returnImprovement > 0 ? '+' : '') + result.improvement.returnImprovement.toFixed(2).padStart(11);
            const origSharpe = result.originalModel.performance.sharpeRatio.toFixed(3).padStart(15);
            const retrainedSharpe = result.retrainedModel.performance.sharpeRatio.toFixed(3).padStart(16);
            const sharpeImpr = (result.improvement.sharpeImprovement > 0 ? '+' : '') + result.improvement.sharpeImprovement.toFixed(3).padStart(8);
            logger_1.logger.info(`   ${config} | ${origReturn}% | ${retrainedReturn}% | ${returnImpr}% | ${origSharpe} | ${retrainedSharpe} | ${sharpeImpr}`);
        });
        // Detailed Analysis
        logger_1.logger.info('\n🔍 DETAILED ANALYSIS:');
        const avgReturnImprovement = results.reduce((sum, r) => sum + r.improvement.returnImprovement, 0) / results.length;
        const avgSharpeImprovement = results.reduce((sum, r) => sum + r.improvement.sharpeImprovement, 0) / results.length;
        const avgDrawdownImprovement = results.reduce((sum, r) => sum + r.improvement.drawdownImprovement, 0) / results.length;
        const avgSignalImprovement = results.reduce((sum, r) => sum + r.improvement.signalImprovement, 0) / results.length;
        logger_1.logger.info(`   Average Return Improvement: ${avgReturnImprovement > 0 ? '+' : ''}${avgReturnImprovement.toFixed(2)}%`);
        logger_1.logger.info(`   Average Sharpe Improvement: ${avgSharpeImprovement > 0 ? '+' : ''}${avgSharpeImprovement.toFixed(3)}`);
        logger_1.logger.info(`   Average Drawdown Improvement: ${avgDrawdownImprovement > 0 ? '+' : ''}${avgDrawdownImprovement.toFixed(2)}%`);
        logger_1.logger.info(`   Average Signal Improvement: ${avgSignalImprovement > 0 ? '+' : ''}${avgSignalImprovement.toFixed(0)} signals`);
        // Win/Loss Analysis
        const improvementWins = results.filter(r => r.improvement.returnImprovement > 0).length;
        const improvementLosses = results.length - improvementWins;
        logger_1.logger.info(`\n🏆 IMPROVEMENT ANALYSIS:`);
        logger_1.logger.info(`   Configurations with Better Returns: ${improvementWins}/${results.length} (${(improvementWins / results.length * 100).toFixed(1)}%)`);
        logger_1.logger.info(`   Configurations with Worse Returns: ${improvementLosses}/${results.length} (${(improvementLosses / results.length * 100).toFixed(1)}%)`);
        // Best Performing Configuration
        const bestConfig = results.reduce((best, current) => current.improvement.returnImprovement > best.improvement.returnImprovement ? current : best);
        logger_1.logger.info(`\n🌟 BEST PERFORMING CONFIGURATION:`);
        logger_1.logger.info(`   Configuration: ${bestConfig.config.strategy}`);
        logger_1.logger.info(`   Return Improvement: ${bestConfig.improvement.returnImprovement > 0 ? '+' : ''}${bestConfig.improvement.returnImprovement.toFixed(2)}%`);
        logger_1.logger.info(`   Sharpe Improvement: ${bestConfig.improvement.sharpeImprovement > 0 ? '+' : ''}${bestConfig.improvement.sharpeImprovement.toFixed(3)}`);
        logger_1.logger.info(`   Signal Improvement: ${bestConfig.improvement.signalImprovement > 0 ? '+' : ''}${bestConfig.improvement.signalImprovement} signals`);
        // Model Effectiveness Assessment
        logger_1.logger.info(`\n⭐ MODEL EFFECTIVENESS ASSESSMENT:`);
        if (avgReturnImprovement > 2) {
            logger_1.logger.info(`   🎉 EXCELLENT: Retrained models show significant improvement (${avgReturnImprovement.toFixed(2)}% avg return boost)`);
        }
        else if (avgReturnImprovement > 0.5) {
            logger_1.logger.info(`   ✅ GOOD: Retrained models show meaningful improvement (${avgReturnImprovement.toFixed(2)}% avg return boost)`);
        }
        else if (avgReturnImprovement > -0.5) {
            logger_1.logger.info(`   ⚠️ NEUTRAL: Retrained models show mixed results (${avgReturnImprovement.toFixed(2)}% avg return change)`);
        }
        else {
            logger_1.logger.info(`   ❌ POOR: Retrained models underperform original models (${avgReturnImprovement.toFixed(2)}% avg return loss)`);
        }
        logger_1.logger.info('='.repeat(120));
    }
    /**
     * Generate recommendations based on results
     */
    generateRecommendations(results) {
        logger_1.logger.info('\n💡 RECOMMENDATIONS:');
        const avgImprovement = results.reduce((sum, r) => sum + r.improvement.returnImprovement, 0) / results.length;
        if (avgImprovement > 1) {
            logger_1.logger.info('   🚀 DEPLOY RETRAINED MODELS: Significant performance improvement detected');
            logger_1.logger.info('   📊 Monitor performance closely in live trading');
            logger_1.logger.info('   🔄 Consider retraining models monthly with new data');
        }
        else if (avgImprovement > 0) {
            logger_1.logger.info('   ✅ GRADUAL DEPLOYMENT: Modest improvement detected');
            logger_1.logger.info('   🧪 Test with smaller position sizes initially');
            logger_1.logger.info('   📈 Continue monitoring and optimization');
        }
        else {
            logger_1.logger.info('   ⚠️ FURTHER OPTIMIZATION NEEDED: Limited or negative improvement');
            logger_1.logger.info('   🔧 Review training data quality and feature engineering');
            logger_1.logger.info('   📚 Consider additional model architectures');
        }
        logger_1.logger.info('   🎯 Next Steps:');
        logger_1.logger.info('     1. Validate results with longer backtesting periods');
        logger_1.logger.info('     2. Test on different market conditions and volatility regimes');
        logger_1.logger.info('     3. Implement ensemble methods combining best models');
        logger_1.logger.info('     4. Set up automated model retraining pipeline');
        logger_1.logger.info('     5. Begin paper trading with retrained models');
    }
}
exports.RetrainedModelBacktestRunner = RetrainedModelBacktestRunner;
/**
 * Main execution function
 */
async function main() {
    const runner = new RetrainedModelBacktestRunner();
    try {
        await runner.runComparativeBacktest();
    }
    catch (error) {
        logger_1.logger.error('💥 Retrained model backtest failed:', error);
        process.exit(1);
    }
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=run-retrained-model-backtest.js.map