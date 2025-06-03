#!/usr/bin/env node
"use strict";
/**
 * Strategy Comparison Script
 * Compares original MA Crossover vs Enhanced Trend Strategy
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.StrategyComparison = void 0;
const marketDataProvider_1 = require("../services/marketDataProvider");
const technicalAnalysis_1 = require("../utils/technicalAnalysis");
const portfolioManager_1 = require("../services/portfolioManager");
const performanceAnalytics_1 = require("../utils/performanceAnalytics");
const movingAverageCrossover_1 = require("../strategies/movingAverageCrossover");
const enhancedTrendStrategy_1 = require("../strategies/enhancedTrendStrategy");
const logger_1 = require("../utils/logger");
const redisStreamsService_1 = require("../services/redisStreamsService");
class StrategyComparison {
    /**
     * Run strategy comparison
     */
    async runComparison() {
        const startTime = Date.now();
        try {
            logger_1.logger.info('🔬 Starting strategy comparison analysis...');
            // Initialize infrastructure
            await this.initializeInfrastructure();
            // Create configurations
            const config = this.createBacktestConfig();
            // Load market data once
            const marketData = await this.loadMarketData(config);
            // Test original strategy
            logger_1.logger.info('\n📊 Testing Original MA Crossover Strategy...');
            const originalResults = await this.testStrategy(config, (0, movingAverageCrossover_1.createMACrossoverStrategy)(), marketData, 'Original MA Crossover');
            // Test enhanced strategy
            logger_1.logger.info('\n🚀 Testing Enhanced Trend Strategy...');
            const enhancedResults = await this.testStrategy(config, (0, enhancedTrendStrategy_1.createEnhancedTrendStrategy)(), marketData, 'Enhanced Trend');
            // Compare results
            this.compareStrategies(originalResults, enhancedResults, startTime);
            logger_1.logger.info('🎉 Strategy comparison completed successfully!');
        }
        catch (error) {
            logger_1.logger.error('❌ Strategy comparison failed:', error);
            throw error;
        }
    }
    async initializeInfrastructure() {
        try {
            await redisStreamsService_1.redisStreamsService.initialize();
            logger_1.logger.info('✅ Redis Streams initialized');
        }
        catch (error) {
            logger_1.logger.warn('⚠️ Redis Streams initialization failed, continuing without it');
        }
    }
    createBacktestConfig() {
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
    async loadMarketData(config) {
        logger_1.logger.info('📊 Loading market data for comparison...');
        const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
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
        const sma20 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 20);
        const sma50 = technicalAnalysis_1.technicalAnalysis.calculateSMA(closes, 50);
        const ema12 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 12);
        const ema26 = technicalAnalysis_1.technicalAnalysis.calculateEMA(closes, 26);
        const rsi = technicalAnalysis_1.technicalAnalysis.calculateRSI(closes, 14);
        const macd = technicalAnalysis_1.technicalAnalysis.calculateMACD(closes, 12, 26, 9);
        const bollinger = technicalAnalysis_1.technicalAnalysis.calculateBollingerBands(closes, 20, 2);
        const volumeSMA = technicalAnalysis_1.technicalAnalysis.calculateSMA(volumes, 20);
        const enhancedData = response.data.map((point, index) => ({
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
        logger_1.logger.info(`✅ Enhanced ${response.data.length} data points with comprehensive indicators`);
        return enhancedData;
    }
    async testStrategy(config, strategy, marketData, strategyName) {
        strategy.initialize(config);
        const portfolioManager = new portfolioManager_1.PortfolioManager(config);
        let signalCount = 0;
        let tradeCount = 0;
        let validSignals = 0;
        for (let i = 0; i < marketData.length; i++) {
            const currentCandle = marketData[i];
            // Update portfolio
            portfolioManager.updatePositions(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            // Check stop loss/take profit
            const closedTrades = portfolioManager.checkStopLossAndTakeProfit(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            tradeCount += closedTrades.length;
            // Generate signal
            const signal = strategy.generateSignal(marketData, i);
            if (signal) {
                signalCount++;
                if (signal.confidence > 0) {
                    validSignals++;
                    // Execute trade
                    const trade = portfolioManager.executeTrade(signal, currentCandle.close, currentCandle.timestamp);
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
        const performance = performanceAnalytics_1.PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, config);
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
    compareStrategies(originalResults, enhancedResults, startTime) {
        const duration = (Date.now() - startTime) / 1000;
        logger_1.logger.info('\n' + '🔬 STRATEGY COMPARISON ANALYSIS'.padStart(50, '='));
        logger_1.logger.info('='.repeat(100));
        // Execution Summary
        logger_1.logger.info('⚡ EXECUTION SUMMARY:');
        logger_1.logger.info(`   Analysis Duration: ${duration.toFixed(2)} seconds`);
        logger_1.logger.info(`   Data Points: ${originalResults.dataPoints.toLocaleString()}`);
        logger_1.logger.info(`   Processing Speed: ${(originalResults.dataPoints / duration).toFixed(0)} points/sec`);
        // Signal Generation Comparison
        logger_1.logger.info('\n📊 SIGNAL GENERATION:');
        logger_1.logger.info(`   Original Strategy:`);
        logger_1.logger.info(`     Total Signals: ${originalResults.signalCount}`);
        logger_1.logger.info(`     Valid Signals: ${originalResults.validSignals}`);
        logger_1.logger.info(`     Signal Quality: ${((originalResults.validSignals / Math.max(originalResults.signalCount, 1)) * 100).toFixed(1)}%`);
        logger_1.logger.info(`   Enhanced Strategy:`);
        logger_1.logger.info(`     Total Signals: ${enhancedResults.signalCount}`);
        logger_1.logger.info(`     Valid Signals: ${enhancedResults.validSignals}`);
        logger_1.logger.info(`     Signal Quality: ${((enhancedResults.validSignals / Math.max(enhancedResults.signalCount, 1)) * 100).toFixed(1)}%`);
        // Performance Comparison
        logger_1.logger.info('\n💰 PERFORMANCE COMPARISON:');
        const metrics = [
            { name: 'Total Return (%)', orig: originalResults.performance.totalReturnPercent, enh: enhancedResults.performance.totalReturnPercent },
            { name: 'Annualized Return (%)', orig: originalResults.performance.annualizedReturn, enh: enhancedResults.performance.annualizedReturn },
            { name: 'Sharpe Ratio', orig: originalResults.performance.sharpeRatio, enh: enhancedResults.performance.sharpeRatio },
            { name: 'Max Drawdown (%)', orig: originalResults.performance.maxDrawdownPercent, enh: enhancedResults.performance.maxDrawdownPercent },
            { name: 'Win Rate (%)', orig: originalResults.performance.winRate, enh: enhancedResults.performance.winRate },
            { name: 'Profit Factor', orig: originalResults.performance.profitFactor, enh: enhancedResults.performance.profitFactor },
            { name: 'Total Trades', orig: originalResults.performance.totalTrades, enh: enhancedResults.performance.totalTrades },
        ];
        logger_1.logger.info('   Metric                    Original      Enhanced      Improvement');
        logger_1.logger.info('   ' + '-'.repeat(65));
        metrics.forEach(metric => {
            const improvement = metric.enh - metric.orig;
            const improvementStr = improvement > 0 ? `+${improvement.toFixed(2)}` : improvement.toFixed(2);
            const indicator = improvement > 0 ? '📈' : improvement < 0 ? '📉' : '➡️';
            logger_1.logger.info(`   ${metric.name.padEnd(25)} ${metric.orig.toFixed(2).padStart(8)} ${metric.enh.toFixed(2).padStart(12)} ${indicator} ${improvementStr.padStart(8)}`);
        });
        // Strategy Analysis
        logger_1.logger.info('\n🎯 STRATEGY ANALYSIS:');
        // Original Strategy Issues
        logger_1.logger.info('   Original MA Crossover Issues:');
        if (originalResults.performance.totalReturnPercent < 0) {
            logger_1.logger.info('     🔴 Negative returns - lagging indicators problem');
        }
        if (originalResults.performance.winRate < 40) {
            logger_1.logger.info('     🔴 Low win rate - poor signal timing');
        }
        if (originalResults.performance.maxDrawdownPercent > 30) {
            logger_1.logger.info('     🔴 High drawdown - inadequate risk management');
        }
        if (originalResults.performance.totalTrades > 10) {
            logger_1.logger.info('     🔴 Overtrading - whipsaw in sideways markets');
        }
        // Enhanced Strategy Improvements
        logger_1.logger.info('   Enhanced Strategy Improvements:');
        if (enhancedResults.performance.totalReturnPercent > originalResults.performance.totalReturnPercent) {
            logger_1.logger.info('     ✅ Better returns through trend analysis');
        }
        if (enhancedResults.performance.winRate > originalResults.performance.winRate) {
            logger_1.logger.info('     ✅ Higher win rate with better signal timing');
        }
        if (enhancedResults.performance.maxDrawdownPercent < originalResults.performance.maxDrawdownPercent) {
            logger_1.logger.info('     ✅ Lower drawdown with dynamic risk management');
        }
        if (enhancedResults.validSignals < originalResults.validSignals) {
            logger_1.logger.info('     ✅ Fewer but higher quality signals (anti-whipsaw)');
        }
        // Overall Assessment
        logger_1.logger.info('\n⭐ OVERALL ASSESSMENT:');
        const improvementScore = this.calculateImprovementScore(originalResults.performance, enhancedResults.performance);
        if (improvementScore >= 3) {
            logger_1.logger.info('   🌟 SIGNIFICANT IMPROVEMENT - Enhanced strategy is substantially better');
        }
        else if (improvementScore >= 1) {
            logger_1.logger.info('   ✅ MODERATE IMPROVEMENT - Enhanced strategy shows promise');
        }
        else if (improvementScore >= -1) {
            logger_1.logger.info('   ⚠️ MIXED RESULTS - Some improvements, some regressions');
        }
        else {
            logger_1.logger.info('   ❌ POOR RESULTS - Enhanced strategy needs further work');
        }
        // Recommendations
        logger_1.logger.info('\n💡 RECOMMENDATIONS:');
        if (enhancedResults.performance.totalReturnPercent > 0) {
            logger_1.logger.info('   ✅ Enhanced strategy shows positive returns - consider live testing');
        }
        if (enhancedResults.performance.winRate > 50) {
            logger_1.logger.info('   ✅ Good win rate - strategy has edge in current market conditions');
        }
        if (enhancedResults.performance.sharpeRatio > 1) {
            logger_1.logger.info('   ✅ Good risk-adjusted returns - acceptable risk profile');
        }
        logger_1.logger.info('   📊 Consider testing on different market periods and timeframes');
        logger_1.logger.info('   🔧 Fine-tune parameters based on market regime analysis');
        logger_1.logger.info('   📈 Monitor performance in live paper trading before real deployment');
        logger_1.logger.info('='.repeat(100));
    }
    calculateImprovementScore(original, enhanced) {
        let score = 0;
        // Return improvement
        if (enhanced.totalReturnPercent > original.totalReturnPercent + 10)
            score += 2;
        else if (enhanced.totalReturnPercent > original.totalReturnPercent)
            score += 1;
        else if (enhanced.totalReturnPercent < original.totalReturnPercent - 10)
            score -= 2;
        else
            score -= 1;
        // Sharpe ratio improvement
        if (enhanced.sharpeRatio > original.sharpeRatio + 0.5)
            score += 1;
        else if (enhanced.sharpeRatio < original.sharpeRatio - 0.5)
            score -= 1;
        // Drawdown improvement
        if (enhanced.maxDrawdownPercent < original.maxDrawdownPercent - 5)
            score += 1;
        else if (enhanced.maxDrawdownPercent > original.maxDrawdownPercent + 5)
            score -= 1;
        // Win rate improvement
        if (enhanced.winRate > original.winRate + 10)
            score += 1;
        else if (enhanced.winRate < original.winRate - 10)
            score -= 1;
        return score;
    }
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
exports.StrategyComparison = StrategyComparison;
/**
 * Main execution function
 */
async function main() {
    const comparison = new StrategyComparison();
    try {
        await comparison.runComparison();
    }
    catch (error) {
        logger_1.logger.error('💥 Strategy comparison failed:', error);
        process.exit(1);
    }
    finally {
        await comparison.cleanup();
    }
}
// Handle graceful shutdown
process.on('SIGINT', async () => {
    logger_1.logger.info('🛑 Received SIGINT, cleaning up...');
    const comparison = new StrategyComparison();
    await comparison.cleanup();
    process.exit(0);
});
// Run if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=run-strategy-comparison.js.map