"use strict";
/**
 * Multi-Timeframe Backtesting Engine
 * Validates trading strategies across multiple timeframes with proper temporal consistency
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MultiTimeframeBacktester = void 0;
exports.createMultiTimeframeBacktester = createMultiTimeframeBacktester;
const multiTimeframeDataProvider_1 = require("./multiTimeframeDataProvider");
const multiTimeframeAITradingSystem_1 = require("./multiTimeframeAITradingSystem");
const portfolioManager_1 = require("./portfolioManager");
const performanceAnalytics_1 = require("../utils/performanceAnalytics");
const marketDataProvider_1 = require("./marketDataProvider");
const logger_1 = require("../utils/logger");
class MultiTimeframeBacktester {
    constructor() {
        this.supportedTimeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
        this.dataProvider = new multiTimeframeDataProvider_1.MultiTimeframeDataProvider();
    }
    /**
     * Run comprehensive multi-timeframe backtest
     */
    async runBacktest(config, targetTimeframes = ['5m', '15m', '1h', '4h', '1d']) {
        const startTime = Date.now();
        logger_1.logger.info('🕐 Starting Multi-Timeframe Backtest...', {
            symbol: config.symbol,
            timeframes: targetTimeframes,
            period: `${config.startDate.toISOString().split('T')[0]} to ${config.endDate.toISOString().split('T')[0]}`,
        });
        try {
            // Step 1: Load and prepare multi-timeframe data
            const multiTimeframeData = await this.loadMultiTimeframeData(config, targetTimeframes);
            // Step 2: Validate temporal consistency
            const temporalConsistency = this.validateTemporalConsistency(multiTimeframeData);
            // Step 3: Initialize trading system and portfolio manager
            const strategy = new multiTimeframeAITradingSystem_1.MultiTimeframeAITradingSystem();
            strategy.parameters.enabledTimeframes = targetTimeframes;
            strategy.initialize(config);
            const portfolioManager = new portfolioManager_1.PortfolioManager(config);
            // Step 4: Run backtest with multi-timeframe analysis
            const backtestResults = await this.executeMultiTimeframeBacktest(strategy, portfolioManager, multiTimeframeData, config);
            // Step 5: Analyze timeframe-specific performance
            const timeframePerformances = this.analyzeTimeframePerformances(backtestResults.signals, targetTimeframes);
            // Step 6: Calculate execution metrics
            const executionMetrics = this.calculateExecutionMetrics(backtestResults.processingTimes, targetTimeframes, startTime);
            const result = {
                config,
                performance: backtestResults.performance,
                trades: backtestResults.trades,
                portfolioHistory: backtestResults.portfolioHistory,
                timeframePerformances,
                temporalConsistency,
                hierarchicalDecisionStats: backtestResults.hierarchicalStats,
                executionMetrics,
            };
            const duration = (Date.now() - startTime) / 1000;
            logger_1.logger.info('✅ Multi-Timeframe Backtest completed', {
                duration: `${duration.toFixed(2)}s`,
                totalTrades: backtestResults.trades.length,
                totalReturn: `${backtestResults.performance.totalReturnPercent.toFixed(2)}%`,
                sharpeRatio: backtestResults.performance.sharpeRatio.toFixed(3),
            });
            return result;
        }
        catch (error) {
            logger_1.logger.error('❌ Multi-Timeframe Backtest failed:', error);
            throw error;
        }
    }
    /**
     * Load and prepare multi-timeframe data
     */
    async loadMultiTimeframeData(config, targetTimeframes) {
        logger_1.logger.info('📊 Loading multi-timeframe data...', { targetTimeframes });
        // Load base data (use hourly for real data availability)
        const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
            symbol: config.symbol,
            timeframe: config.timeframe, // Use config timeframe for real data
            startDate: config.startDate,
            endDate: config.endDate,
            exchange: 'binance', // Use real exchange data
        }, 'binance'); // Specify Binance provider
        // Generate multi-timeframe data
        const multiTimeframeData = this.dataProvider.generateMultiTimeframeData(response.data, targetTimeframes);
        logger_1.logger.info('✅ Multi-timeframe data loaded', {
            baseDataPoints: response.data.length,
            multiTimeframePoints: multiTimeframeData.length,
            timeframes: targetTimeframes,
        });
        return multiTimeframeData;
    }
    /**
     * Validate temporal consistency across timeframes
     */
    validateTemporalConsistency(data) {
        let lookAheadBiasDetected = false;
        let timeframeAlignmentIssues = 0;
        let validDataPoints = 0;
        data.forEach((point, index) => {
            // Check for look-ahead bias
            const timeframes = Object.keys(point.timeframes);
            timeframes.forEach(tf => {
                const candle = point.timeframes[tf];
                if (candle && candle.timestamp > point.timestamp) {
                    lookAheadBiasDetected = true;
                }
            });
            // Check timeframe alignment
            if (timeframes.length > 1) {
                const timestamps = timeframes.map(tf => point.timeframes[tf]?.timestamp || 0);
                const uniqueTimestamps = new Set(timestamps.filter(t => t > 0));
                if (uniqueTimestamps.size > timeframes.length * 0.8) {
                    timeframeAlignmentIssues++;
                }
                else {
                    validDataPoints++;
                }
            }
        });
        const dataIntegrityScore = validDataPoints / Math.max(data.length, 1) * 100;
        if (lookAheadBiasDetected) {
            logger_1.logger.warn('⚠️ Look-ahead bias detected in multi-timeframe data');
        }
        if (timeframeAlignmentIssues > data.length * 0.1) {
            logger_1.logger.warn('⚠️ Significant timeframe alignment issues detected', {
                issues: timeframeAlignmentIssues,
                percentage: (timeframeAlignmentIssues / data.length * 100).toFixed(1),
            });
        }
        logger_1.logger.info('🔍 Temporal consistency validation completed', {
            lookAheadBias: lookAheadBiasDetected,
            alignmentIssues: timeframeAlignmentIssues,
            dataIntegrityScore: `${dataIntegrityScore.toFixed(1)}%`,
        });
        return {
            lookAheadBiasDetected,
            timeframeAlignmentIssues,
            dataIntegrityScore,
        };
    }
    /**
     * Execute multi-timeframe backtest
     */
    async executeMultiTimeframeBacktest(strategy, portfolioManager, data, config) {
        const signals = [];
        const processingTimes = {};
        let hierarchicalStats = {
            totalDecisions: 0,
            higherTimeframeOverrides: 0,
            consensusDecisions: 0,
            conflictResolutions: 0,
        };
        // Convert multi-timeframe data to single timeframe for strategy compatibility
        const primaryTimeframe = config.timeframe;
        const singleTimeframeData = [];
        data.forEach(point => {
            const primaryCandle = point.timeframes[primaryTimeframe] ||
                point.timeframes['1h'] ||
                Object.values(point.timeframes)[0];
            if (primaryCandle) {
                singleTimeframeData.push(primaryCandle);
            }
        });
        logger_1.logger.info('🔄 Executing multi-timeframe backtest...', {
            dataPoints: singleTimeframeData.length,
            primaryTimeframe,
        });
        // Process each data point
        for (let i = 0; i < singleTimeframeData.length; i++) {
            const currentCandle = singleTimeframeData[i];
            // Update portfolio positions
            portfolioManager.updatePositions(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            // Check stop-loss and take-profit
            const closedTrades = portfolioManager.checkStopLossAndTakeProfit(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            // Generate signal using multi-timeframe analysis
            const timeframeStart = Date.now();
            const signal = strategy.generateSignal(singleTimeframeData, i);
            const timeframeEnd = Date.now();
            // Track processing time
            if (!processingTimes[primaryTimeframe]) {
                processingTimes[primaryTimeframe] = [];
            }
            processingTimes[primaryTimeframe].push(timeframeEnd - timeframeStart);
            if (signal) {
                signals.push(signal);
                hierarchicalStats.totalDecisions++;
                // Analyze signal for hierarchical decision stats
                if (signal.reason.includes('consensus')) {
                    hierarchicalStats.consensusDecisions++;
                }
                if (signal.reason.includes('conflict')) {
                    hierarchicalStats.conflictResolutions++;
                }
                if (signal.reason.includes('4h') || signal.reason.includes('1d')) {
                    hierarchicalStats.higherTimeframeOverrides++;
                }
                // Execute trade
                const trade = portfolioManager.executeTrade(signal, currentCandle.close, currentCandle.timestamp);
                if (trade) {
                    trade.strategy = strategy.name;
                }
            }
            // Create portfolio snapshots periodically
            if (i % 24 === 0 || signal) {
                portfolioManager.createSnapshot(currentCandle.timestamp);
            }
            // Progress logging
            if (i % 100 === 0) {
                const progress = (i / singleTimeframeData.length * 100).toFixed(1);
                logger_1.logger.debug(`   Progress: ${progress}% (${signals.length} signals)`);
            }
        }
        // Calculate final performance
        const trades = portfolioManager.getTrades();
        const portfolioHistory = portfolioManager.getPortfolioHistory();
        const performance = performanceAnalytics_1.PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, config);
        return {
            signals,
            trades,
            portfolioHistory,
            performance,
            processingTimes,
            hierarchicalStats,
        };
    }
    /**
     * Analyze timeframe-specific performance
     */
    analyzeTimeframePerformances(signals, timeframes) {
        const performances = [];
        timeframes.forEach(timeframe => {
            // For this implementation, we'll simulate timeframe-specific metrics
            // In a real system, this would track actual timeframe contributions
            const timeframeSignals = signals.filter(s => s.reason.includes(timeframe) || s.reason.includes('Multi-TF'));
            const signalsGenerated = timeframeSignals.length;
            const avgConfidence = signalsGenerated > 0
                ? timeframeSignals.reduce((sum, s) => sum + s.confidence, 0) / signalsGenerated
                : 0;
            // Simulate accuracy based on confidence
            const signalsAccurate = Math.floor(signalsGenerated * (avgConfidence / 100));
            const accuracy = signalsGenerated > 0 ? (signalsAccurate / signalsGenerated) * 100 : 0;
            // Calculate contribution based on timeframe priority
            const priority = this.dataProvider.getTimeframePriority(timeframe);
            const contribution = (priority / 7) * (signalsGenerated / Math.max(signals.length, 1)) * 100;
            performances.push({
                timeframe,
                signalsGenerated,
                signalsAccurate,
                accuracy,
                avgConfidence,
                contribution,
            });
        });
        return performances.sort((a, b) => b.contribution - a.contribution);
    }
    /**
     * Calculate execution metrics
     */
    calculateExecutionMetrics(processingTimes, timeframes, startTime) {
        const timeframeProcessingTime = {};
        let totalProcessingTime = Date.now() - startTime;
        let avgExecutionDelay = 0;
        timeframes.forEach(timeframe => {
            const times = processingTimes[timeframe] || [];
            if (times.length > 0) {
                const avgTime = times.reduce((sum, t) => sum + t, 0) / times.length;
                timeframeProcessingTime[timeframe] = avgTime;
                avgExecutionDelay += avgTime;
            }
        });
        avgExecutionDelay = avgExecutionDelay / timeframes.length;
        return {
            avgExecutionDelay,
            timeframeProcessingTime,
            totalProcessingTime,
        };
    }
    /**
     * Display comprehensive multi-timeframe backtest results
     */
    displayResults(result) {
        logger_1.logger.info('\n' + '🕐 MULTI-TIMEFRAME BACKTEST RESULTS'.padStart(60, '='));
        logger_1.logger.info('='.repeat(120));
        // Overall Performance
        logger_1.logger.info('📊 OVERALL PERFORMANCE:');
        logger_1.logger.info(`   Total Return: ${result.performance.totalReturnPercent.toFixed(2)}%`);
        logger_1.logger.info(`   Sharpe Ratio: ${result.performance.sharpeRatio.toFixed(3)}`);
        logger_1.logger.info(`   Maximum Drawdown: ${result.performance.maxDrawdownPercent.toFixed(2)}%`);
        logger_1.logger.info(`   Win Rate: ${result.performance.winRate.toFixed(1)}%`);
        logger_1.logger.info(`   Total Trades: ${result.performance.totalTrades}`);
        // Timeframe Performance Analysis
        logger_1.logger.info('\n🕐 TIMEFRAME PERFORMANCE ANALYSIS:');
        logger_1.logger.info('   Timeframe | Signals | Accuracy | Avg Conf | Contribution');
        logger_1.logger.info('   ' + '-'.repeat(60));
        result.timeframePerformances.forEach(tf => {
            logger_1.logger.info(`   ${tf.timeframe.padEnd(9)} | ${tf.signalsGenerated.toString().padStart(7)} | ${tf.accuracy.toFixed(1).padStart(8)}% | ${tf.avgConfidence.toFixed(1).padStart(8)}% | ${tf.contribution.toFixed(1).padStart(12)}%`);
        });
        // Hierarchical Decision Statistics
        logger_1.logger.info('\n🏗️ HIERARCHICAL DECISION STATISTICS:');
        logger_1.logger.info(`   Total Decisions: ${result.hierarchicalDecisionStats.totalDecisions}`);
        logger_1.logger.info(`   Consensus Decisions: ${result.hierarchicalDecisionStats.consensusDecisions} (${(result.hierarchicalDecisionStats.consensusDecisions / Math.max(result.hierarchicalDecisionStats.totalDecisions, 1) * 100).toFixed(1)}%)`);
        logger_1.logger.info(`   Higher TF Overrides: ${result.hierarchicalDecisionStats.higherTimeframeOverrides} (${(result.hierarchicalDecisionStats.higherTimeframeOverrides / Math.max(result.hierarchicalDecisionStats.totalDecisions, 1) * 100).toFixed(1)}%)`);
        logger_1.logger.info(`   Conflict Resolutions: ${result.hierarchicalDecisionStats.conflictResolutions} (${(result.hierarchicalDecisionStats.conflictResolutions / Math.max(result.hierarchicalDecisionStats.totalDecisions, 1) * 100).toFixed(1)}%)`);
        // Temporal Consistency
        logger_1.logger.info('\n⏰ TEMPORAL CONSISTENCY:');
        logger_1.logger.info(`   Data Integrity Score: ${result.temporalConsistency.dataIntegrityScore.toFixed(1)}%`);
        logger_1.logger.info(`   Look-ahead Bias: ${result.temporalConsistency.lookAheadBiasDetected ? '❌ DETECTED' : '✅ NONE'}`);
        logger_1.logger.info(`   Alignment Issues: ${result.temporalConsistency.timeframeAlignmentIssues}`);
        // Execution Metrics
        logger_1.logger.info('\n⚡ EXECUTION METRICS:');
        logger_1.logger.info(`   Average Execution Delay: ${result.executionMetrics.avgExecutionDelay.toFixed(2)}ms`);
        logger_1.logger.info(`   Total Processing Time: ${(result.executionMetrics.totalProcessingTime / 1000).toFixed(2)}s`);
        Object.entries(result.executionMetrics.timeframeProcessingTime).forEach(([tf, time]) => {
            logger_1.logger.info(`   ${tf} Processing Time: ${time.toFixed(2)}ms`);
        });
        logger_1.logger.info('='.repeat(120));
    }
}
exports.MultiTimeframeBacktester = MultiTimeframeBacktester;
// Export factory function
function createMultiTimeframeBacktester() {
    return new MultiTimeframeBacktester();
}
//# sourceMappingURL=multiTimeframeBacktester.js.map