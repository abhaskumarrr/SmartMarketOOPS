"use strict";
/**
 * Backtesting Engine
 * Orchestrates the complete backtesting process with event-driven architecture
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.BacktestingEngine = void 0;
const marketDataProvider_1 = require("./marketDataProvider");
const technicalAnalysis_1 = require("../utils/technicalAnalysis");
const portfolioManager_1 = require("./portfolioManager");
const performanceAnalytics_1 = require("../utils/performanceAnalytics");
const questdbService_1 = require("./questdbService");
const redisStreamsService_1 = require("./redisStreamsService");
const eventDrivenTradingSystem_1 = require("./eventDrivenTradingSystem");
const logger_1 = require("../utils/logger");
class BacktestingEngine {
    constructor(config, strategy) {
        this.marketData = [];
        this.currentIndex = 0;
        this.config = config;
        this.strategy = strategy;
        this.portfolioManager = new portfolioManager_1.PortfolioManager(config);
        // Initialize strategy
        this.strategy.initialize(config);
        logger_1.logger.info('🚀 Backtesting Engine initialized', {
            strategy: strategy.name,
            symbol: config.symbol,
            timeframe: config.timeframe,
            period: `${config.startDate.toISOString().split('T')[0]} to ${config.endDate.toISOString().split('T')[0]}`,
            capital: config.initialCapital,
            leverage: config.leverage,
        });
    }
    /**
     * Run the complete backtesting process
     */
    async run() {
        const startTime = Date.now();
        try {
            logger_1.logger.info('🎯 Starting backtesting process...');
            // Step 1: Initialize infrastructure
            await this.initializeInfrastructure();
            // Step 2: Load and prepare market data
            await this.loadMarketData();
            // Step 3: Process data chronologically
            await this.processMarketData();
            // Step 4: Calculate performance metrics
            const performance = this.calculatePerformance();
            // Step 5: Store results in QuestDB
            await this.storeResults(performance);
            // Step 6: Generate final result
            const result = this.generateResult(startTime, performance);
            logger_1.logger.info('🎉 Backtesting completed successfully', {
                duration: `${(Date.now() - startTime) / 1000}s`,
                totalTrades: result.trades.length,
                finalReturn: `${result.performance.totalReturnPercent.toFixed(2)}%`,
                winRate: `${result.performance.winRate.toFixed(1)}%`,
            });
            return result;
        }
        catch (error) {
            logger_1.logger.error('❌ Backtesting failed:', error);
            throw error;
        }
    }
    /**
     * Initialize infrastructure services
     */
    async initializeInfrastructure() {
        logger_1.logger.info('🔧 Initializing infrastructure...');
        await questdbService_1.questdbService.initialize();
        await redisStreamsService_1.redisStreamsService.initialize();
        logger_1.logger.info('✅ Infrastructure initialized');
    }
    /**
     * Load and enhance market data
     */
    async loadMarketData() {
        logger_1.logger.info('📊 Loading market data...');
        // Fetch historical data
        const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
            symbol: this.config.symbol,
            timeframe: this.config.timeframe,
            startDate: this.config.startDate,
            endDate: this.config.endDate,
            exchange: 'enhanced-mock',
        });
        logger_1.logger.info(`📈 Loaded ${response.data.length} data points`);
        // Store raw market data in QuestDB
        await this.storeMarketData(response.data);
        // Enhance with technical indicators
        this.enhanceMarketData(response.data);
        logger_1.logger.info(`✅ Market data enhanced with technical indicators`);
    }
    /**
     * Store market data in QuestDB
     */
    async storeMarketData(data) {
        logger_1.logger.info('💾 Storing market data in QuestDB...');
        const batchSize = 100;
        for (let i = 0; i < data.length; i += batchSize) {
            const batch = data.slice(i, i + batchSize);
            const questdbBatch = batch.map(point => ({
                timestamp: new Date(point.timestamp),
                symbol: point.symbol,
                exchange: point.exchange,
                timeframe: point.timeframe,
                open: point.open,
                high: point.high,
                low: point.low,
                close: point.close,
                volume: point.volume,
            }));
            await questdbService_1.questdbService.insertMarketDataBatch(questdbBatch);
        }
        logger_1.logger.info(`✅ Stored ${data.length} market data points in QuestDB`);
    }
    /**
     * Enhance market data with technical indicators
     */
    enhanceMarketData(data) {
        const closes = data.map(d => d.close);
        const volumes = data.map(d => d.volume);
        const highs = data.map(d => d.high);
        const lows = data.map(d => d.low);
        // Calculate all technical indicators
        const indicators = technicalAnalysis_1.technicalAnalysis.calculateAllIndicators(closes, volumes);
        // Create enhanced market data
        this.marketData = data.map((point, index) => ({
            ...point,
            indicators: indicators[index],
        }));
    }
    /**
     * Process market data chronologically
     */
    async processMarketData() {
        logger_1.logger.info('⚡ Processing market data chronologically...');
        let signalCount = 0;
        let tradeCount = 0;
        for (let i = 0; i < this.marketData.length; i++) {
            this.currentIndex = i;
            const currentCandle = this.marketData[i];
            // Update portfolio with current prices
            this.portfolioManager.updatePositions(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            // Check for stop loss and take profit triggers
            const closedTrades = this.portfolioManager.checkStopLossAndTakeProfit(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
            // Store closed trades
            for (const trade of closedTrades) {
                await this.storeTrade(trade);
                tradeCount++;
            }
            // Generate trading signal
            const signal = this.strategy.generateSignal(this.marketData, i);
            if (signal && signal.confidence > 0) {
                signalCount++;
                // Publish signal to Redis Streams
                await this.publishTradingSignal(signal);
                // Execute trade
                const trade = this.portfolioManager.executeTrade(signal, currentCandle.close, currentCandle.timestamp);
                if (trade) {
                    trade.strategy = this.strategy.name;
                    await this.storeTrade(trade);
                    tradeCount++;
                }
            }
            // Create portfolio snapshot (every 24 hours or at significant events)
            if (i % 24 === 0 || signal) {
                const snapshot = this.portfolioManager.createSnapshot(currentCandle.timestamp);
                await this.storePortfolioSnapshot(snapshot);
            }
            // Progress logging
            if (i % 100 === 0) {
                const progress = ((i / this.marketData.length) * 100).toFixed(1);
                logger_1.logger.debug(`📊 Progress: ${progress}% (${i}/${this.marketData.length})`);
            }
        }
        logger_1.logger.info(`✅ Processing completed`, {
            dataPoints: this.marketData.length,
            signalsGenerated: signalCount,
            tradesExecuted: tradeCount,
        });
    }
    /**
     * Publish trading signal to Redis Streams
     */
    async publishTradingSignal(signal) {
        try {
            await eventDrivenTradingSystem_1.eventDrivenTradingSystem.publishTradingSignalEvent({
                signalId: signal.id,
                symbol: signal.symbol,
                signalType: signal.type === 'BUY' ? 'ENTRY' : 'EXIT',
                direction: signal.type === 'BUY' ? 'LONG' : 'SHORT',
                strength: signal.confidence > 80 ? 'STRONG' : signal.confidence > 60 ? 'MODERATE' : 'WEAK',
                timeframe: this.config.timeframe,
                price: signal.price,
                confidenceScore: signal.confidence,
                expectedReturn: signal.riskReward ? signal.riskReward * 2 : 4,
                expectedRisk: 2,
                riskRewardRatio: signal.riskReward || 2,
                modelSource: this.strategy.name,
            });
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to publish trading signal:', error);
        }
    }
    /**
     * Store trade in QuestDB
     */
    async storeTrade(trade) {
        try {
            await questdbService_1.questdbService.insertTrade({
                timestamp: new Date(trade.exitTime || trade.entryTime),
                id: trade.id,
                symbol: trade.symbol,
                side: trade.side,
                entryPrice: trade.entryPrice,
                exitPrice: trade.exitPrice || trade.entryPrice,
                quantity: trade.quantity,
                entryTime: new Date(trade.entryTime),
                exitTime: new Date(trade.exitTime || trade.entryTime),
                pnl: trade.pnl,
                pnlPercent: trade.pnlPercent,
                commission: trade.commission,
                strategy: trade.strategy,
                reason: trade.reason,
                duration: trade.duration,
            });
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to store trade:', error);
        }
    }
    /**
     * Store portfolio snapshot in QuestDB
     */
    async storePortfolioSnapshot(snapshot) {
        try {
            await questdbService_1.questdbService.insertPortfolioSnapshot({
                timestamp: new Date(snapshot.timestamp),
                totalValue: snapshot.totalValue,
                cash: snapshot.cash,
                totalPnl: snapshot.totalPnl,
                totalPnlPercent: snapshot.totalPnlPercent,
                drawdown: snapshot.drawdown,
                maxDrawdown: snapshot.maxDrawdown,
                leverage: snapshot.leverage,
                positionCount: snapshot.positions.length,
            });
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to store portfolio snapshot:', error);
        }
    }
    /**
     * Calculate performance metrics
     */
    calculatePerformance() {
        logger_1.logger.info('📊 Calculating performance metrics...');
        const trades = this.portfolioManager.getTrades();
        const portfolioHistory = this.portfolioManager.getPortfolioHistory();
        return performanceAnalytics_1.PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, this.config);
    }
    /**
     * Store final results in QuestDB
     */
    async storeResults(performance) {
        logger_1.logger.info('💾 Storing final results...');
        // Store performance metrics as metrics
        const metrics = [
            { name: 'total_return_percent', value: performance.totalReturnPercent },
            { name: 'annualized_return', value: performance.annualizedReturn },
            { name: 'sharpe_ratio', value: performance.sharpeRatio },
            { name: 'max_drawdown_percent', value: performance.maxDrawdownPercent },
            { name: 'win_rate', value: performance.winRate },
            { name: 'profit_factor', value: performance.profitFactor },
            { name: 'total_trades', value: performance.totalTrades },
        ];
        for (const metric of metrics) {
            await questdbService_1.questdbService.insertMetric({
                timestamp: new Date(),
                name: `backtest_${metric.name}`,
                value: metric.value,
                tags: {
                    strategy: this.strategy.name,
                    symbol: this.config.symbol,
                    timeframe: this.config.timeframe,
                    test_type: 'backtest',
                },
            });
        }
        logger_1.logger.info('✅ Results stored in QuestDB');
    }
    /**
     * Generate final backtest result
     */
    generateResult(startTime, performance) {
        const trades = this.portfolioManager.getTrades();
        const portfolioHistory = this.portfolioManager.getPortfolioHistory();
        const finalPortfolio = portfolioHistory[portfolioHistory.length - 1];
        return {
            config: this.config,
            performance,
            trades,
            portfolioHistory,
            finalPortfolio,
            startTime,
            endTime: Date.now(),
            duration: Date.now() - startTime,
            dataPoints: this.marketData.length,
        };
    }
    /**
     * Generate and log performance report
     */
    generateReport(result) {
        const report = performanceAnalytics_1.PerformanceAnalytics.generateReport(result.performance, result.trades, result.config);
        logger_1.logger.info('\n' + report);
        return report;
    }
}
exports.BacktestingEngine = BacktestingEngine;
//# sourceMappingURL=backtestingEngine.js.map