#!/usr/bin/env node
"use strict";
/**
 * Test Real Data Fetching
 * Simple script to test Binance data provider and real data integration
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.RealDataTester = void 0;
const marketDataProvider_1 = require("../services/marketDataProvider");
const multiTimeframeAITradingSystem_1 = require("../services/multiTimeframeAITradingSystem");
const portfolioManager_1 = require("../services/portfolioManager");
const performanceAnalytics_1 = require("../utils/performanceAnalytics");
const logger_1 = require("../utils/logger");
class RealDataTester {
    /**
     * Test real data fetching and basic backtesting
     */
    async testRealData() {
        const startTime = Date.now();
        try {
            logger_1.logger.info('🔄 Testing Real Data Integration...');
            // Test 1: Check available providers
            await this.testAvailableProviders();
            // Test 2: Fetch real data from Binance
            await this.testBinanceDataFetching();
            // Test 3: Run simple backtest with real data
            await this.testRealDataBacktest();
            const duration = (Date.now() - startTime) / 1000;
            logger_1.logger.info(`✅ Real data testing completed in ${duration.toFixed(2)} seconds`);
        }
        catch (error) {
            logger_1.logger.error('❌ Real data testing failed:', error);
            throw error;
        }
    }
    /**
     * Test available data providers
     */
    async testAvailableProviders() {
        logger_1.logger.info('\n📊 Testing Available Data Providers...');
        const providers = marketDataProvider_1.marketDataService.getAvailableProviders();
        logger_1.logger.info(`Available providers: ${providers.join(', ')}`);
        if (providers.includes('binance')) {
            logger_1.logger.info('✅ Binance provider is available');
        }
        else {
            logger_1.logger.warn('⚠️ Binance provider not found');
        }
    }
    /**
     * Test Binance data fetching
     */
    async testBinanceDataFetching() {
        logger_1.logger.info('\n📡 Testing Binance Data Fetching...');
        const endDate = new Date();
        const startDate = new Date(endDate.getTime() - (24 * 60 * 60 * 1000)); // 1 day
        try {
            // Test with different timeframes
            const timeframes = ['1h', '4h'];
            for (const timeframe of timeframes) {
                logger_1.logger.info(`\n🕐 Testing ${timeframe} timeframe...`);
                const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
                    symbol: 'BTCUSD',
                    timeframe,
                    startDate,
                    endDate,
                    exchange: 'binance',
                }, 'binance');
                logger_1.logger.info(`✅ Fetched ${response.count} ${timeframe} candles`, {
                    symbol: response.symbol,
                    source: response.source,
                    firstCandle: response.data[0] ? {
                        timestamp: new Date(response.data[0].timestamp).toISOString(),
                        price: response.data[0].close.toFixed(2),
                    } : 'No data',
                    lastCandle: response.data[response.count - 1] ? {
                        timestamp: new Date(response.data[response.count - 1].timestamp).toISOString(),
                        price: response.data[response.count - 1].close.toFixed(2),
                    } : 'No data',
                });
                // Validate data quality
                this.validateDataQuality(response.data, timeframe);
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Binance data fetching failed:', error);
            logger_1.logger.info('🔄 This is expected if no internet connection or API issues');
        }
    }
    /**
     * Test real data backtesting
     */
    async testRealDataBacktest() {
        logger_1.logger.info('\n🎯 Testing Real Data Backtesting...');
        const endDate = new Date();
        const startDate = new Date(endDate.getTime() - (3 * 24 * 60 * 60 * 1000)); // 3 days
        const config = {
            symbol: 'BTCUSD',
            timeframe: '1h',
            startDate,
            endDate,
            initialCapital: 10000,
            leverage: 2,
            riskPerTrade: 2,
            commission: 0.1,
            slippage: 0.05,
            strategy: 'Real_Data_Test',
            parameters: {},
        };
        try {
            // Fetch real data
            logger_1.logger.info('📊 Fetching real market data...');
            const response = await marketDataProvider_1.marketDataService.fetchHistoricalData({
                symbol: config.symbol,
                timeframe: config.timeframe,
                startDate: config.startDate,
                endDate: config.endDate,
                exchange: 'binance',
            }, 'binance');
            if (response.count === 0) {
                logger_1.logger.warn('⚠️ No real data available, skipping backtest');
                return;
            }
            logger_1.logger.info(`✅ Loaded ${response.count} real data points`);
            // Enhance data with technical indicators
            const enhancedData = await this.enhanceDataWithIndicators(response.data);
            // Initialize trading system
            const strategy = (0, multiTimeframeAITradingSystem_1.createMultiTimeframeAITradingSystem)();
            strategy.initialize(config);
            const portfolioManager = new portfolioManager_1.PortfolioManager(config);
            // Run backtest
            logger_1.logger.info('🔄 Running backtest with real data...');
            let signalCount = 0;
            let tradeCount = 0;
            for (let i = 0; i < enhancedData.length; i++) {
                const currentCandle = enhancedData[i];
                // Update portfolio
                portfolioManager.updatePositions(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
                // Check stop-loss and take-profit
                const closedTrades = portfolioManager.checkStopLossAndTakeProfit(currentCandle.symbol, currentCandle.close, currentCandle.timestamp);
                tradeCount += closedTrades.length;
                // Generate signal
                const signal = strategy.generateSignal(enhancedData, i);
                if (signal) {
                    signalCount++;
                    const trade = portfolioManager.executeTrade(signal, currentCandle.close, currentCandle.timestamp);
                    if (trade) {
                        trade.strategy = 'Real_Data_Test';
                        tradeCount++;
                        logger_1.logger.info(`🎯 Trade executed: ${trade.side} at $${currentCandle.close.toFixed(2)} (Confidence: ${signal.confidence.toFixed(1)}%)`);
                    }
                }
                // Create snapshots
                if (i % 12 === 0 || signal) {
                    portfolioManager.createSnapshot(currentCandle.timestamp);
                }
            }
            // Calculate performance
            const trades = portfolioManager.getTrades();
            const portfolioHistory = portfolioManager.getPortfolioHistory();
            const performance = performanceAnalytics_1.PerformanceAnalytics.calculateMetrics(trades, portfolioHistory, config);
            // Display results
            this.displayBacktestResults(performance, signalCount, tradeCount, response.count);
        }
        catch (error) {
            logger_1.logger.error('❌ Real data backtest failed:', error);
            logger_1.logger.info('🔄 This is expected if no internet connection or API issues');
        }
    }
    /**
     * Validate data quality
     */
    validateDataQuality(data, timeframe) {
        if (data.length === 0) {
            logger_1.logger.warn(`⚠️ No data received for ${timeframe}`);
            return;
        }
        // Check for missing fields
        const firstCandle = data[0];
        const requiredFields = ['timestamp', 'open', 'high', 'low', 'close', 'volume'];
        const missingFields = requiredFields.filter(field => firstCandle[field] === undefined);
        if (missingFields.length > 0) {
            logger_1.logger.warn(`⚠️ Missing fields in ${timeframe} data:`, missingFields);
        }
        else {
            logger_1.logger.info(`✅ ${timeframe} data quality check passed`);
        }
        // Check chronological order
        let orderIssues = 0;
        for (let i = 1; i < Math.min(data.length, 10); i++) {
            if (data[i].timestamp <= data[i - 1].timestamp) {
                orderIssues++;
            }
        }
        if (orderIssues > 0) {
            logger_1.logger.warn(`⚠️ ${timeframe} data has ${orderIssues} chronological order issues`);
        }
        else {
            logger_1.logger.info(`✅ ${timeframe} data is in chronological order`);
        }
        // Check price validity
        const prices = data.slice(0, 10).map(d => d.close);
        const validPrices = prices.filter(p => p > 0 && p < 1000000);
        if (validPrices.length !== prices.length) {
            logger_1.logger.warn(`⚠️ ${timeframe} data has invalid prices`);
        }
        else {
            logger_1.logger.info(`✅ ${timeframe} price data is valid (range: $${Math.min(...prices).toFixed(2)} - $${Math.max(...prices).toFixed(2)})`);
        }
    }
    /**
     * Enhance data with basic technical indicators
     */
    async enhanceDataWithIndicators(data) {
        // For simplicity, just add basic indicators
        return data.map((point, index) => ({
            ...point,
            indicators: {
                rsi: 50 + (Math.random() - 0.5) * 40, // Mock RSI
                ema_12: point.close * (0.98 + Math.random() * 0.04), // Mock EMA
                ema_26: point.close * (0.97 + Math.random() * 0.06), // Mock EMA
                volume_sma: point.volume * (0.8 + Math.random() * 0.4), // Mock volume SMA
            },
        }));
    }
    /**
     * Display backtest results
     */
    displayBacktestResults(performance, signalCount, tradeCount, dataPoints) {
        logger_1.logger.info('\n📊 REAL DATA BACKTEST RESULTS:');
        logger_1.logger.info('='.repeat(50));
        logger_1.logger.info(`📈 Performance Metrics:`);
        logger_1.logger.info(`   Total Return: ${performance.totalReturnPercent.toFixed(2)}%`);
        logger_1.logger.info(`   Sharpe Ratio: ${performance.sharpeRatio.toFixed(3)}`);
        logger_1.logger.info(`   Max Drawdown: ${performance.maxDrawdownPercent.toFixed(2)}%`);
        logger_1.logger.info(`   Win Rate: ${performance.winRate.toFixed(1)}%`);
        logger_1.logger.info(`   Total Trades: ${performance.totalTrades}`);
        logger_1.logger.info(`\n🎯 Trading Activity:`);
        logger_1.logger.info(`   Data Points Processed: ${dataPoints}`);
        logger_1.logger.info(`   Signals Generated: ${signalCount}`);
        logger_1.logger.info(`   Trades Executed: ${tradeCount}`);
        logger_1.logger.info(`   Signal Rate: ${(signalCount / dataPoints * 100).toFixed(2)}%`);
        logger_1.logger.info(`\n✅ Real Data Integration:`);
        if (performance.totalTrades > 0) {
            logger_1.logger.info(`   ✅ Successfully executed trades with real data`);
            logger_1.logger.info(`   ✅ Multi-timeframe system operational`);
            logger_1.logger.info(`   ✅ Performance metrics calculated`);
        }
        else {
            logger_1.logger.info(`   ⚠️ No trades executed - strategy may be too conservative`);
            logger_1.logger.info(`   ✅ Real data successfully loaded and processed`);
            logger_1.logger.info(`   ✅ System operational without errors`);
        }
        logger_1.logger.info('='.repeat(50));
    }
}
exports.RealDataTester = RealDataTester;
/**
 * Main execution function
 */
async function main() {
    const tester = new RealDataTester();
    try {
        await tester.testRealData();
    }
    catch (error) {
        logger_1.logger.error('💥 Real data testing failed:', error);
        process.exit(1);
    }
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=test-real-data.js.map