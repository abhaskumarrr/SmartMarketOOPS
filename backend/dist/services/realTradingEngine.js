"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.RealTradingEngine = void 0;
const dynamicTakeProfitManager_1 = require("./dynamicTakeProfitManager");
const deltaExchangeService_1 = require("./deltaExchangeService");
const logger_1 = require("../utils/logger");
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
class RealTradingEngine {
    constructor(deltaCredentials, config = {}) {
        this.activeTrades = new Map();
        this.closedTrades = [];
        this.isRunning = false;
        this.tradingAssets = ['BTCUSD', 'ETHUSD'];
        this.dailyTradeCount = 0;
        this.lastTradeDate = '';
        this.sessionStartTime = Date.now();
        this.takeProfitManager = new dynamicTakeProfitManager_1.DynamicTakeProfitManager();
        this.deltaService = new deltaExchangeService_1.DeltaExchangeService(deltaCredentials);
        // Enhanced configuration for real trading
        this.config = {
            balanceAllocationPercent: 75, // Use 75% of available balance
            maxLeverage: 100, // Delta Exchange testnet supports max 100x leverage
            riskPerTrade: 40, // Start with high risk
            targetTradesPerDay: 4,
            targetWinRate: 75,
            mlConfidenceThreshold: 80,
            signalScoreThreshold: 72,
            qualityScoreThreshold: 78,
            maxDrawdownPercent: 20,
            tradingAssets: ['BTCUSD', 'ETHUSD'],
            checkIntervalMs: 30000, // 30 seconds
            progressReportIntervalMs: 60000, // 1 minute
            ...config
        };
        // Initialize portfolio
        this.portfolio = {
            totalBalance: 0,
            allocatedBalance: 0,
            currentBalance: 0,
            initialBalance: 0,
            peakBalance: 0,
            totalPnl: 0,
            unrealizedPnl: 0,
            realizedPnl: 0,
            totalTrades: 0,
            winningTrades: 0,
            losingTrades: 0,
            winRate: 0,
            currentDrawdown: 0,
            maxDrawdown: 0,
            leverage: this.config.maxLeverage,
            riskPerTrade: this.config.riskPerTrade
        };
        this.tradingAssets = this.config.tradingAssets;
    }
    /**
     * Start real trading with actual Delta Exchange orders
     */
    async startRealTrading() {
        try {
            logger_1.logger.info('\n🚀 STARTING REAL TRADING ENGINE');
            logger_1.logger.info('⚠️  WARNING: This will place REAL ORDERS on Delta Exchange!');
            logger_1.logger.info('💰 Using REAL MONEY - All trades will be executed live!');
            // Initialize balance from Delta Exchange
            await this.initializeRealBalance();
            // Wait for Delta service to be ready
            await this.waitForDeltaService();
            this.isRunning = true;
            logger_1.logger.info('✅ Real trading engine started successfully');
            logger_1.logger.info(`💰 Allocated Balance: $${this.portfolio.allocatedBalance.toFixed(2)}`);
            logger_1.logger.info(`⚡ Max Leverage: ${this.portfolio.leverage}x`);
            logger_1.logger.info(`🎲 Risk per Trade: ${this.portfolio.riskPerTrade}%`);
            // Start trading loop
            await this.runTradingLoop();
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start real trading:', error);
            throw error;
        }
    }
    /**
     * Initialize real balance from Delta Exchange
     */
    async initializeRealBalance() {
        try {
            logger_1.logger.info('💰 Fetching real balance from Delta Exchange...');
            // Wait for Delta service to be ready
            let attempts = 0;
            while ((!this.deltaService.isReady()) && attempts < 30) {
                await this.delay(1000);
                attempts++;
            }
            if (!this.deltaService.isReady()) {
                throw new Error('Delta Exchange service failed to initialize');
            }
            const balances = await this.deltaService.getBalances();
            if (balances.length === 0) {
                throw new Error('No balances found in Delta Exchange testnet account. Please fund your testnet account first.');
            }
            // Find USD balance (settling asset for perpetual futures)
            const usdBalance = balances.find(b => b.asset_symbol === 'USD' || b.asset_symbol === 'USDT');
            if (usdBalance) {
                this.portfolio.totalBalance = parseFloat(usdBalance.available_balance);
                logger_1.logger.info(`✅ Found USD balance: $${this.portfolio.totalBalance.toFixed(2)}`);
            }
            else {
                // Use first available balance
                this.portfolio.totalBalance = parseFloat(balances[0].available_balance);
                logger_1.logger.info(`✅ Using ${balances[0].asset_symbol} balance: ${this.portfolio.totalBalance.toFixed(2)}`);
            }
            // Calculate allocation
            this.portfolio.allocatedBalance = this.portfolio.totalBalance * (this.config.balanceAllocationPercent / 100);
            this.portfolio.initialBalance = this.portfolio.allocatedBalance;
            this.portfolio.currentBalance = this.portfolio.allocatedBalance;
            this.portfolio.peakBalance = this.portfolio.allocatedBalance;
            logger_1.logger.info(`💰 Real balance allocation complete: $${this.portfolio.allocatedBalance.toFixed(2)} (${this.config.balanceAllocationPercent}%)`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize real balance:', error);
            throw error;
        }
    }
    /**
     * Wait for Delta Exchange service to be ready
     */
    async waitForDeltaService() {
        let attempts = 0;
        while (!this.deltaService.isReady() && attempts < 30) {
            logger_1.logger.info(`🔄 Waiting for Delta Exchange service... (${attempts + 1}/30)`);
            await this.delay(2000);
            attempts++;
        }
        if (!this.deltaService.isReady()) {
            throw new Error('Delta Exchange service not ready after 60 seconds');
        }
        logger_1.logger.info('✅ Delta Exchange service is ready for real trading');
    }
    /**
     * Main trading loop for real trading
     */
    async runTradingLoop() {
        logger_1.logger.info('\n🔄 Starting real trading loop...');
        let lastProgressReport = Date.now();
        while (this.isRunning) {
            try {
                // Update existing trades with real market data
                await this.updateExistingTrades();
                // Generate new trading signals
                await this.generateAndExecuteSignals();
                // Progress reporting
                if (Date.now() - lastProgressReport >= this.config.progressReportIntervalMs) {
                    this.generateProgressReport();
                    lastProgressReport = Date.now();
                }
                // Check if we should stop (daily targets met, etc.)
                if (this.shouldStopTrading()) {
                    logger_1.logger.info('🎯 Daily trading targets achieved, stopping for today');
                    break;
                }
                // Wait before next iteration
                await this.delay(this.config.checkIntervalMs);
            }
            catch (error) {
                logger_1.logger.error('❌ Error in trading loop:', error);
                await this.delay(5000); // Wait 5 seconds before retrying
            }
        }
        // Generate final report
        this.generateFinalReport();
    }
    /**
     * Delay utility
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    /**
     * Stop real trading
     */
    async stopRealTrading() {
        logger_1.logger.info('🛑 Stopping real trading engine...');
        this.isRunning = false;
        // Close all open positions if needed
        // await this.closeAllPositions();
        logger_1.logger.info('✅ Real trading engine stopped');
    }
    /**
     * Update existing trades with real market data and manage exits
     */
    async updateExistingTrades() {
        for (const trade of this.activeTrades.values()) {
            try {
                // Get real market data from Delta Exchange
                const marketData = await this.deltaService.getMarketData(trade.symbol);
                if (!marketData)
                    continue;
                trade.currentPrice = marketData.price;
                // Calculate unrealized P&L
                const priceChange = trade.side === 'BUY'
                    ? trade.currentPrice - trade.entryPrice
                    : trade.entryPrice - trade.currentPrice;
                trade.unrealizedPnl = (priceChange / trade.entryPrice) * trade.size * this.portfolio.leverage;
                // Update max profit/loss tracking
                trade.maxProfit = Math.max(trade.maxProfit, trade.unrealizedPnl);
                trade.maxLoss = Math.min(trade.maxLoss, trade.unrealizedPnl);
                // Check for take profit exits
                await this.checkTakeProfitExits(trade);
                // Check for stop loss
                await this.checkStopLoss(trade);
            }
            catch (error) {
                logger_1.logger.error(`❌ Error updating trade ${trade.id}:`, error);
            }
        }
    }
    /**
     * Check and execute take profit exits with REAL orders
     */
    async checkTakeProfitExits(trade) {
        for (const level of trade.takeProfitLevels) {
            if (level.executed)
                continue;
            const shouldExit = trade.side === 'BUY'
                ? trade.currentPrice >= level.priceTarget
                : trade.currentPrice <= level.priceTarget;
            if (shouldExit) {
                await this.executePartialExit(trade, level);
            }
        }
    }
    /**
     * Execute partial exit with REAL Delta Exchange order
     */
    async executePartialExit(trade, level) {
        try {
            const exitSize = trade.size * (level.percentage / 100);
            const exitSide = trade.side === 'BUY' ? 'SELL' : 'BUY';
            // Place REAL order on Delta Exchange
            const orderRequest = {
                product_id: trade.productId,
                size: exitSize,
                side: exitSide.toLowerCase(),
                order_type: 'market_order',
                time_in_force: 'ioc'
            };
            logger_1.logger.info(`🎯 Executing REAL partial exit: ${exitSide} ${exitSize.toFixed(4)} ${trade.symbol} @ $${trade.currentPrice.toFixed(2)}`);
            const order = await this.deltaService.placeOrder(orderRequest);
            if (order) {
                // Calculate realized P&L
                const priceChange = trade.side === 'BUY'
                    ? trade.currentPrice - trade.entryPrice
                    : trade.entryPrice - trade.currentPrice;
                const realizedPnl = (priceChange / trade.entryPrice) * exitSize * this.portfolio.leverage;
                // Record partial exit
                const partialExit = {
                    level: level.riskRewardRatio,
                    size: exitSize,
                    price: trade.currentPrice,
                    pnl: realizedPnl,
                    timestamp: Date.now(),
                    orderId: order.id
                };
                trade.partialExits.push(partialExit);
                trade.realizedPnl += realizedPnl;
                trade.size -= exitSize; // Reduce remaining position size
                level.executed = true;
                // Update portfolio
                this.portfolio.currentBalance += realizedPnl;
                this.portfolio.realizedPnl += realizedPnl;
                logger_1.logger.info(`✅ REAL partial exit executed: $${realizedPnl.toFixed(2)} P&L, Order ID: ${order.id}`);
                // Close trade if all levels executed or position size too small
                if (trade.size < 0.001 || trade.takeProfitLevels.every(l => l.executed)) {
                    await this.closeTrade(trade, 'All take profit levels hit');
                }
            }
            else {
                logger_1.logger.error(`❌ Failed to place partial exit order for ${trade.symbol}`);
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Error executing partial exit for ${trade.symbol}:`, error);
        }
    }
    /**
     * Check and execute stop loss with REAL order
     */
    async checkStopLoss(trade) {
        const shouldStopLoss = trade.side === 'BUY'
            ? trade.currentPrice <= trade.stopLoss
            : trade.currentPrice >= trade.stopLoss;
        if (shouldStopLoss) {
            await this.executeStopLoss(trade);
        }
    }
    /**
     * Execute stop loss with REAL Delta Exchange order
     */
    async executeStopLoss(trade) {
        try {
            const exitSide = trade.side === 'BUY' ? 'SELL' : 'BUY';
            // Place REAL stop loss order
            const orderRequest = {
                product_id: trade.productId,
                size: trade.size,
                side: exitSide.toLowerCase(),
                order_type: 'market_order',
                time_in_force: 'ioc'
            };
            logger_1.logger.info(`🛑 Executing REAL stop loss: ${exitSide} ${trade.size.toFixed(4)} ${trade.symbol} @ $${trade.currentPrice.toFixed(2)}`);
            const order = await this.deltaService.placeOrder(orderRequest);
            if (order) {
                trade.stopLossOrderId = order.id;
                await this.closeTrade(trade, `Stop loss hit - Order ID: ${order.id}`);
                logger_1.logger.info(`✅ REAL stop loss executed, Order ID: ${order.id}`);
            }
            else {
                logger_1.logger.error(`❌ Failed to place stop loss order for ${trade.symbol}`);
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Error executing stop loss for ${trade.symbol}:`, error);
        }
    }
    /**
     * Generate and execute trading signals with REAL orders
     */
    async generateAndExecuteSignals() {
        // Check daily trade limit
        const today = new Date().toDateString();
        if (this.lastTradeDate !== today) {
            this.dailyTradeCount = 0;
            this.lastTradeDate = today;
        }
        if (this.dailyTradeCount >= this.config.targetTradesPerDay) {
            return; // Daily limit reached
        }
        // Generate signals for each asset
        for (const asset of this.tradingAssets) {
            try {
                const signal = await this.generateTradingSignal(asset);
                if (signal && this.shouldExecuteSignal(signal)) {
                    await this.executeRealTrade(signal);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ Error generating signal for ${asset}:`, error);
            }
        }
    }
    /**
     * Generate trading signal (simplified for demo)
     */
    async generateTradingSignal(asset) {
        try {
            const marketData = await this.deltaService.getMarketData(asset);
            if (!marketData)
                return null;
            // Simplified signal generation (replace with your ML model)
            const mlConfidence = 80 + Math.random() * 15; // 80-95%
            const signalScore = 70 + Math.random() * 25; // 70-95
            const qualityScore = 75 + Math.random() * 20; // 75-95
            if (mlConfidence < this.config.mlConfidenceThreshold ||
                signalScore < this.config.signalScoreThreshold ||
                qualityScore < this.config.qualityScoreThreshold) {
                return null;
            }
            // Random signal direction (replace with actual ML prediction)
            const signalType = Math.random() > 0.5 ? 'BUY' : 'SELL';
            // Calculate position size based on risk
            const riskAmount = this.portfolio.currentBalance * (this.portfolio.riskPerTrade / 100);
            const stopLossDistance = marketData.price * 0.02; // 2% stop loss
            const positionSize = (riskAmount / stopLossDistance) * this.portfolio.leverage;
            const signal = {
                id: `${asset}_${Date.now()}`,
                symbol: asset,
                type: signalType,
                price: marketData.price,
                quantity: positionSize,
                timestamp: Date.now(),
                confidence: mlConfidence,
                strategy: 'real_trading_engine',
                reason: `ML confidence: ${mlConfidence.toFixed(1)}%, Signal score: ${signalScore.toFixed(1)}, Quality: ${qualityScore.toFixed(1)}`,
                stopLoss: signalType === 'BUY'
                    ? marketData.price - stopLossDistance
                    : marketData.price + stopLossDistance
            };
            return signal;
        }
        catch (error) {
            logger_1.logger.error(`❌ Error generating signal for ${asset}:`, error);
            return null;
        }
    }
    /**
     * Check if signal should be executed
     */
    shouldExecuteSignal(signal) {
        // Check if we already have a position in this asset
        const existingTrade = Array.from(this.activeTrades.values())
            .find(trade => trade.symbol === signal.symbol);
        if (existingTrade) {
            return false; // Don't open multiple positions in same asset
        }
        // Check risk limits
        const riskAmount = this.portfolio.currentBalance * (this.portfolio.riskPerTrade / 100);
        const maxPositionValue = riskAmount * this.portfolio.leverage;
        const signalValue = signal.quantity * signal.price;
        if (signalValue > maxPositionValue) {
            logger_1.logger.warn(`⚠️ Signal value too large: $${signalValue.toFixed(2)} > $${maxPositionValue.toFixed(2)}`);
            return false;
        }
        return true;
    }
    /**
     * Execute real trade with actual Delta Exchange order
     */
    async executeRealTrade(signal) {
        try {
            // Get product ID for the symbol
            const productId = this.getProductId(signal.symbol);
            if (!productId) {
                logger_1.logger.error(`❌ Product ID not found for ${signal.symbol}`);
                return;
            }
            // Place REAL order on Delta Exchange
            const orderRequest = {
                product_id: productId,
                size: signal.quantity,
                side: signal.type.toLowerCase(),
                order_type: 'market_order',
                time_in_force: 'ioc'
            };
            logger_1.logger.info(`🚀 Placing REAL order: ${signal.type} ${signal.quantity.toFixed(4)} ${signal.symbol} @ $${signal.price.toFixed(2)}`);
            logger_1.logger.info(`⚠️  WARNING: This is a REAL order with REAL money!`);
            const order = await this.deltaService.placeOrder(orderRequest);
            if (order) {
                // Create real trade record
                await this.createRealTrade(signal, order, productId);
                this.dailyTradeCount++;
                logger_1.logger.info(`✅ REAL order placed successfully! Order ID: ${order.id}`);
                logger_1.logger.info(`💰 Position opened with REAL money on Delta Exchange`);
            }
            else {
                logger_1.logger.error(`❌ Failed to place REAL order for ${signal.symbol}`);
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Error executing real trade for ${signal.symbol}:`, error);
        }
    }
    /**
     * Create real trade record after successful order placement
     */
    async createRealTrade(signal, order, productId) {
        try {
            // Generate dynamic take profit levels
            const marketRegime = {
                type: 'TRENDING',
                strength: 75,
                direction: signal.type === 'BUY' ? 'UP' : 'DOWN',
                volatility: 0.03,
                volume: 1.2,
            };
            const takeProfitConfig = {
                asset: signal.symbol,
                entryPrice: signal.price,
                stopLoss: signal.stopLoss,
                positionSize: signal.quantity,
                side: signal.type,
                marketRegime,
                momentum: signal.type === 'BUY' ? 50 : -50,
                volume: 1.2,
            };
            const takeProfitLevels = this.takeProfitManager.generateDynamicTakeProfitLevels(takeProfitConfig);
            const trade = {
                id: signal.id,
                symbol: signal.symbol,
                side: signal.type,
                size: signal.quantity,
                entryPrice: signal.price,
                entryTime: signal.timestamp,
                status: 'OPEN',
                deltaOrderId: order.id,
                productId,
                takeProfitLevels,
                partialExits: [],
                stopLoss: signal.stopLoss,
                currentPrice: signal.price,
                unrealizedPnl: 0,
                realizedPnl: 0,
                maxProfit: 0,
                maxLoss: 0,
            };
            this.activeTrades.set(trade.id, trade);
            this.portfolio.totalTrades++;
            logger_1.logger.info(`🔥 REAL Trade Opened: ${trade.side} ${trade.size.toFixed(4)} ${trade.symbol} @ $${trade.entryPrice.toFixed(2)}`);
            logger_1.logger.info(`   Delta Order ID: ${order.id}`);
            logger_1.logger.info(`   Product ID: ${productId}`);
            logger_1.logger.info(`   Stop Loss: $${trade.stopLoss.toFixed(2)}`);
            logger_1.logger.info(`   Take Profit Levels: ${takeProfitLevels.length} levels`);
            logger_1.logger.info(`   💰 REAL MONEY POSITION ACTIVE ON DELTA EXCHANGE`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to create real trade record:', error);
        }
    }
    /**
     * Get product ID for symbol
     */
    getProductId(symbol) {
        const productIds = {
            'BTCUSD': 84, // BTC perpetual futures on Delta Exchange testnet
            'ETHUSD': 1699 // ETH perpetual futures on Delta Exchange testnet
        };
        return productIds[symbol] || null;
    }
    /**
     * Close a real trade
     */
    async closeTrade(trade, reason) {
        try {
            trade.status = 'CLOSED';
            trade.exitPrice = trade.currentPrice;
            trade.exitTime = Date.now();
            trade.reason = reason;
            // Calculate final P&L (including any partial exits)
            const remainingSize = trade.size;
            if (remainingSize > 0) {
                const priceChange = trade.side === 'BUY'
                    ? trade.currentPrice - trade.entryPrice
                    : trade.entryPrice - trade.currentPrice;
                const finalPnl = (priceChange / trade.entryPrice) * remainingSize * this.portfolio.leverage;
                trade.realizedPnl += finalPnl;
                this.portfolio.currentBalance += finalPnl;
                this.portfolio.realizedPnl += finalPnl;
            }
            trade.pnl = trade.realizedPnl;
            this.portfolio.totalPnl += trade.pnl;
            if (trade.pnl > 0) {
                this.portfolio.winningTrades++;
            }
            else {
                this.portfolio.losingTrades++;
            }
            // Move to closed trades
            this.activeTrades.delete(trade.id);
            this.closedTrades.push(trade);
            logger_1.logger.info(`✅ REAL Trade Closed: ${trade.symbol} - ${reason}`);
            logger_1.logger.info(`   Final P&L: $${trade.pnl.toFixed(2)} (${((trade.pnl / this.portfolio.initialBalance) * 100).toFixed(2)}%)`);
            logger_1.logger.info(`   Partial Exits: ${trade.partialExits.length}`);
            logger_1.logger.info(`   💰 REAL MONEY POSITION CLOSED ON DELTA EXCHANGE`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to close real trade:', error);
        }
    }
    /**
     * Check if trading should stop
     */
    shouldStopTrading() {
        // Stop if daily trade limit reached
        if (this.dailyTradeCount >= this.config.targetTradesPerDay) {
            return true;
        }
        // Stop if max drawdown exceeded
        if (this.portfolio.currentDrawdown >= this.config.maxDrawdownPercent) {
            logger_1.logger.warn(`⚠️ Max drawdown exceeded: ${this.portfolio.currentDrawdown.toFixed(1)}%`);
            return true;
        }
        return false;
    }
    /**
     * Generate progress report
     */
    generateProgressReport() {
        const elapsed = (Date.now() - this.sessionStartTime) / 1000 / 60; // minutes
        const returnPercent = ((this.portfolio.currentBalance - this.portfolio.initialBalance) / this.portfolio.initialBalance) * 100;
        logger_1.logger.info('\n📊 REAL TRADING PROGRESS REPORT');
        logger_1.logger.info('─'.repeat(50));
        logger_1.logger.info(`⏱️  Session Time: ${elapsed.toFixed(1)} minutes`);
        logger_1.logger.info(`💰 Balance: $${this.portfolio.currentBalance.toFixed(2)} (${returnPercent.toFixed(1)}%)`);
        logger_1.logger.info(`📈 Realized P&L: $${this.portfolio.realizedPnl.toFixed(2)}`);
        logger_1.logger.info(`📊 Unrealized P&L: $${this.portfolio.unrealizedPnl.toFixed(2)}`);
        logger_1.logger.info(`🎯 Daily Trades: ${this.dailyTradeCount}/${this.config.targetTradesPerDay}`);
        logger_1.logger.info(`📊 Win Rate: ${this.portfolio.winRate.toFixed(1)}% (Target: ${this.config.targetWinRate}%)`);
        logger_1.logger.info(`🔄 Active Trades: ${this.activeTrades.size}`);
        logger_1.logger.info(`📉 Drawdown: ${this.portfolio.currentDrawdown.toFixed(1)}%`);
        logger_1.logger.info(`⚡ Current Leverage: ${this.portfolio.leverage}x`);
        logger_1.logger.info(`🎲 Current Risk: ${this.portfolio.riskPerTrade}%`);
        logger_1.logger.info(`💰 REAL MONEY POSITIONS ON DELTA EXCHANGE`);
    }
    /**
     * Generate final report
     */
    generateFinalReport() {
        logger_1.logger.info('\n' + '🎉 REAL TRADING FINAL REPORT'.padStart(80, '='));
        logger_1.logger.info('='.repeat(120));
        logger_1.logger.info('💰 ALL TRADES EXECUTED WITH REAL MONEY ON DELTA EXCHANGE');
        logger_1.logger.info('='.repeat(120));
        // Portfolio Summary
        logger_1.logger.info('💰 PORTFOLIO SUMMARY:');
        logger_1.logger.info(`   Initial Balance: $${this.portfolio.initialBalance.toFixed(2)}`);
        logger_1.logger.info(`   Final Balance: $${this.portfolio.currentBalance.toFixed(2)}`);
        logger_1.logger.info(`   Realized P&L: $${this.portfolio.realizedPnl.toFixed(2)}`);
        logger_1.logger.info(`   Unrealized P&L: $${this.portfolio.unrealizedPnl.toFixed(2)}`);
        logger_1.logger.info(`   Total P&L: $${this.portfolio.totalPnl.toFixed(2)}`);
        logger_1.logger.info(`   Return: ${((this.portfolio.currentBalance - this.portfolio.initialBalance) / this.portfolio.initialBalance * 100).toFixed(2)}%`);
        logger_1.logger.info(`   Max Drawdown: ${this.portfolio.maxDrawdown.toFixed(2)}%`);
        // Trading Statistics
        logger_1.logger.info('\n📊 TRADING STATISTICS:');
        logger_1.logger.info(`   Total Trades: ${this.portfolio.totalTrades}`);
        logger_1.logger.info(`   Closed Trades: ${this.closedTrades.length}`);
        logger_1.logger.info(`   Active Trades: ${this.activeTrades.size}`);
        logger_1.logger.info(`   Winning Trades: ${this.portfolio.winningTrades}`);
        logger_1.logger.info(`   Losing Trades: ${this.portfolio.losingTrades}`);
        logger_1.logger.info(`   Win Rate: ${this.portfolio.winRate.toFixed(1)}%`);
        logger_1.logger.info('\n🚀 REAL TRADING SYSTEM RESULTS:');
        if (this.portfolio.totalPnl > 0) {
            logger_1.logger.info('   ✅ PROFITABLE: Real trading system generated positive returns');
            logger_1.logger.info('   💰 REAL MONEY PROFITS ACHIEVED ON DELTA EXCHANGE');
        }
        else {
            logger_1.logger.info('   ❌ LOSS: Real trading system generated losses');
            logger_1.logger.info('   💸 REAL MONEY LOSSES ON DELTA EXCHANGE');
        }
        logger_1.logger.info('='.repeat(120));
    }
}
exports.RealTradingEngine = RealTradingEngine;
//# sourceMappingURL=realTradingEngine.js.map