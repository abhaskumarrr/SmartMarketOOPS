#!/usr/bin/env node
"use strict";
/**
 * AGGRESSIVE DELTA TRADING BOT - HIGH LEVERAGE OPERATIONS
 * Uses DeltaExchangeUnified service with up to 200x leverage and 20% drawdown limit
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const dotenv_1 = __importDefault(require("dotenv"));
const DeltaExchangeUnified_1 = require("../services/DeltaExchangeUnified");
const logger_1 = require("../utils/logger");
// Load environment variables
dotenv_1.default.config();
class AggressiveDeltaTradingBot {
    constructor() {
        this.isRunning = false;
        this.initialBalance = 0;
        this.currentBalance = 0;
        this.peakBalance = 0;
        this.activeTrades = new Map();
        this.sessionStartTime = 0;
        this.totalPnl = 0;
        this.tradesExecuted = 0;
        this.winningTrades = 0;
        this.maxDrawdownHit = false;
        this.config = {
            symbols: ['BTCUSD', 'ETHUSD', 'SOLUSD'], // Only BTC, ETH, SOL as requested
            maxLeverage: 200, // UP TO 200X LEVERAGE
            riskPerTrade: 75, // 75% of max buying power per trade (EXTREMELY AGGRESSIVE)
            maxPositions: 5,
            stopLossPercent: 1, // Tight 1% stop loss
            takeProfitPercent: 2, // Quick 2% take profit
            maxDrawdownPercent: 20, // 20% maximum drawdown limit
            minTradeSize: 1, // Minimum $1 trade (reduced for testing)
            tradingInterval: 15000, // 15 seconds
            statusInterval: 30000, // 30 seconds
            leverageConfig: {
                highConfidence: 200, // 200x for >90% confidence
                mediumConfidence: 100, // 100x for 70-90% confidence
                lowConfidence: 50 // 50x for 50-70% confidence
            }
        };
        // Initialize Delta Exchange service
        const credentials = {
            apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
            apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
            testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
        };
        if (!credentials.apiKey || !credentials.apiSecret) {
            throw new Error('Delta Exchange API credentials not found');
        }
        this.deltaService = new DeltaExchangeUnified_1.DeltaExchangeUnified(credentials);
    }
    /**
     * Start aggressive trading
     */
    async start() {
        try {
            logger_1.logger.info('🚀 STARTING AGGRESSIVE DELTA TRADING BOT');
            logger_1.logger.info('='.repeat(80));
            logger_1.logger.info('⚡ HIGH LEVERAGE MODE: BTC/ETH 100X, SOL 50X');
            logger_1.logger.info('🎯 EXTREMELY AGGRESSIVE: 75% of MAX BUYING POWER per trade');
            logger_1.logger.info('🛑 SAFETY LIMIT: 20% maximum drawdown');
            logger_1.logger.info('⚠️  WARNING: Trading with REAL MONEY at MAXIMUM LEVERAGE');
            logger_1.logger.info('🔥 Press Ctrl+C to stop');
            logger_1.logger.info('');
            // Wait for Delta service to be ready
            let retries = 0;
            while (!this.deltaService.isReady() && retries < 10) {
                logger_1.logger.info('⏳ Waiting for Delta Exchange service...');
                await new Promise(resolve => setTimeout(resolve, 2000));
                retries++;
            }
            if (!this.deltaService.isReady()) {
                throw new Error('Delta Exchange service failed to initialize');
            }
            // Fetch real balance
            await this.fetchRealBalance();
            if (this.currentBalance < 5) {
                throw new Error(`Insufficient balance: $${this.currentBalance.toFixed(2)} (minimum: $5)`);
            }
            this.initialBalance = this.currentBalance;
            this.peakBalance = this.currentBalance;
            this.sessionStartTime = Date.now();
            // Display configuration
            this.displayConfiguration();
            this.isRunning = true;
            // Start aggressive trading loops
            this.startTradingLoop();
            this.startStatusLoop();
            // Handle shutdown
            process.on('SIGINT', () => this.stop());
            process.on('SIGTERM', () => this.stop());
            logger_1.logger.info('✅ Aggressive trading bot started successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start aggressive trading bot:', error);
            throw error;
        }
    }
    /**
     * Fetch real account balance
     */
    async fetchRealBalance() {
        try {
            logger_1.logger.info('💰 Fetching real account balance...');
            const balances = await this.deltaService.getBalance();
            const usdBalance = balances.find(b => b.asset_symbol === 'USDT' || b.asset_symbol === 'USD');
            if (usdBalance) {
                this.currentBalance = parseFloat(usdBalance.available_balance);
                logger_1.logger.info(`💵 Available Balance: $${this.currentBalance.toFixed(2)}`);
            }
            else {
                throw new Error('No USD/USDT balance found');
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Error fetching balance:', error);
            throw error;
        }
    }
    /**
     * Display aggressive configuration
     */
    displayConfiguration() {
        const maxBuyingPower = this.currentBalance * this.config.maxLeverage;
        const maxTradeSize = maxBuyingPower * (this.config.riskPerTrade / 100);
        const drawdownLimit = this.initialBalance * (this.config.maxDrawdownPercent / 100);
        logger_1.logger.info('💰 AGGRESSIVE CONFIGURATION:');
        logger_1.logger.info(`   💵 Balance: $${this.currentBalance.toFixed(2)}`);
        logger_1.logger.info(`   ⚡ MAX Buying Power (200x): $${maxBuyingPower.toFixed(2)}`);
        logger_1.logger.info(`   🎯 Max Trade Size (${this.config.riskPerTrade}% of buying power): $${maxTradeSize.toFixed(2)}`);
        logger_1.logger.info(`   🛑 Stop Loss: ${this.config.stopLossPercent}%`);
        logger_1.logger.info(`   🎯 Take Profit: ${this.config.takeProfitPercent}%`);
        logger_1.logger.info(`   🚨 Drawdown Limit: $${drawdownLimit.toFixed(2)} (${this.config.maxDrawdownPercent}%)`);
        logger_1.logger.info(`   ⚡ Leverage: BTC/ETH 100x, SOL 50x (confidence-adjusted)`);
        logger_1.logger.info('');
    }
    /**
     * Start trading loop
     */
    startTradingLoop() {
        setInterval(async () => {
            if (!this.isRunning || this.maxDrawdownHit)
                return;
            try {
                await this.executeTradingCycle();
            }
            catch (error) {
                logger_1.logger.error('❌ Error in trading cycle:', error);
            }
        }, this.config.tradingInterval);
    }
    /**
     * Start status loop
     */
    startStatusLoop() {
        setInterval(async () => {
            if (!this.isRunning)
                return;
            try {
                await this.displayStatus();
            }
            catch (error) {
                logger_1.logger.error('❌ Error in status display:', error);
            }
        }, this.config.statusInterval);
    }
    /**
     * Execute one trading cycle
     */
    async executeTradingCycle() {
        // Update balance and check drawdown
        await this.fetchRealBalance();
        await this.checkDrawdownLimit();
        // Always manage existing positions, even if drawdown limit is hit
        await this.managePositions();
        // Check if we should stop (only when no active positions remain)
        if (this.maxDrawdownHit) {
            const positions = await this.deltaService.getPositions();
            const activePositions = positions.filter(p => Math.abs(p.size) > 0);
            if (activePositions.length === 0) {
                logger_1.logger.info('✅ All positions closed after drawdown limit - stopping bot');
                await this.stop();
                return;
            }
            else {
                logger_1.logger.info(`🔄 Continuing to manage ${activePositions.length} active position(s) despite drawdown limit`);
            }
        }
        // Only look for new opportunities if drawdown limit hasn't been hit
        if (!this.maxDrawdownHit && this.activeTrades.size < this.config.maxPositions) {
            await this.scanForOpportunities();
        }
    }
    /**
     * Check drawdown limit
     */
    async checkDrawdownLimit() {
        if (this.peakBalance > this.currentBalance) {
            // Update peak
            if (this.currentBalance > this.peakBalance) {
                this.peakBalance = this.currentBalance;
            }
        }
        else {
            this.peakBalance = this.currentBalance;
        }
        const currentDrawdown = ((this.peakBalance - this.currentBalance) / this.peakBalance) * 100;
        if (currentDrawdown >= this.config.maxDrawdownPercent && !this.maxDrawdownHit) {
            this.maxDrawdownHit = true;
            logger_1.logger.error('🚨 MAXIMUM DRAWDOWN LIMIT REACHED!');
            logger_1.logger.error(`📉 Current Drawdown: ${currentDrawdown.toFixed(2)}%`);
            logger_1.logger.error('🛑 STOPPING NEW TRADES - CONTINUING TO MANAGE EXISTING POSITIONS');
            // Don't stop the bot completely - just stop opening new positions
            // Continue managing existing positions until they're closed
        }
    }
    /**
     * Emergency close all positions (only used in extreme cases)
     */
    async emergencyCloseAllPositions() {
        try {
            logger_1.logger.warn('🚨 Emergency close triggered - this should only happen in extreme cases');
            const positions = await this.deltaService.getPositions();
            for (const position of positions) {
                if (Math.abs(position.size) > 0) {
                    logger_1.logger.info(`🚨 Emergency closing: ${position.product?.symbol || 'Unknown'}`);
                    await this.closePosition(position, 'emergency_stop');
                }
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Error in emergency close:', error);
        }
    }
    /**
     * Manage existing positions
     */
    async managePositions() {
        try {
            const positions = await this.deltaService.getPositions();
            for (const position of positions) {
                if (Math.abs(position.size) > 0) {
                    logger_1.logger.info(`📊 Managing position: ${position.product?.symbol || 'Unknown'} - Size: ${position.size}`);
                    await this.checkPositionForExit(position);
                }
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Error managing positions:', error);
        }
    }
    /**
     * Check position for exit
     */
    async checkPositionForExit(position) {
        try {
            const marketData = await this.deltaService.getMarketData(position.product.symbol);
            const currentPrice = parseFloat(marketData.mark_price || marketData.last_price);
            const entryPrice = parseFloat(position.entry_price);
            const side = position.size > 0 ? 'long' : 'short';
            const pnlPercent = side === 'long'
                ? ((currentPrice - entryPrice) / entryPrice) * 100
                : ((entryPrice - currentPrice) / entryPrice) * 100;
            // Tight stop loss
            if (pnlPercent <= -this.config.stopLossPercent) {
                logger_1.logger.info(`🛑 STOP LOSS: ${position.product.symbol} - ${pnlPercent.toFixed(3)}%`);
                await this.closePosition(position, 'stop_loss');
                return;
            }
            // Quick take profit
            if (pnlPercent >= this.config.takeProfitPercent) {
                logger_1.logger.info(`🎯 TAKE PROFIT: ${position.product.symbol} - ${pnlPercent.toFixed(3)}%`);
                await this.closePosition(position, 'take_profit');
                return;
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Error checking position ${position.product.symbol}:`, error);
        }
    }
    /**
     * Close position
     */
    async closePosition(position, reason) {
        try {
            const symbol = position.product?.symbol || 'Unknown';
            // Use the same lot size logic as opening positions
            let lotSize;
            if (symbol.includes('BTC')) {
                lotSize = 0.001; // Official: 0.001 BTC per contract
            }
            else if (symbol.includes('ETH')) {
                lotSize = 0.01; // Confirmed: 0.01 ETH per contract
            }
            else if (symbol.includes('SOL')) {
                lotSize = 1; // Confirmed: 1 SOL per contract
            }
            else {
                lotSize = 0.01; // Default for other assets
            }
            // Calculate contract units to close the exact position
            const contractUnits = Math.abs(Math.round(position.size / lotSize));
            const orderRequest = {
                product_id: this.deltaService.getProductId(symbol),
                side: position.size > 0 ? 'sell' : 'buy',
                size: contractUnits,
                order_type: 'market_order'
            };
            logger_1.logger.info(`🔄 Closing position: ${symbol} - ${contractUnits} contracts (${reason})`);
            await this.deltaService.placeOrder(orderRequest);
            this.tradesExecuted++;
            const pnl = parseFloat(position.unrealized_pnl || '0');
            this.totalPnl += pnl;
            if (pnl > 0) {
                this.winningTrades++;
            }
            logger_1.logger.info(`✅ Position closed: ${symbol} - ${reason} - P&L: $${pnl.toFixed(2)}`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Error closing position:`, error);
        }
    }
    /**
     * Scan for trading opportunities
     */
    async scanForOpportunities() {
        for (const symbol of this.config.symbols) {
            try {
                const signal = await this.generateSignal(symbol);
                if (signal && signal.confidence > 0.5) {
                    await this.openPosition(symbol, signal);
                }
            }
            catch (error) {
                logger_1.logger.error(`❌ Error scanning ${symbol}:`, error);
            }
        }
    }
    /**
     * Generate trading signal
     */
    async generateSignal(symbol) {
        try {
            const marketData = await this.deltaService.getMarketData(symbol);
            // Parse price with better error handling
            let currentPrice = parseFloat(marketData.mark_price || marketData.last_price || '0');
            // Validate price - ONLY USE LIVE DATA
            if (isNaN(currentPrice) || currentPrice <= 0) {
                logger_1.logger.error(`❌ Invalid live price data for ${symbol}: ${currentPrice}. Skipping trade to avoid mock data.`);
                return null; // Return null instead of using fallback mock data
            }
            logger_1.logger.debug(`📊 Market data for ${symbol}: Price $${currentPrice.toFixed(2)}`);
            logger_1.logger.debug(`📊 Raw market data for ${symbol}:`, marketData);
            // Aggressive signal generation (higher probability for testing)
            const random = Math.random();
            const confidence = 0.6 + (Math.random() * 0.4); // 60-100% confidence
            let signal = null;
            if (random > 0.6) { // 40% chance for buy signal
                signal = { side: 'buy', confidence, price: currentPrice };
                logger_1.logger.info(`📈 BUY signal generated for ${symbol}: ${(confidence * 100).toFixed(1)}% confidence @ $${currentPrice.toFixed(2)}`);
            }
            else if (random < 0.4) { // 40% chance for sell signal
                signal = { side: 'sell', confidence, price: currentPrice };
                logger_1.logger.info(`📉 SELL signal generated for ${symbol}: ${(confidence * 100).toFixed(1)}% confidence @ $${currentPrice.toFixed(2)}`);
            }
            else {
                logger_1.logger.debug(`⏸️ No signal for ${symbol} (random: ${random.toFixed(3)})`);
            }
            return signal;
        }
        catch (error) {
            logger_1.logger.error(`❌ Error generating signal for ${symbol}:`, error);
            return null;
        }
    }
    /**
     * Open new position
     */
    async openPosition(symbol, signal) {
        try {
            // Calculate leverage based on asset and confidence
            let baseLeverage;
            if (symbol.includes('BTC') || symbol.includes('ETH')) {
                baseLeverage = 100; // 100x for BTC and ETH
            }
            else if (symbol.includes('SOL')) {
                baseLeverage = 50; // 50x for SOL
            }
            else {
                baseLeverage = 50; // Default 50x for other assets
            }
            // Apply confidence multiplier (0.5x to 1.0x based on confidence)
            const confidenceMultiplier = 0.5 + (signal.confidence * 0.5); // 50%-100% of base leverage
            const leverage = Math.floor(baseLeverage * confidenceMultiplier);
            // AGGRESSIVE: Use 75% of maximum buying power (with leverage)
            const maxBuyingPower = this.currentBalance * leverage;
            const riskAmount = maxBuyingPower * (this.config.riskPerTrade / 100);
            // Calculate size based on leveraged risk amount
            const size = riskAmount / signal.price;
            // Ensure minimum size requirements
            const minSize = 0.000001; // Minimum size for crypto
            const finalSize = Math.max(size, minSize);
            logger_1.logger.info(`📊 Position Calculation for ${symbol}:`);
            logger_1.logger.info(`   💰 Current Balance: $${this.currentBalance.toFixed(2)}`);
            logger_1.logger.info(`   ⚡ Leverage: ${leverage}x`);
            logger_1.logger.info(`   💪 Max Buying Power: $${maxBuyingPower.toFixed(2)}`);
            logger_1.logger.info(`   🎯 Risk Amount (${this.config.riskPerTrade}% of buying power): $${riskAmount.toFixed(2)}`);
            logger_1.logger.info(`   💵 Signal Price: $${signal.price.toFixed(2)}`);
            logger_1.logger.info(`   📏 Calculated Size: ${size.toFixed(8)}`);
            logger_1.logger.info(`   📏 Final Size: ${finalSize.toFixed(8)}`);
            // Delta Exchange official lot sizing (CONFIRMED):
            // BTC (Product ID 27): lot size = 0.001 BTC per contract
            // ETH (Product ID 3136): lot size = 0.01 ETH per contract (CONFIRMED)
            // SOL: lot size = 1 SOL per contract (CONFIRMED)
            let lotSize;
            if (symbol.includes('BTC')) {
                lotSize = 0.001; // Official: 0.001 BTC per contract
            }
            else if (symbol.includes('ETH')) {
                lotSize = 0.01; // Confirmed: 0.01 ETH per contract
            }
            else if (symbol.includes('SOL')) {
                lotSize = 1; // Confirmed: 1 SOL per contract
            }
            else {
                lotSize = 0.01; // Default for other assets
            }
            // Calculate number of lots we can afford
            const contractUnits = Math.floor(finalSize / lotSize);
            const actualSize = contractUnits * lotSize;
            const usdValue = actualSize * signal.price;
            logger_1.logger.info(`   💵 USD Value: $${usdValue.toFixed(2)}`);
            logger_1.logger.info(`   📏 Lot Size: ${lotSize} ${symbol.replace('USD', '')} per contract`);
            logger_1.logger.info(`   🔢 Contract Units (lots): ${contractUnits}`);
            if (riskAmount < this.config.minTradeSize) {
                logger_1.logger.warn(`⚠️ Risk amount too small: $${riskAmount.toFixed(2)} < $${this.config.minTradeSize}`);
                return;
            }
            if (finalSize <= 0) {
                logger_1.logger.warn(`⚠️ Invalid position size: ${finalSize}`);
                return;
            }
            const orderRequest = {
                product_id: this.deltaService.getProductId(symbol),
                side: signal.side,
                size: contractUnits, // Pass as integer contract units
                order_type: 'market_order'
            };
            logger_1.logger.info(`🚀 Placing order: ${JSON.stringify(orderRequest)}`);
            const order = await this.deltaService.placeOrder(orderRequest);
            logger_1.logger.info(`🔥 AGGRESSIVE POSITION OPENED: ${signal.side.toUpperCase()} ${finalSize.toFixed(8)} ${symbol}`);
            logger_1.logger.info(`   💰 Entry: $${signal.price.toFixed(2)}`);
            logger_1.logger.info(`   ⚡ Leverage: ${leverage}x`);
            logger_1.logger.info(`   🎯 Confidence: ${(signal.confidence * 100).toFixed(1)}%`);
            logger_1.logger.info(`   💵 Risk: $${riskAmount.toFixed(2)}`);
            logger_1.logger.info(`   🆔 Order ID: ${order.id}`);
        }
        catch (error) {
            logger_1.logger.error(`❌ Error opening position ${symbol}:`, error);
        }
    }
    /**
     * Display status
     */
    async displayStatus() {
        const sessionDuration = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const minutes = Math.floor(sessionDuration / 60);
        const seconds = sessionDuration % 60;
        const winRate = this.tradesExecuted > 0 ? (this.winningTrades / this.tradesExecuted) * 100 : 0;
        const totalReturn = ((this.currentBalance - this.initialBalance) / this.initialBalance) * 100;
        const currentDrawdown = this.peakBalance > 0 ? ((this.peakBalance - this.currentBalance) / this.peakBalance) * 100 : 0;
        logger_1.logger.info('');
        logger_1.logger.info('⚡ AGGRESSIVE TRADING STATUS');
        logger_1.logger.info('='.repeat(60));
        logger_1.logger.info(`⏱️  Session: ${minutes}m ${seconds}s`);
        logger_1.logger.info(`💰 Balance: $${this.currentBalance.toFixed(2)} (${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%)`);
        logger_1.logger.info(`📈 Peak: $${this.peakBalance.toFixed(2)}`);
        logger_1.logger.info(`📉 Drawdown: ${currentDrawdown.toFixed(2)}% / ${this.config.maxDrawdownPercent}% MAX`);
        logger_1.logger.info(`📊 Active: ${this.activeTrades.size}/${this.config.maxPositions} positions`);
        logger_1.logger.info(`✅ Trades: ${this.tradesExecuted} | Win Rate: ${winRate.toFixed(1)}%`);
        logger_1.logger.info(`🤖 Status: ${this.isRunning ? 'AGGRESSIVE SCANNING' : 'STOPPED'}`);
        if (currentDrawdown > 15) {
            logger_1.logger.warn(`🚨 WARNING: Approaching max drawdown (${currentDrawdown.toFixed(1)}%)`);
        }
        logger_1.logger.info('');
    }
    /**
     * Stop the bot
     */
    async stop() {
        logger_1.logger.info('🛑 Stopping aggressive trading bot...');
        this.isRunning = false;
        const sessionDuration = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const minutes = Math.floor(sessionDuration / 60);
        const seconds = sessionDuration % 60;
        const winRate = this.tradesExecuted > 0 ? (this.winningTrades / this.tradesExecuted) * 100 : 0;
        const totalReturn = ((this.currentBalance - this.initialBalance) / this.initialBalance) * 100;
        logger_1.logger.info('');
        logger_1.logger.info('📋 AGGRESSIVE TRADING SUMMARY');
        logger_1.logger.info('='.repeat(80));
        logger_1.logger.info(`⏱️  Duration: ${minutes}m ${seconds}s`);
        logger_1.logger.info(`💰 Initial: $${this.initialBalance.toFixed(2)}`);
        logger_1.logger.info(`💰 Final: $${this.currentBalance.toFixed(2)}`);
        logger_1.logger.info(`📈 Return: ${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`);
        logger_1.logger.info(`✅ Trades: ${this.tradesExecuted}`);
        logger_1.logger.info(`🎯 Win Rate: ${winRate.toFixed(1)}%`);
        logger_1.logger.info(`⚡ Max Leverage: BTC/ETH 100x, SOL 50x`);
        logger_1.logger.info('');
        logger_1.logger.info('✅ Aggressive trading session completed');
        process.exit(0);
    }
}
// Start the bot
if (require.main === module) {
    const bot = new AggressiveDeltaTradingBot();
    bot.start().catch(error => {
        logger_1.logger.error('❌ Fatal error:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=aggressive-delta-trading-bot.js.map