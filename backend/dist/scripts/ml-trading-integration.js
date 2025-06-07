"use strict";
/**
 * ML Trading Integration Script
 *
 * This script integrates the ML Trading Decision Engine with our existing
 * analysis and execution infrastructure to create a complete automated
 * trading system that uses ML models as primary decision makers.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MLTradingIntegration = void 0;
const logger_1 = require("../utils/logger");
const MLTradingDecisionEngine_1 = require("../services/MLTradingDecisionEngine");
const MultiTimeframeAnalysisEngine_1 = require("../services/MultiTimeframeAnalysisEngine");
const EnhancedMLIntegrationService_1 = require("../services/EnhancedMLIntegrationService");
const delta_trading_bot_1 = require("./delta-trading-bot");
class MLTradingIntegration {
    constructor(config) {
        this.isRunning = false;
        this.activePositions = new Map();
        this.dailyStats = {
            tradesExecuted: 0,
            profitLoss: 0,
            winRate: 0,
            startBalance: 0
        };
        this.config = config;
    }
    /**
     * Initialize all components and start ML-driven trading
     */
    async initialize() {
        try {
            logger_1.logger.info('🚀 Initializing ML Trading Integration...');
            // Initialize components
            await this.initializeComponents();
            // Validate configuration
            this.validateConfiguration();
            // Initialize ML Trading Decision Engine
            await this.mlEngine.initialize();
            // Record starting balance
            this.dailyStats.startBalance = await this.tradingBot.getAccountBalance();
            logger_1.logger.info('✅ ML Trading Integration initialized successfully');
            logger_1.logger.info(`📊 Starting balance: $${this.dailyStats.startBalance.toFixed(2)}`);
            logger_1.logger.info(`🎯 Trading symbols: ${this.config.symbols.join(', ')}`);
            logger_1.logger.info(`⚡ Refresh interval: ${this.config.refreshInterval}s`);
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize ML Trading Integration:', error);
            throw error;
        }
    }
    /**
     * Start automated ML-driven trading
     */
    async startTrading() {
        if (this.isRunning) {
            logger_1.logger.warn('⚠️ ML Trading already running');
            return;
        }
        try {
            logger_1.logger.info('🤖 Starting ML-driven automated trading...');
            this.isRunning = true;
            // Start main trading loop
            this.runTradingLoop();
            // Start position monitoring
            this.monitorPositions();
            // Start performance tracking
            this.trackPerformance();
            logger_1.logger.info('✅ ML Trading started successfully');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start ML Trading:', error);
            this.isRunning = false;
            throw error;
        }
    }
    /**
     * Stop automated trading
     */
    async stopTrading() {
        logger_1.logger.info('🛑 Stopping ML-driven trading...');
        this.isRunning = false;
        // Close all positions if needed
        if (this.config.enablePaperTrading === false) {
            await this.closeAllPositions();
        }
        // Log final statistics
        this.logFinalStats();
        logger_1.logger.info('✅ ML Trading stopped');
    }
    /**
     * Main trading loop - analyzes markets and executes ML-driven trades
     */
    async runTradingLoop() {
        while (this.isRunning) {
            try {
                // Check risk management constraints
                if (!(await this.checkRiskConstraints())) {
                    logger_1.logger.warn('⚠️ Risk constraints violated - skipping trading cycle');
                    await this.sleep(this.config.refreshInterval * 1000);
                    continue;
                }
                // Process each symbol
                for (const symbol of this.config.symbols) {
                    if (!this.isRunning)
                        break;
                    try {
                        await this.processSymbol(symbol);
                    }
                    catch (error) {
                        logger_1.logger.error(`❌ Error processing ${symbol}:`, error);
                    }
                }
                // Wait for next cycle
                await this.sleep(this.config.refreshInterval * 1000);
            }
            catch (error) {
                logger_1.logger.error('❌ Error in trading loop:', error);
                await this.sleep(5000); // Wait 5 seconds before retrying
            }
        }
    }
    /**
     * Process a single symbol for ML trading decisions
     */
    async processSymbol(symbol) {
        try {
            // Skip if already have position and max concurrent reached
            if (this.activePositions.size >= this.config.maxConcurrentTrades) {
                return;
            }
            // Get current price
            const currentPrice = await this.tradingBot.getCurrentPrice(symbol);
            if (!currentPrice) {
                logger_1.logger.warn(`⚠️ Could not get current price for ${symbol}`);
                return;
            }
            // Generate ML trading decision
            const decision = await this.mlEngine.generateTradingDecision(symbol, currentPrice);
            // Log decision
            logger_1.logger.info(`🧠 ML Decision for ${symbol}: ${decision.action} (${(decision.confidence * 100).toFixed(1)}% confidence)`);
            // Check if decision meets our thresholds
            if (decision.confidence < this.config.minConfidenceThreshold) {
                logger_1.logger.debug(`⚠️ ML confidence too low for ${symbol}: ${(decision.confidence * 100).toFixed(1)}%`);
                return;
            }
            // Skip if we already have a position in this symbol
            if (this.activePositions.has(symbol)) {
                logger_1.logger.debug(`⚠️ Already have position in ${symbol}`);
                return;
            }
            // Execute trade if action is BUY or SELL
            if (decision.action !== 'HOLD') {
                const success = await this.mlEngine.executeTrade(symbol, decision, currentPrice);
                if (success) {
                    // Track the new position
                    this.activePositions.set(symbol, {
                        symbol,
                        action: decision.action,
                        entryPrice: currentPrice,
                        entryTime: Date.now(),
                        positionSize: decision.positionSize,
                        stopLoss: decision.stopLoss,
                        takeProfit: decision.takeProfit,
                        mlConfidence: decision.confidence,
                        reasoning: decision.reasoning
                    });
                    this.dailyStats.tradesExecuted++;
                    logger_1.logger.info(`✅ Executed ML-driven ${decision.action} for ${symbol} at $${currentPrice.toFixed(2)}`);
                }
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Error processing symbol ${symbol}:`, error);
        }
    }
    /**
     * Monitor active positions and manage exits
     */
    async monitorPositions() {
        while (this.isRunning) {
            try {
                for (const [symbol, position] of this.activePositions.entries()) {
                    await this.checkPositionExit(symbol, position);
                }
                await this.sleep(10000); // Check positions every 10 seconds
            }
            catch (error) {
                logger_1.logger.error('❌ Error monitoring positions:', error);
                await this.sleep(5000);
            }
        }
    }
    /**
     * Check if a position should be exited
     */
    async checkPositionExit(symbol, position) {
        try {
            const currentPrice = await this.tradingBot.getCurrentPrice(symbol);
            if (!currentPrice)
                return;
            // Calculate P&L
            const pnlPercent = position.action === 'BUY'
                ? (currentPrice - position.entryPrice) / position.entryPrice
                : (position.entryPrice - currentPrice) / position.entryPrice;
            // Check stop loss
            if (pnlPercent <= -position.stopLoss) {
                await this.exitPosition(symbol, position, 'STOP_LOSS', currentPrice);
                return;
            }
            // Check take profit
            if (pnlPercent >= position.takeProfit) {
                await this.exitPosition(symbol, position, 'TAKE_PROFIT', currentPrice);
                return;
            }
            // Check ML-driven exit signal
            const exitDecision = await this.mlEngine.generateTradingDecision(symbol, currentPrice);
            if (this.shouldExitBasedOnML(position, exitDecision)) {
                await this.exitPosition(symbol, position, 'ML_SIGNAL', currentPrice);
                return;
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Error checking position exit for ${symbol}:`, error);
        }
    }
    /**
     * Exit a position
     */
    async exitPosition(symbol, position, reason, exitPrice) {
        try {
            // Calculate final P&L
            const pnlPercent = position.action === 'BUY'
                ? (exitPrice - position.entryPrice) / position.entryPrice
                : (position.entryPrice - exitPrice) / position.entryPrice;
            const pnlAmount = pnlPercent * position.positionSize * this.dailyStats.startBalance;
            // Execute exit trade
            const exitAction = position.action === 'BUY' ? 'SELL' : 'BUY';
            const success = await this.tradingBot.executeTrade({
                symbol,
                side: exitAction.toLowerCase(),
                size: position.positionSize,
                price: exitPrice,
                reason: `EXIT_${reason}`
            });
            if (success) {
                // Update statistics
                this.dailyStats.profitLoss += pnlAmount;
                if (pnlAmount > 0) {
                    this.dailyStats.winRate = (this.dailyStats.winRate * (this.dailyStats.tradesExecuted - 1) + 1) / this.dailyStats.tradesExecuted;
                }
                // Remove from active positions
                this.activePositions.delete(symbol);
                logger_1.logger.info(`🔄 Exited ${symbol} position: ${reason} | P&L: ${pnlPercent > 0 ? '+' : ''}${(pnlPercent * 100).toFixed(2)}% ($${pnlAmount.toFixed(2)})`);
            }
        }
        catch (error) {
            logger_1.logger.error(`❌ Error exiting position for ${symbol}:`, error);
        }
    }
    // Additional utility methods...
    async initializeComponents() {
        this.mtfAnalyzer = new MultiTimeframeAnalysisEngine_1.MultiTimeframeAnalysisEngine();
        this.mlService = new EnhancedMLIntegrationService_1.EnhancedMLIntegrationService();
        this.tradingBot = new delta_trading_bot_1.DeltaTradingBot();
        this.mlEngine = new MLTradingDecisionEngine_1.MLTradingDecisionEngine(this.mtfAnalyzer, this.mlService, this.tradingBot);
    }
    validateConfiguration() {
        if (!this.config.symbols || this.config.symbols.length === 0) {
            throw new Error('No trading symbols configured');
        }
        if (this.config.refreshInterval < 5) {
            throw new Error('Refresh interval too low (minimum 5 seconds)');
        }
    }
    async checkRiskConstraints() {
        const currentBalance = await this.tradingBot.getAccountBalance();
        const dailyLoss = (this.dailyStats.startBalance - currentBalance) / this.dailyStats.startBalance;
        // Only check daily loss limit - no minimum balance restriction for small capital + high leverage strategy
        return dailyLoss < this.config.riskManagement.maxDailyLoss;
    }
    shouldExitBasedOnML(position, exitDecision) {
        // Exit if ML suggests opposite action with high confidence
        return (position.action === 'BUY' && exitDecision.action === 'SELL' && exitDecision.confidence > 0.75) ||
            (position.action === 'SELL' && exitDecision.action === 'BUY' && exitDecision.confidence > 0.75);
    }
    async closeAllPositions() {
        for (const [symbol, position] of this.activePositions.entries()) {
            const currentPrice = await this.tradingBot.getCurrentPrice(symbol);
            if (currentPrice) {
                await this.exitPosition(symbol, position, 'SYSTEM_SHUTDOWN', currentPrice);
            }
        }
    }
    logFinalStats() {
        const totalReturn = (this.dailyStats.profitLoss / this.dailyStats.startBalance) * 100;
        logger_1.logger.info('📊 Final ML Trading Statistics:');
        logger_1.logger.info(`   Trades Executed: ${this.dailyStats.tradesExecuted}`);
        logger_1.logger.info(`   Total P&L: $${this.dailyStats.profitLoss.toFixed(2)} (${totalReturn.toFixed(2)}%)`);
        logger_1.logger.info(`   Win Rate: ${(this.dailyStats.winRate * 100).toFixed(1)}%`);
        logger_1.logger.info(`   Active Positions: ${this.activePositions.size}`);
    }
    trackPerformance() {
        setInterval(() => {
            if (this.isRunning) {
                logger_1.logger.info(`📈 ML Trading Performance: ${this.dailyStats.tradesExecuted} trades, $${this.dailyStats.profitLoss.toFixed(2)} P&L`);
            }
        }, 300000); // Log every 5 minutes
    }
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
exports.MLTradingIntegration = MLTradingIntegration;
//# sourceMappingURL=ml-trading-integration.js.map