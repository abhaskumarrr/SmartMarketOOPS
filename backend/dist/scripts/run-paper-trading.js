#!/usr/bin/env node
"use strict";
/**
 * Paper Trading System Runner
 * Runs live paper trading with dynamic take profit system
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PaperTradingRunner = void 0;
const paperTradingEngine_1 = require("../services/paperTradingEngine");
const logger_1 = require("../utils/logger");
class PaperTradingRunner {
    constructor() {
        this.statusInterval = null;
        // Initialize with Delta Exchange spot trading configuration
        const initialBalance = 2000; // $2000 starting capital for spot trading
        const leverage = 3; // 3x leverage for spot trading (more conservative)
        const riskPerTrade = 2; // 2% risk per trade
        this.engine = new paperTradingEngine_1.PaperTradingEngine(initialBalance, leverage, riskPerTrade);
    }
    /**
     * Start paper trading system
     */
    async startPaperTrading() {
        try {
            logger_1.logger.info('🚀 STARTING PAPER TRADING SYSTEM');
            logger_1.logger.info('='.repeat(80));
            logger_1.logger.info('📊 PAPER TRADING CONFIGURATION:');
            logger_1.logger.info('   💰 Initial Balance: $2,000 (Delta Exchange spot trading)');
            logger_1.logger.info('   ⚡ Leverage: 3x ($6,000 max buying power)');
            logger_1.logger.info('   🎯 Risk Per Trade: 2% ($40 max risk per trade)');
            logger_1.logger.info('   📈 Assets: BTC/USDT, ETH/USDT (Delta Exchange spot pairs)');
            logger_1.logger.info('   🔄 Strategy: Enhanced Dynamic Take Profit System');
            logger_1.logger.info('   📊 Features: Real market data, market regime detection, partial exits, trailing stops');
            logger_1.logger.info('   🏢 Exchange: Delta Exchange (Testnet)');
            // Start status monitoring
            this.startStatusMonitoring();
            // Start the paper trading engine
            await this.engine.startPaperTrading();
        }
        catch (error) {
            logger_1.logger.error('❌ Paper trading failed:', error);
            this.stopStatusMonitoring();
            throw error;
        }
    }
    /**
     * Stop paper trading system
     */
    stopPaperTrading() {
        logger_1.logger.info('🛑 Stopping paper trading system...');
        this.engine.stopPaperTrading();
        this.stopStatusMonitoring();
    }
    /**
     * Start status monitoring
     */
    startStatusMonitoring() {
        this.statusInterval = setInterval(() => {
            this.displayStatus();
        }, 30000); // Every 30 seconds
    }
    /**
     * Stop status monitoring
     */
    stopStatusMonitoring() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }
    /**
     * Display current status
     */
    displayStatus() {
        const portfolio = this.engine.getPortfolioStatus();
        const activeTrades = this.engine.getActiveTrades();
        const closedTrades = this.engine.getClosedTrades();
        logger_1.logger.info('\n📊 PAPER TRADING STATUS UPDATE:');
        logger_1.logger.info(`   💰 Current Balance: $${portfolio.currentBalance.toFixed(2)}`);
        logger_1.logger.info(`   📈 Total P&L: $${portfolio.totalPnl.toFixed(2)} (${((portfolio.currentBalance - portfolio.initialBalance) / portfolio.initialBalance * 100).toFixed(2)}%)`);
        logger_1.logger.info(`   📊 Active Trades: ${activeTrades.length}`);
        logger_1.logger.info(`   ✅ Closed Trades: ${closedTrades.length}`);
        logger_1.logger.info(`   🎯 Win Rate: ${portfolio.winRate.toFixed(1)}%`);
        logger_1.logger.info(`   ⚠️ Current Drawdown: ${portfolio.currentDrawdown.toFixed(2)}%`);
        // Show active trades
        if (activeTrades.length > 0) {
            logger_1.logger.info('   🔥 Active Trades:');
            activeTrades.forEach(trade => {
                const unrealizedPercent = ((trade.unrealizedPnl / portfolio.initialBalance) * 100);
                logger_1.logger.info(`     ${trade.symbol}: ${trade.side} $${trade.unrealizedPnl.toFixed(2)} (${unrealizedPercent.toFixed(2)}%) - ${trade.partialExits.length} exits`);
            });
        }
        // Show recent closed trades
        if (closedTrades.length > 0) {
            const recentTrades = closedTrades.slice(-3); // Last 3 trades
            logger_1.logger.info('   📋 Recent Closed Trades:');
            recentTrades.forEach(trade => {
                const pnlPercent = ((trade.pnl / portfolio.initialBalance) * 100);
                const status = trade.pnl > 0 ? '✅' : '❌';
                logger_1.logger.info(`     ${status} ${trade.symbol}: $${trade.pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%) - ${trade.reason}`);
            });
        }
    }
    /**
     * Handle graceful shutdown
     */
    setupGracefulShutdown() {
        const shutdown = () => {
            logger_1.logger.info('\n🛑 Received shutdown signal, stopping paper trading...');
            this.stopPaperTrading();
            process.exit(0);
        };
        process.on('SIGINT', shutdown);
        process.on('SIGTERM', shutdown);
        process.on('SIGUSR2', shutdown); // For nodemon
    }
}
exports.PaperTradingRunner = PaperTradingRunner;
/**
 * Main execution function
 */
async function main() {
    const runner = new PaperTradingRunner();
    try {
        // Setup graceful shutdown
        runner.setupGracefulShutdown();
        // Start paper trading
        await runner.startPaperTrading();
    }
    catch (error) {
        logger_1.logger.error('💥 Paper trading runner failed:', error);
        process.exit(1);
    }
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}
//# sourceMappingURL=run-paper-trading.js.map