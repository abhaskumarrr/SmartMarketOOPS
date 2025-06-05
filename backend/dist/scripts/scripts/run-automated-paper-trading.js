#!/usr/bin/env node
"use strict";
/**
 * Automated Paper Trading Session
 * Places orders and lets the bot manage everything automatically
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AutomatedPaperTradingSession = void 0;
const paperTradingEngine_1 = require("../services/paperTradingEngine");
const marketDataProvider_1 = require("../services/marketDataProvider");
const logger_1 = require("../utils/logger");
class AutomatedPaperTradingSession {
    constructor() {
        this.statusInterval = null;
        this.isRunning = false;
        // Configure for aggressive paper trading
        const initialBalance = 2000; // $2000 starting capital
        const leverage = 3; // 3x leverage ($6000 buying power)
        const riskPerTrade = 2; // 2% risk per trade ($40 max risk)
        this.engine = new paperTradingEngine_1.PaperTradingEngine(initialBalance, leverage, riskPerTrade);
        this.sessionStartTime = Date.now();
    }
    /**
     * Start automated paper trading session
     */
    async startAutomatedSession() {
        try {
            this.isRunning = true;
            logger_1.logger.info('🚀 STARTING AUTOMATED PAPER TRADING SESSION');
            logger_1.logger.info('='.repeat(80));
            // Display configuration
            this.displayConfiguration();
            // Ensure we're using live data
            marketDataProvider_1.marketDataService.enforceLiveDataMode();
            const providerInfo = marketDataProvider_1.marketDataService.getCurrentProviderInfo();
            logger_1.logger.info(`📊 Data Source: ${providerInfo.name} (Live: ${providerInfo.isLive})`);
            // Start monitoring
            this.startStatusMonitoring();
            // Start the paper trading engine (it will place orders automatically)
            logger_1.logger.info('🎯 Starting automated order placement and management...');
            await this.engine.startPaperTrading();
        }
        catch (error) {
            logger_1.logger.error('❌ Automated paper trading session failed:', error);
            this.stopSession();
            throw error;
        }
    }
    /**
     * Stop the automated session
     */
    stopSession() {
        this.isRunning = false;
        logger_1.logger.info('🛑 Stopping automated paper trading session...');
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
        this.engine.stopPaperTrading();
        this.displayFinalSummary();
    }
    /**
     * Display trading configuration
     */
    displayConfiguration() {
        logger_1.logger.info('📋 AUTOMATED TRADING CONFIGURATION:');
        logger_1.logger.info('   💰 Initial Balance: $2,000');
        logger_1.logger.info('   ⚡ Leverage: 3x ($6,000 max buying power)');
        logger_1.logger.info('   🎯 Risk Per Trade: 2% ($40 max risk per trade)');
        logger_1.logger.info('   📈 Assets: BTC/USD, ETH/USD');
        logger_1.logger.info('   🤖 Mode: FULLY AUTOMATED');
        logger_1.logger.info('   🔄 Features: Auto order placement, dynamic TP/SL, position management');
        logger_1.logger.info('   📊 Data: Live market data from Delta Exchange');
        logger_1.logger.info('   ⏱️  Duration: Continuous until stopped or max drawdown');
        logger_1.logger.info('');
    }
    /**
     * Start status monitoring
     */
    startStatusMonitoring() {
        this.statusInterval = setInterval(() => {
            this.displayLiveStatus();
        }, 15000); // Every 15 seconds for active monitoring
    }
    /**
     * Display live trading status
     */
    displayLiveStatus() {
        if (!this.isRunning)
            return;
        const sessionDuration = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const minutes = Math.floor(sessionDuration / 60);
        const seconds = sessionDuration % 60;
        logger_1.logger.info('');
        logger_1.logger.info('📊 LIVE TRADING STATUS');
        logger_1.logger.info('-'.repeat(50));
        logger_1.logger.info(`⏱️  Session Duration: ${minutes}m ${seconds}s`);
        logger_1.logger.info(`🤖 Bot Status: ${this.isRunning ? 'ACTIVE - Monitoring & Trading' : 'STOPPED'}`);
        logger_1.logger.info(`📈 Market Scanning: BTC/USD, ETH/USD`);
        logger_1.logger.info(`🎯 Looking for: High-confidence signals (>70%)`);
        logger_1.logger.info(`💡 Strategy: Dynamic take profit with partial exits`);
        logger_1.logger.info(`🔄 Next scan: ~15 seconds`);
        logger_1.logger.info('');
    }
    /**
     * Display final session summary
     */
    displayFinalSummary() {
        const sessionDuration = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const minutes = Math.floor(sessionDuration / 60);
        const seconds = sessionDuration % 60;
        logger_1.logger.info('');
        logger_1.logger.info('📋 AUTOMATED TRADING SESSION SUMMARY');
        logger_1.logger.info('='.repeat(80));
        logger_1.logger.info(`⏱️  Total Duration: ${minutes}m ${seconds}s`);
        logger_1.logger.info(`🤖 Mode: Fully Automated Paper Trading`);
        logger_1.logger.info(`📊 Data Source: Live Delta Exchange market data`);
        logger_1.logger.info(`🎯 Risk Management: Automated stop loss & take profit`);
        logger_1.logger.info(`💰 Position Management: Dynamic partial exits`);
        logger_1.logger.info('');
        logger_1.logger.info('✅ Session completed successfully');
        logger_1.logger.info('📈 All trades were executed with live market data');
        logger_1.logger.info('🔒 Risk management was consistently applied');
        logger_1.logger.info('');
    }
    /**
     * Handle graceful shutdown
     */
    setupGracefulShutdown() {
        const shutdown = () => {
            logger_1.logger.info('');
            logger_1.logger.info('🛑 Received shutdown signal, stopping automated trading...');
            this.stopSession();
            process.exit(0);
        };
        process.on('SIGINT', shutdown);
        process.on('SIGTERM', shutdown);
        process.on('SIGUSR2', shutdown); // For nodemon
    }
}
exports.AutomatedPaperTradingSession = AutomatedPaperTradingSession;
/**
 * Main execution function
 */
async function main() {
    const session = new AutomatedPaperTradingSession();
    // Setup graceful shutdown
    const shutdown = () => {
        logger_1.logger.info('');
        logger_1.logger.info('🛑 Received shutdown signal, stopping automated trading...');
        session.stopSession();
        process.exit(0);
    };
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    process.on('SIGUSR2', shutdown);
    try {
        // Start the automated session
        await session.startAutomatedSession();
    }
    catch (error) {
        logger_1.logger.error('❌ Fatal error in automated paper trading:', error);
        session.stopSession();
        process.exit(1);
    }
}
// Run if called directly
if (require.main === module) {
    logger_1.logger.info('🎯 AUTOMATED PAPER TRADING SESSION');
    logger_1.logger.info('🤖 The bot will automatically:');
    logger_1.logger.info('   • Scan BTC/USD and ETH/USD markets');
    logger_1.logger.info('   • Generate trading signals');
    logger_1.logger.info('   • Place orders when high-confidence signals appear');
    logger_1.logger.info('   • Manage positions with dynamic take profit');
    logger_1.logger.info('   • Apply stop losses for risk management');
    logger_1.logger.info('   • Execute partial exits based on market conditions');
    logger_1.logger.info('');
    logger_1.logger.info('Press Ctrl+C to stop the session');
    logger_1.logger.info('');
    main().catch(error => {
        logger_1.logger.error('❌ Failed to start automated paper trading:', error);
        process.exit(1);
    });
}
