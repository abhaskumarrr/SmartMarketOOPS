#!/usr/bin/env node
"use strict";
/**
 * Start ML Trading System
 *
 * This script starts the complete ML-driven trading system that integrates
 * all our trading analysis (Fibonacci, SMC, confluence, candle formation)
 * as features for ML models to make actual trading decisions.
 *
 * Usage:
 *   npm run ml-trading              # Start with default config
 *   npm run ml-trading -- --paper   # Start in paper trading mode
 *   npm run ml-trading -- --live    # Start with real money (DANGEROUS!)
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ML_TRADING_CONFIG = exports.MLTradingSystemLauncher = void 0;
const logger_1 = require("../utils/logger");
const ml_trading_integration_1 = require("./ml-trading-integration");
// Configuration for ML Trading System
const ML_TRADING_CONFIG = {
    // Trading symbols (Delta Exchange product IDs)
    symbols: ['BTCUSD', 'ETHUSD'], // BTC and ETH perpetual futures
    // Analysis and execution settings
    refreshInterval: 30, // Analyze markets every 30 seconds
    maxConcurrentTrades: 2, // Maximum 2 positions at once
    minConfidenceThreshold: 0.65, // Minimum 65% ML confidence for trades
    // Trading mode
    enablePaperTrading: true, // Start in paper trading mode by default
    // Risk management settings
    riskManagement: {
        maxDailyLoss: 0.10, // Stop trading if daily loss exceeds 10% (higher for small capital + high leverage)
        maxPositionSize: 0.05, // Maximum 5% of balance per position (higher for pinpoint entries)
        // Removed stopTradingBalance - we use small amounts with high leverage for maximum profit
    }
};
exports.ML_TRADING_CONFIG = ML_TRADING_CONFIG;
class MLTradingSystemLauncher {
    constructor() {
        this.isShuttingDown = false;
        this.mlTrading = new ml_trading_integration_1.MLTradingIntegration(ML_TRADING_CONFIG);
        this.setupSignalHandlers();
    }
    /**
     * Start the ML Trading System
     */
    async start() {
        try {
            // Parse command line arguments
            this.parseArguments();
            // Display startup banner
            this.displayStartupBanner();
            // Validate environment and configuration
            await this.validateEnvironment();
            // Initialize and start ML trading
            await this.mlTrading.initialize();
            await this.mlTrading.startTrading();
            logger_1.logger.info('🚀 ML Trading System is now running...');
            logger_1.logger.info('📊 Monitor the logs for trading decisions and performance');
            logger_1.logger.info('🛑 Press Ctrl+C to stop the system gracefully');
            // Keep the process running
            await this.keepAlive();
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start ML Trading System:', error);
            process.exit(1);
        }
    }
    /**
     * Parse command line arguments
     */
    parseArguments() {
        const args = process.argv.slice(2);
        if (args.includes('--paper')) {
            ML_TRADING_CONFIG.enablePaperTrading = true;
            logger_1.logger.info('📝 Paper trading mode enabled');
        }
        if (args.includes('--live')) {
            ML_TRADING_CONFIG.enablePaperTrading = false;
            logger_1.logger.warn('💰 LIVE TRADING MODE ENABLED - REAL MONEY AT RISK!');
        }
        if (args.includes('--fast')) {
            ML_TRADING_CONFIG.refreshInterval = 15; // Faster analysis for testing
            logger_1.logger.info('⚡ Fast mode enabled (15s refresh interval)');
        }
        if (args.includes('--conservative')) {
            ML_TRADING_CONFIG.minConfidenceThreshold = 0.80; // Higher confidence required
            ML_TRADING_CONFIG.riskManagement.maxPositionSize = 0.01; // Smaller positions
            logger_1.logger.info('🛡️ Conservative mode enabled');
        }
    }
    /**
     * Display startup banner with system information
     */
    displayStartupBanner() {
        console.log('\n' + '='.repeat(80));
        console.log('🤖 ML-DRIVEN TRADING SYSTEM');
        console.log('   Integrating Analysis + Machine Learning + Execution');
        console.log('='.repeat(80));
        console.log(`📊 Trading Mode: ${ML_TRADING_CONFIG.enablePaperTrading ? 'PAPER TRADING' : '🚨 LIVE TRADING 🚨'}`);
        console.log(`🎯 Symbols: ${ML_TRADING_CONFIG.symbols.join(', ')}`);
        console.log(`⚡ Refresh Interval: ${ML_TRADING_CONFIG.refreshInterval}s`);
        console.log(`🧠 Min ML Confidence: ${(ML_TRADING_CONFIG.minConfidenceThreshold * 100).toFixed(0)}%`);
        console.log(`🔒 Max Position Size: ${(ML_TRADING_CONFIG.riskManagement.maxPositionSize * 100).toFixed(1)}%`);
        console.log(`🛡️ Max Daily Loss: ${(ML_TRADING_CONFIG.riskManagement.maxDailyLoss * 100).toFixed(1)}%`);
        console.log('='.repeat(80) + '\n');
    }
    /**
     * Validate environment and configuration
     */
    async validateEnvironment() {
        logger_1.logger.info('🔍 Validating environment...');
        // Check required environment variables
        const requiredEnvVars = [
            'DELTA_API_KEY',
            'DELTA_API_SECRET'
        ];
        for (const envVar of requiredEnvVars) {
            if (!process.env[envVar]) {
                throw new Error(`Missing required environment variable: ${envVar}`);
            }
        }
        // Validate trading symbols
        if (!ML_TRADING_CONFIG.symbols || ML_TRADING_CONFIG.symbols.length === 0) {
            throw new Error('No trading symbols configured');
        }
        // Validate risk management settings
        if (ML_TRADING_CONFIG.riskManagement.maxDailyLoss <= 0 || ML_TRADING_CONFIG.riskManagement.maxDailyLoss > 0.25) {
            throw new Error('Invalid max daily loss setting (must be between 0 and 25%)');
        }
        if (ML_TRADING_CONFIG.riskManagement.maxPositionSize <= 0 || ML_TRADING_CONFIG.riskManagement.maxPositionSize > 0.15) {
            throw new Error('Invalid max position size setting (must be between 0 and 15%)');
        }
        // Warn about live trading
        if (!ML_TRADING_CONFIG.enablePaperTrading) {
            logger_1.logger.warn('🚨 LIVE TRADING MODE DETECTED!');
            logger_1.logger.warn('🚨 REAL MONEY WILL BE USED FOR TRADING!');
            logger_1.logger.warn('🚨 ENSURE YOU UNDERSTAND THE RISKS!');
            // Add a delay to make sure user sees the warning
            await new Promise(resolve => setTimeout(resolve, 3000));
        }
        logger_1.logger.info('✅ Environment validation completed');
    }
    /**
     * Setup signal handlers for graceful shutdown
     */
    setupSignalHandlers() {
        const signals = ['SIGINT', 'SIGTERM', 'SIGQUIT'];
        signals.forEach(signal => {
            process.on(signal, async () => {
                if (this.isShuttingDown) {
                    logger_1.logger.warn('⚠️ Force shutdown requested');
                    process.exit(1);
                }
                this.isShuttingDown = true;
                logger_1.logger.info(`📡 Received ${signal} - initiating graceful shutdown...`);
                try {
                    await this.mlTrading.stopTrading();
                    logger_1.logger.info('✅ ML Trading System stopped gracefully');
                    process.exit(0);
                }
                catch (error) {
                    logger_1.logger.error('❌ Error during shutdown:', error);
                    process.exit(1);
                }
            });
        });
        // Handle uncaught exceptions
        process.on('uncaughtException', (error) => {
            logger_1.logger.error('💥 Uncaught Exception:', error);
            this.emergencyShutdown();
        });
        process.on('unhandledRejection', (reason, promise) => {
            logger_1.logger.error('💥 Unhandled Rejection at:', promise, 'reason:', reason);
            this.emergencyShutdown();
        });
    }
    /**
     * Emergency shutdown procedure
     */
    async emergencyShutdown() {
        logger_1.logger.error('🚨 EMERGENCY SHUTDOWN INITIATED');
        try {
            if (this.mlTrading) {
                await this.mlTrading.stopTrading();
            }
        }
        catch (error) {
            logger_1.logger.error('❌ Error during emergency shutdown:', error);
        }
        process.exit(1);
    }
    /**
     * Keep the process alive
     */
    async keepAlive() {
        return new Promise((resolve) => {
            // The process will be kept alive by the trading loops
            // This promise never resolves unless the system is shut down
        });
    }
}
exports.MLTradingSystemLauncher = MLTradingSystemLauncher;
// Main execution
async function main() {
    const launcher = new MLTradingSystemLauncher();
    await launcher.start();
}
// Start the system if this file is run directly
if (require.main === module) {
    main().catch((error) => {
        console.error('💥 Fatal error starting ML Trading System:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=start-ml-trading.js.map