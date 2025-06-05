#!/usr/bin/env node
"use strict";
/**
 * REAL TRADING ENGINE LAUNCHER
 *
 * ⚠️  WARNING: This script places REAL ORDERS on Delta Exchange with REAL MONEY!
 *
 * This is NOT a simulation - all trades will be executed live on your Delta Exchange account.
 * Make sure you understand the risks before running this script.
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.startRealTrading = startRealTrading;
const realTradingEngine_1 = require("../services/realTradingEngine");
const logger_1 = require("../utils/logger");
const dotenv_1 = __importDefault(require("dotenv"));
// Load environment variables
dotenv_1.default.config();
async function startRealTrading() {
    try {
        logger_1.logger.info('\n🚨 REAL TRADING ENGINE STARTUP');
        logger_1.logger.info('='.repeat(80));
        logger_1.logger.info('⚠️  WARNING: This will place REAL ORDERS with REAL MONEY!');
        logger_1.logger.info('💰 All trades will be executed live on Delta Exchange');
        logger_1.logger.info('🚨 Make sure you understand the risks before proceeding');
        logger_1.logger.info('='.repeat(80));
        // Get Delta Exchange credentials
        const deltaCredentials = {
            apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
            apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
            testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true' // Use testnet for safety
        };
        if (!deltaCredentials.apiKey || !deltaCredentials.apiSecret) {
            throw new Error('Delta Exchange API credentials not found in environment variables');
        }
        logger_1.logger.info(`🔗 Delta Exchange Mode: ${deltaCredentials.testnet ? 'TESTNET' : 'PRODUCTION'}`);
        if (!deltaCredentials.testnet) {
            logger_1.logger.warn('⚠️  PRODUCTION MODE: Using real money on live exchange!');
        }
        else {
            logger_1.logger.info('✅ TESTNET MODE: Using test environment');
        }
        // Enhanced real trading configuration
        const config = {
            balanceAllocationPercent: 75, // Use 75% of available balance
            maxLeverage: 100, // Delta Exchange testnet supports max 100x leverage
            riskPerTrade: 40, // High risk per trade (40% of balance)
            targetTradesPerDay: 4, // Target 4 trades per day
            targetWinRate: 75, // Target 75% win rate
            mlConfidenceThreshold: 80, // Require 80%+ ML confidence
            signalScoreThreshold: 72, // Require 72+ signal score
            qualityScoreThreshold: 78, // Require 78+ quality score
            maxDrawdownPercent: 20, // Stop if 20% drawdown
            tradingAssets: ['BTCUSD', 'ETHUSD'], // Trade BTC and ETH perpetuals
            checkIntervalMs: 30000, // Check every 30 seconds
            progressReportIntervalMs: 60000, // Report every 1 minute
        };
        logger_1.logger.info('\n⚙️  REAL TRADING CONFIGURATION:');
        logger_1.logger.info(`   Balance Allocation: ${config.balanceAllocationPercent}%`);
        logger_1.logger.info(`   Max Leverage: ${config.maxLeverage}x (Delta Exchange testnet limit)`);
        logger_1.logger.info(`   Risk per Trade: ${config.riskPerTrade}%`);
        logger_1.logger.info(`   Target Trades/Day: ${config.targetTradesPerDay}`);
        logger_1.logger.info(`   Target Win Rate: ${config.targetWinRate}%`);
        logger_1.logger.info(`   ML Confidence Threshold: ${config.mlConfidenceThreshold}%`);
        logger_1.logger.info(`   Signal Score Threshold: ${config.signalScoreThreshold}`);
        logger_1.logger.info(`   Quality Score Threshold: ${config.qualityScoreThreshold}`);
        logger_1.logger.info(`   Max Drawdown: ${config.maxDrawdownPercent}%`);
        logger_1.logger.info(`   Trading Assets: ${config.tradingAssets.join(', ')}`);
        logger_1.logger.info(`   💰 Will use ACTUAL testnet balance from your account`);
        // Initialize real trading engine
        const realTrader = new realTradingEngine_1.RealTradingEngine(deltaCredentials, config);
        logger_1.logger.info('\n🔄 Initializing real trading engine...');
        logger_1.logger.info('💰 This will fetch your REAL balance from Delta Exchange');
        logger_1.logger.info('🚀 All subsequent trades will use REAL MONEY');
        // Start real trading
        await realTrader.startRealTrading();
    }
    catch (error) {
        logger_1.logger.error('❌ Failed to start real trading:', error);
        process.exit(1);
    }
}
// Handle graceful shutdown
process.on('SIGINT', () => {
    logger_1.logger.info('\n🛑 Received SIGINT, shutting down real trading...');
    process.exit(0);
});
process.on('SIGTERM', () => {
    logger_1.logger.info('\n🛑 Received SIGTERM, shutting down real trading...');
    process.exit(0);
});
// Start the real trading engine
if (require.main === module) {
    startRealTrading().catch(error => {
        logger_1.logger.error('❌ Fatal error in real trading:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=start-real-trading.js.map