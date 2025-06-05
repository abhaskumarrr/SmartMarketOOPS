#!/usr/bin/env ts-node
"use strict";
/**
 * Enhanced Paper Trading Engine Startup Script
 * Starts paper trading with 75% balance allocation and frequency optimization
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.startEnhancedPaperTrading = startEnhancedPaperTrading;
const paperTradingEngine_1 = require("../services/paperTradingEngine");
const logger_1 = require("../utils/logger");
const dotenv_1 = __importDefault(require("dotenv"));
// Load environment variables
dotenv_1.default.config();
async function startEnhancedPaperTrading() {
    try {
        logger_1.logger.info('\n🚀 STARTING ENHANCED PAPER TRADING ENGINE');
        logger_1.logger.info('═'.repeat(80));
        logger_1.logger.info('🎯 FREQUENCY OPTIMIZED TRADING WITH 75% BALANCE ALLOCATION');
        logger_1.logger.info('⚡ TARGETING 3-5 PROFITABLE TRADES DAILY WITH 75%+ WIN RATE');
        logger_1.logger.info('═'.repeat(80));
        // Delta Exchange credentials from environment
        const deltaCredentials = {
            apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
            apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
            testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true' // Using testnet for paper trading
        };
        // Validate credentials
        if (!deltaCredentials.apiKey || !deltaCredentials.apiSecret) {
            throw new Error('❌ Delta Exchange API credentials not found in environment variables');
        }
        logger_1.logger.info('✅ Delta Exchange credentials loaded');
        logger_1.logger.info(`🔗 Using: ${deltaCredentials.testnet ? 'TESTNET' : 'PRODUCTION'} mode`);
        // Enhanced frequency-optimized configuration
        const config = {
            mlConfidenceThreshold: 80, // 80%+ ML confidence required
            signalScoreThreshold: 72, // 72+/100 signal score required
            qualityScoreThreshold: 78, // 78+/100 quality score required
            targetTradesPerDay: 4, // Target 3-5 trades daily
            targetWinRate: 75, // Target 75% win rate
            mlAccuracy: 85, // 85% ML accuracy
            maxConcurrentTrades: 3, // Max 3 concurrent trades
            balanceAllocationPercent: 75 // Use 75% of total balance
        };
        logger_1.logger.info('\n⚡ FREQUENCY OPTIMIZATION CONFIGURATION:');
        logger_1.logger.info(`   ML Confidence Threshold: ${config.mlConfidenceThreshold}%`);
        logger_1.logger.info(`   Signal Score Threshold: ${config.signalScoreThreshold}/100`);
        logger_1.logger.info(`   Quality Score Threshold: ${config.qualityScoreThreshold}/100`);
        logger_1.logger.info(`   Target Trades Per Day: ${config.targetTradesPerDay}`);
        logger_1.logger.info(`   Target Win Rate: ${config.targetWinRate}%`);
        logger_1.logger.info(`   ML Accuracy: ${config.mlAccuracy}%`);
        logger_1.logger.info(`   Max Concurrent Trades: ${config.maxConcurrentTrades}`);
        logger_1.logger.info(`   Balance Allocation: ${config.balanceAllocationPercent}%`);
        // Initialize enhanced paper trading engine
        const paperTrader = new paperTradingEngine_1.PaperTradingEngine(deltaCredentials, config);
        logger_1.logger.info('\n🔄 Initializing enhanced paper trading engine...');
        // Start paper trading
        await paperTrader.startPaperTrading();
        // Handle graceful shutdown
        process.on('SIGINT', () => {
            logger_1.logger.info('\n🛑 Received shutdown signal, stopping paper trading...');
            paperTrader.stopPaperTrading();
            process.exit(0);
        });
        process.on('SIGTERM', () => {
            logger_1.logger.info('\n🛑 Received termination signal, stopping paper trading...');
            paperTrader.stopPaperTrading();
            process.exit(0);
        });
    }
    catch (error) {
        logger_1.logger.error('❌ Failed to start enhanced paper trading engine:', error);
        process.exit(1);
    }
}
// Start the enhanced paper trading engine
if (require.main === module) {
    startEnhancedPaperTrading().catch(error => {
        logger_1.logger.error('❌ Unhandled error in paper trading startup:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=start-paper-trading.js.map