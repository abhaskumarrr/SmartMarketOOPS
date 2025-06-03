#!/usr/bin/env node
"use strict";
/**
 * AI Position Manager Runner
 * Manages your Delta Exchange positions with AI-powered dynamic take profit
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AIPositionManagerRunner = void 0;
const deltaApiService_1 = __importDefault(require("../services/deltaApiService"));
const aiPositionManager_1 = require("../services/aiPositionManager");
const logger_1 = require("../utils/logger");
class AIPositionManagerRunner {
    constructor() {
        this.isRunning = false;
        this.deltaApi = new deltaApiService_1.default({ testnet: true });
        this.aiManager = new aiPositionManager_1.AIPositionManager(this.deltaApi);
    }
    /**
     * Start AI position management
     */
    async start() {
        try {
            logger_1.logger.info('🤖 STARTING AI POSITION MANAGEMENT SYSTEM');
            logger_1.logger.info('='.repeat(80));
            // Get credentials
            const credentials = {
                key: process.env.DELTA_EXCHANGE_API_KEY || '',
                secret: process.env.DELTA_EXCHANGE_API_SECRET || ''
            };
            if (!credentials.key || !credentials.secret) {
                logger_1.logger.error('❌ Delta API credentials not found');
                logger_1.logger.info('🔧 Please set DELTA_EXCHANGE_API_KEY and DELTA_EXCHANGE_API_SECRET');
                return;
            }
            logger_1.logger.info('🔑 Credentials found');
            logger_1.logger.info(`   API Key: ${credentials.key.substring(0, 8)}...`);
            // Initialize Delta API
            await this.deltaApi.initialize(credentials);
            logger_1.logger.info('✅ Delta Exchange API initialized');
            // Test connection
            await this.testConnection();
            // Start AI management
            await this.aiManager.startManagement();
            this.isRunning = true;
            logger_1.logger.info('🚀 AI Position Management System is now running!');
            logger_1.logger.info('📊 Monitoring your positions every 30 seconds...');
            logger_1.logger.info('🤖 AI will automatically manage take profits, stop losses, and exits');
            logger_1.logger.info('\n💡 Features active:');
            logger_1.logger.info('   ✅ Dynamic take profit levels');
            logger_1.logger.info('   ✅ Trailing stop losses');
            logger_1.logger.info('   ✅ Partial exit optimization');
            logger_1.logger.info('   ✅ Market regime adaptation');
            logger_1.logger.info('   ✅ Risk management');
            // Keep running
            await this.keepRunning();
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to start AI position manager:', error.message);
            if (error.message.includes('ip_not_whitelisted')) {
                logger_1.logger.info('\n🔧 IP WHITELISTING REQUIRED:');
                logger_1.logger.info('   1. Login to your Delta Exchange account');
                logger_1.logger.info('   2. Go to API Management section');
                logger_1.logger.info('   3. Edit your API key settings');
                logger_1.logger.info('   4. Add your IP address: 223.226.141.59');
                logger_1.logger.info('   5. Save changes and try again');
            }
        }
    }
    /**
     * Test connection to Delta Exchange
     */
    async testConnection() {
        try {
            // Test public endpoint
            const markets = await this.deltaApi.getMarkets();
            logger_1.logger.info(`✅ Connection test passed - ${markets.length} markets available`);
            // Test authenticated endpoint
            try {
                const positions = await this.deltaApi.getPositions();
                logger_1.logger.info(`✅ Authentication successful - ${positions.length} positions found`);
                if (positions.length > 0) {
                    logger_1.logger.info('🎯 POSITIONS TO MANAGE:');
                    positions.forEach((pos, index) => {
                        const side = parseFloat(pos.size) > 0 ? 'LONG' : 'SHORT';
                        logger_1.logger.info(`   ${index + 1}. ${pos.symbol}: ${side} ${Math.abs(parseFloat(pos.size))} @ $${pos.entry_price}`);
                    });
                }
                else {
                    logger_1.logger.info('📊 No active positions found - AI will monitor for new positions');
                }
            }
            catch (authError) {
                if (authError.message.includes('ip_not_whitelisted')) {
                    throw new Error('ip_not_whitelisted_for_api_key');
                }
                throw authError;
            }
        }
        catch (error) {
            throw error;
        }
    }
    /**
     * Keep the system running
     */
    async keepRunning() {
        // Set up graceful shutdown
        process.on('SIGINT', () => {
            logger_1.logger.info('\n🛑 Received shutdown signal...');
            this.stop();
            process.exit(0);
        });
        process.on('SIGTERM', () => {
            logger_1.logger.info('\n🛑 Received termination signal...');
            this.stop();
            process.exit(0);
        });
        // Display status every 5 minutes
        setInterval(() => {
            this.displayStatus();
        }, 300000); // 5 minutes
        // Keep process alive
        while (this.isRunning) {
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }
    /**
     * Display current status
     */
    displayStatus() {
        const managedPositions = this.aiManager.getManagedPositions();
        logger_1.logger.info('\n🤖 AI POSITION MANAGER STATUS:');
        logger_1.logger.info(`   📊 Positions under management: ${managedPositions.length}`);
        logger_1.logger.info(`   🕐 Last update: ${new Date().toLocaleTimeString()}`);
        if (managedPositions.length > 0) {
            logger_1.logger.info('   📈 Position Summary:');
            managedPositions.forEach(pos => {
                const profitPercent = ((pos.currentPrice - pos.entryPrice) / pos.entryPrice) * 100;
                const executedExits = pos.partialExits.filter(e => e.executed).length;
                logger_1.logger.info(`     ${pos.symbol}: ${pos.side} $${pos.unrealizedPnl.toFixed(2)} (${profitPercent.toFixed(1)}%)`);
                logger_1.logger.info(`       Exits: ${executedExits}/${pos.partialExits.length}, Stop: $${pos.stopLoss.toFixed(2)}`);
            });
        }
    }
    /**
     * Stop AI position management
     */
    stop() {
        this.isRunning = false;
        this.aiManager.stopManagement();
        logger_1.logger.info('🛑 AI Position Management System stopped');
    }
}
exports.AIPositionManagerRunner = AIPositionManagerRunner;
/**
 * Main execution
 */
async function main() {
    const runner = new AIPositionManagerRunner();
    await runner.start();
}
// Run if this script is executed directly
if (require.main === module) {
    main().catch(error => {
        logger_1.logger.error('💥 AI Position Manager failed:', error);
        process.exit(1);
    });
}
//# sourceMappingURL=run-ai-position-manager.js.map