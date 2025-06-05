"use strict";
/**
 * Delta Exchange Bot Manager
 * Manages multiple trading bots for Delta Exchange India testnet
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DeltaBotManager = void 0;
const events_1 = require("events");
const DeltaExchangeUnified_1 = require("./DeltaExchangeUnified");
const DeltaTradingBot_1 = require("./DeltaTradingBot");
const logger_1 = require("../utils/logger");
class DeltaBotManager extends events_1.EventEmitter {
    constructor() {
        super();
        this.bots = new Map();
        this.isInitialized = false;
    }
    /**
     * Initialize the bot manager with Delta Exchange credentials
     */
    async initialize() {
        try {
            logger_1.logger.info('🚀 Initializing Delta Bot Manager...');
            // Get credentials from environment
            const credentials = {
                apiKey: process.env.DELTA_EXCHANGE_API_KEY || '',
                apiSecret: process.env.DELTA_EXCHANGE_API_SECRET || '',
                testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
            };
            // Validate credentials
            if (!credentials.apiKey || !credentials.apiSecret) {
                throw new Error('Delta Exchange API credentials are required');
            }
            // Initialize Delta Exchange service
            this.deltaService = new DeltaExchangeUnified_1.DeltaExchangeUnified(credentials);
            // Wait for Delta service to be ready
            await this.waitForDeltaService();
            this.isInitialized = true;
            logger_1.logger.info('✅ Delta Bot Manager initialized successfully');
            this.emit('initialized');
        }
        catch (error) {
            logger_1.logger.error('❌ Failed to initialize Delta Bot Manager:', error);
            throw error;
        }
    }
    /**
     * Wait for Delta Exchange service to be ready
     */
    async waitForDeltaService() {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Delta Exchange service initialization timeout'));
            }, 30000); // 30 second timeout
            if (this.deltaService.isReady()) {
                clearTimeout(timeout);
                resolve();
                return;
            }
            this.deltaService.once('initialized', () => {
                clearTimeout(timeout);
                resolve();
            });
            this.deltaService.once('error', (error) => {
                clearTimeout(timeout);
                reject(error);
            });
        });
    }
    /**
     * Create a new trading bot
     */
    async createBot(config) {
        if (!this.isInitialized) {
            throw new Error('Bot manager not initialized');
        }
        try {
            // Validate bot configuration
            this.validateBotConfig(config);
            // Check if bot with same ID already exists
            if (this.bots.has(config.id)) {
                throw new Error(`Bot with ID ${config.id} already exists`);
            }
            // Create new bot instance
            const bot = new DeltaTradingBot_1.DeltaTradingBot(config, this.deltaService);
            // Set up bot event listeners
            this.setupBotEventListeners(bot);
            // Add bot to collection
            this.bots.set(config.id, bot);
            logger_1.logger.info(`✅ Created bot: ${config.name} (${config.id})`);
            this.emit('botCreated', { botId: config.id, config });
            return config.id;
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to create bot ${config.id}:`, error);
            throw error;
        }
    }
    /**
     * Start a trading bot
     */
    async startBot(botId) {
        const bot = this.bots.get(botId);
        if (!bot) {
            throw new Error(`Bot ${botId} not found`);
        }
        try {
            await bot.start();
            logger_1.logger.info(`🚀 Started bot: ${botId}`);
            this.emit('botStarted', { botId });
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to start bot ${botId}:`, error);
            throw error;
        }
    }
    /**
     * Stop a trading bot
     */
    async stopBot(botId) {
        const bot = this.bots.get(botId);
        if (!bot) {
            throw new Error(`Bot ${botId} not found`);
        }
        try {
            await bot.stop();
            logger_1.logger.info(`🛑 Stopped bot: ${botId}`);
            this.emit('botStopped', { botId });
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to stop bot ${botId}:`, error);
            throw error;
        }
    }
    /**
     * Pause a trading bot
     */
    pauseBot(botId) {
        const bot = this.bots.get(botId);
        if (!bot) {
            throw new Error(`Bot ${botId} not found`);
        }
        bot.pause();
        logger_1.logger.info(`⏸️ Paused bot: ${botId}`);
        this.emit('botPaused', { botId });
    }
    /**
     * Resume a trading bot
     */
    resumeBot(botId) {
        const bot = this.bots.get(botId);
        if (!bot) {
            throw new Error(`Bot ${botId} not found`);
        }
        bot.resume();
        logger_1.logger.info(`▶️ Resumed bot: ${botId}`);
        this.emit('botResumed', { botId });
    }
    /**
     * Remove a trading bot
     */
    async removeBot(botId) {
        const bot = this.bots.get(botId);
        if (!bot) {
            throw new Error(`Bot ${botId} not found`);
        }
        try {
            // Stop the bot first
            await bot.stop();
            // Clean up resources
            bot.cleanup();
            // Remove from collection
            this.bots.delete(botId);
            logger_1.logger.info(`🗑️ Removed bot: ${botId}`);
            this.emit('botRemoved', { botId });
        }
        catch (error) {
            logger_1.logger.error(`❌ Failed to remove bot ${botId}:`, error);
            throw error;
        }
    }
    /**
     * Get bot status
     */
    getBotStatus(botId) {
        const bot = this.bots.get(botId);
        if (!bot) {
            throw new Error(`Bot ${botId} not found`);
        }
        return bot.getStatus();
    }
    /**
     * Get all bot statuses
     */
    getAllBotStatuses() {
        return Array.from(this.bots.values()).map(bot => bot.getStatus());
    }
    /**
     * Get bot manager status
     */
    getManagerStatus() {
        const statuses = this.getAllBotStatuses();
        return {
            totalBots: statuses.length,
            runningBots: statuses.filter(s => s.status === 'running').length,
            pausedBots: statuses.filter(s => s.status === 'paused').length,
            stoppedBots: statuses.filter(s => s.status === 'stopped').length,
            errorBots: statuses.filter(s => s.status === 'error').length,
            totalTrades: statuses.reduce((sum, s) => sum + s.totalTrades, 0),
            totalPnL: statuses.reduce((sum, s) => sum + s.totalPnL, 0)
        };
    }
    /**
     * Update bot configuration
     */
    updateBotConfig(botId, newConfig) {
        const bot = this.bots.get(botId);
        if (!bot) {
            throw new Error(`Bot ${botId} not found`);
        }
        bot.updateConfig(newConfig);
        logger_1.logger.info(`🔧 Updated configuration for bot: ${botId}`);
        this.emit('botConfigUpdated', { botId, config: newConfig });
    }
    /**
     * Emergency stop all bots
     */
    async emergencyStopAll() {
        logger_1.logger.warn('🚨 Emergency stop all bots initiated');
        const stopPromises = Array.from(this.bots.entries()).map(async ([botId, bot]) => {
            try {
                await bot.emergencyCloseAll();
                await bot.stop();
                logger_1.logger.info(`🛑 Emergency stopped bot: ${botId}`);
            }
            catch (error) {
                logger_1.logger.error(`❌ Error in emergency stop for bot ${botId}:`, error);
            }
        });
        await Promise.allSettled(stopPromises);
        this.emit('emergencyStopAll');
    }
    /**
     * Get bot performance metrics
     */
    getBotPerformance(botId) {
        const bot = this.bots.get(botId);
        if (!bot) {
            throw new Error(`Bot ${botId} not found`);
        }
        return bot.getPerformanceMetrics();
    }
    /**
     * Get all bots performance summary
     */
    getAllBotsPerformance() {
        const performances = Array.from(this.bots.entries()).map(([botId, bot]) => ({
            botId,
            ...bot.getPerformanceMetrics()
        }));
        const summary = {
            totalBots: performances.length,
            totalTrades: performances.reduce((sum, p) => sum + p.totalTrades, 0),
            totalWinningTrades: performances.reduce((sum, p) => sum + p.winningTrades, 0),
            totalLosingTrades: performances.reduce((sum, p) => sum + p.losingTrades, 0),
            totalPnL: performances.reduce((sum, p) => sum + p.totalPnL, 0),
            averageWinRate: performances.length > 0
                ? performances.reduce((sum, p) => sum + parseFloat(p.winRate), 0) / performances.length
                : 0,
            bots: performances
        };
        return summary;
    }
    /**
     * Validate bot configuration
     */
    validateBotConfig(config) {
        if (!config.id || !config.name || !config.symbol) {
            throw new Error('Bot ID, name, and symbol are required');
        }
        if (config.capital <= 0) {
            throw new Error('Capital must be greater than 0');
        }
        if (config.leverage <= 0 || config.leverage > 100) {
            throw new Error('Leverage must be between 1 and 100');
        }
        if (config.riskPerTrade <= 0 || config.riskPerTrade > 100) {
            throw new Error('Risk per trade must be between 0 and 100');
        }
        // Check if product exists in Delta Exchange
        const productId = this.deltaService.getProductId(config.symbol);
        if (!productId) {
            throw new Error(`Product not found for symbol: ${config.symbol}`);
        }
    }
    /**
     * Set up event listeners for a bot
     */
    setupBotEventListeners(bot) {
        const botId = bot.getStatus().id;
        bot.on('started', (status) => {
            this.emit('botStatusChanged', { botId, status });
        });
        bot.on('stopped', (status) => {
            this.emit('botStatusChanged', { botId, status });
        });
        bot.on('paused', (status) => {
            this.emit('botStatusChanged', { botId, status });
        });
        bot.on('resumed', (status) => {
            this.emit('botStatusChanged', { botId, status });
        });
        bot.on('error', (error) => {
            logger_1.logger.error(`Bot ${botId} error:`, error);
            this.emit('botError', { botId, error });
        });
        bot.on('tradeExecuted', (data) => {
            logger_1.logger.info(`Bot ${botId} executed trade:`, data);
            this.emit('botTradeExecuted', { botId, ...data });
        });
        bot.on('positionClosed', (data) => {
            logger_1.logger.info(`Bot ${botId} closed position:`, data);
            this.emit('botPositionClosed', { botId, ...data });
        });
    }
    /**
     * Cleanup all resources
     */
    async cleanup() {
        logger_1.logger.info('🧹 Cleaning up Delta Bot Manager...');
        // Stop all bots
        const cleanupPromises = Array.from(this.bots.entries()).map(async ([botId, bot]) => {
            try {
                await bot.stop();
                bot.cleanup();
            }
            catch (error) {
                logger_1.logger.error(`Error cleaning up bot ${botId}:`, error);
            }
        });
        await Promise.allSettled(cleanupPromises);
        // Clear bots collection
        this.bots.clear();
        // Cleanup Delta service
        if (this.deltaService) {
            this.deltaService.cleanup();
        }
        this.removeAllListeners();
        logger_1.logger.info('✅ Delta Bot Manager cleaned up');
    }
}
exports.DeltaBotManager = DeltaBotManager;
//# sourceMappingURL=DeltaBotManager.js.map