"use strict";
/**
 * Bot Management Service
 * Handles bot configuration, lifecycle, and monitoring
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getBacktestHistory = exports.runBacktest = exports.updateBotHealth = exports.configureBotRiskSettings = exports.getBotStatus = exports.pauseBot = exports.stopBot = exports.startBot = exports.deleteBot = exports.updateBot = exports.getBotById = exports.getBotsByUser = exports.createBot = void 0;
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
const axios_1 = __importDefault(require("axios"));
const auditLog_1 = require("../../utils/auditLog");
const decisionLogService_1 = require("../decisionLogService");
// Import WebSocket broadcasting function
let broadcastBotUpdate = null;
try {
    const websocketModule = require('../../sockets/websocketServer');
    broadcastBotUpdate = websocketModule.broadcastBotUpdate;
}
catch (error) {
    console.warn('WebSocket module not available for bot broadcasting');
}
// In-memory storage for bot status and health
const botStatusRegistry = new Map();
// Default bot status
const getDefaultBotStatus = () => ({
    isRunning: false,
    lastUpdate: new Date().toISOString(),
    health: 'unknown',
    metrics: {
        tradesExecuted: 0,
        profitLoss: 0,
        successRate: 0,
        latency: 0
    },
    activePositions: 0,
    errors: [],
    logs: []
});
/**
 * Create a new trading bot
 */
const createBot = async (userId, botData) => {
    try {
        // Create the bot
        const bot = await prismaClient_1.default.bot.create({
            data: {
                userId,
                name: botData.name,
                symbol: botData.symbol,
                strategy: botData.strategy,
                timeframe: botData.timeframe,
                parameters: botData.parameters || {},
                isActive: false
            }
        });
        // Initialize bot status in registry
        botStatusRegistry.set(bot.id, getDefaultBotStatus());
        // Create audit log entry
        await (0, auditLog_1.createAuditLog)({
            userId,
            action: 'bot.create',
            details: {
                botId: bot.id,
                botName: bot.name,
                symbol: bot.symbol,
                strategy: bot.strategy
            }
        });
        return bot;
    }
    catch (error) {
        console.error('Error creating bot:', error);
        throw error;
    }
};
exports.createBot = createBot;
/**
 * Get all bots for a user
 */
const getBotsByUser = async (userId) => {
    try {
        return await prismaClient_1.default.bot.findMany({
            where: {
                userId
            },
            include: {
                riskSettings: true
            }
        });
    }
    catch (error) {
        console.error('Error getting bots:', error);
        throw error;
    }
};
exports.getBotsByUser = getBotsByUser;
/**
 * Get a specific bot by ID
 */
const getBotById = async (botId, userId) => {
    try {
        return await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            },
            include: {
                riskSettings: true
            }
        });
    }
    catch (error) {
        console.error('Error getting bot:', error);
        throw error;
    }
};
exports.getBotById = getBotById;
/**
 * Update a bot's configuration
 */
const updateBot = async (botId, userId, updateData) => {
    try {
        // Find bot first to check existence and ownership
        const existingBot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            }
        });
        if (!existingBot) {
            throw new Error('Bot not found or access denied');
        }
        // Don't allow updates while bot is active
        if (existingBot.isActive) {
            throw new Error('Cannot update bot while it is active. Stop the bot first.');
        }
        // Update the bot
        const updatedBot = await prismaClient_1.default.bot.update({
            where: {
                id: botId
            },
            data: {
                ...(updateData.name && { name: updateData.name }),
                ...(updateData.symbol && { symbol: updateData.symbol }),
                ...(updateData.strategy && { strategy: updateData.strategy }),
                ...(updateData.timeframe && { timeframe: updateData.timeframe }),
                ...(updateData.parameters && { parameters: updateData.parameters })
            }
        });
        // Create audit log entry
        await (0, auditLog_1.createAuditLog)({
            userId,
            action: 'bot.update',
            details: {
                botId,
                updatedFields: Object.keys(updateData)
            }
        });
        return updatedBot;
    }
    catch (error) {
        console.error('Error updating bot:', error);
        throw error;
    }
};
exports.updateBot = updateBot;
/**
 * Delete a bot
 */
const deleteBot = async (botId, userId) => {
    try {
        // Find bot first to check existence and ownership
        const existingBot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            }
        });
        if (!existingBot) {
            throw new Error('Bot not found or access denied');
        }
        // Stop bot if it's active
        if (existingBot.isActive) {
            await (0, exports.stopBot)(botId, userId);
        }
        // Delete bot
        await prismaClient_1.default.bot.delete({
            where: {
                id: botId
            }
        });
        // Remove from status registry
        botStatusRegistry.delete(botId);
        // Create audit log entry
        await (0, auditLog_1.createAuditLog)({
            userId,
            action: 'bot.delete',
            details: {
                botId,
                botName: existingBot.name
            }
        });
        return true;
    }
    catch (error) {
        console.error('Error deleting bot:', error);
        throw error;
    }
};
exports.deleteBot = deleteBot;
/**
 * Start a bot
 */
const startBot = async (botId, userId) => {
    try {
        // Find bot first to check existence and ownership
        const bot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            },
            include: {
                riskSettings: true
            }
        });
        if (!bot) {
            throw new Error('Bot not found or access denied');
        }
        // Check if bot is already active
        if (bot.isActive) {
            throw new Error('Bot is already running');
        }
        // Perform risk check before starting
        if (!bot.riskSettings || bot.riskSettings.length === 0) {
            throw new Error('Bot has no risk settings. Please configure risk settings before starting.');
        }
        // Start bot process
        const success = await startBotProcess(bot);
        if (!success) {
            throw new Error('Failed to start bot process');
        }
        // Update bot status in database
        await prismaClient_1.default.bot.update({
            where: {
                id: botId
            },
            data: {
                isActive: true
            }
        });
        // Create decision log
        await (0, decisionLogService_1.createDecisionLog)({
            source: 'User',
            actionType: 'BotControl',
            decision: `Start bot ${bot.name}`,
            userId,
            botId,
            reasonDetails: 'User initiated bot start',
            symbol: bot.symbol,
            importance: 'HIGH',
            tags: ['bot-control', 'start']
        });
        // Create audit log entry
        await (0, auditLog_1.createAuditLog)({
            userId,
            action: 'bot.start',
            details: {
                botId,
                botName: bot.name,
                symbol: bot.symbol,
                strategy: bot.strategy
            }
        });
        // Broadcast bot status update via WebSocket
        if (broadcastBotUpdate) {
            const status = botStatusRegistry.get(botId) || getDefaultBotStatus();
            status.isRunning = true;
            status.lastUpdate = new Date().toISOString();
            status.logs.push({
                timestamp: new Date().toISOString(),
                level: 'info',
                message: 'Bot started successfully'
            });
            botStatusRegistry.set(botId, status);
            broadcastBotUpdate(botId, {
                status: 'started',
                isRunning: true,
                botName: bot.name,
                symbol: bot.symbol,
                strategy: bot.strategy,
                timestamp: new Date().toISOString()
            });
        }
        return true;
    }
    catch (error) {
        console.error('Error starting bot:', error);
        throw error;
    }
};
exports.startBot = startBot;
/**
 * Stop a bot
 */
const stopBot = async (botId, userId) => {
    try {
        // Find bot first to check existence and ownership
        const bot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            }
        });
        if (!bot) {
            throw new Error('Bot not found or access denied');
        }
        // Check if bot is actually active
        if (!bot.isActive) {
            throw new Error('Bot is not running');
        }
        // Stop bot process
        const success = await stopBotProcess(botId);
        if (!success) {
            throw new Error('Failed to stop bot process');
        }
        // Update bot status in database
        await prismaClient_1.default.bot.update({
            where: {
                id: botId
            },
            data: {
                isActive: false
            }
        });
        // Create decision log
        await (0, decisionLogService_1.createDecisionLog)({
            source: 'User',
            actionType: 'BotControl',
            decision: `Stop bot ${bot.name}`,
            userId,
            botId,
            reasonDetails: 'User initiated bot stop',
            symbol: bot.symbol,
            importance: 'HIGH',
            tags: ['bot-control', 'stop']
        });
        // Create audit log entry
        await (0, auditLog_1.createAuditLog)({
            userId,
            action: 'bot.stop',
            details: {
                botId,
                botName: bot.name
            }
        });
        // Broadcast bot status update via WebSocket
        if (broadcastBotUpdate) {
            const status = botStatusRegistry.get(botId) || getDefaultBotStatus();
            status.isRunning = false;
            status.lastUpdate = new Date().toISOString();
            status.logs.push({
                timestamp: new Date().toISOString(),
                level: 'info',
                message: 'Bot stopped successfully'
            });
            botStatusRegistry.set(botId, status);
            broadcastBotUpdate(botId, {
                status: 'stopped',
                isRunning: false,
                botName: bot.name,
                timestamp: new Date().toISOString()
            });
        }
        return true;
    }
    catch (error) {
        console.error('Error stopping bot:', error);
        throw error;
    }
};
exports.stopBot = stopBot;
/**
 * Pause a bot (temporarily suspend operations without full stop)
 */
const pauseBot = async (botId, userId, duration) => {
    try {
        // Find bot first to check existence and ownership
        const bot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            }
        });
        if (!bot) {
            throw new Error('Bot not found or access denied');
        }
        // Check if bot is active
        if (!bot.isActive) {
            throw new Error('Bot is not running');
        }
        // Get current status
        const status = botStatusRegistry.get(botId) || getDefaultBotStatus();
        // Update status to indicate pause
        status.logs.push({
            timestamp: new Date().toISOString(),
            level: 'info',
            message: `Bot paused ${duration ? `for ${duration} seconds` : 'indefinitely'}`
        });
        botStatusRegistry.set(botId, status);
        // Call ML service to pause bot
        try {
            await axios_1.default.post(`${process.env.ML_SERVICE_URL || 'http://localhost:8000'}/api/bots/pause`, {
                botId,
                duration
            });
        }
        catch (error) {
            console.error('Error calling ML service for pause:', error);
            throw new Error('Failed to communicate with ML service to pause bot');
        }
        // Create decision log
        await (0, decisionLogService_1.createDecisionLog)({
            source: 'User',
            actionType: 'BotControl',
            decision: `Pause bot ${bot.name}${duration ? ` for ${duration} seconds` : ''}`,
            userId,
            botId,
            reasonDetails: 'User initiated bot pause',
            symbol: bot.symbol,
            importance: 'MEDIUM',
            tags: ['bot-control', 'pause']
        });
        // Create audit log entry
        await (0, auditLog_1.createAuditLog)({
            userId,
            action: 'bot.pause',
            details: {
                botId,
                botName: bot.name,
                duration
            }
        });
        return true;
    }
    catch (error) {
        console.error('Error pausing bot:', error);
        throw error;
    }
};
exports.pauseBot = pauseBot;
/**
 * Get bot status
 */
const getBotStatus = async (botId, userId) => {
    try {
        // First check in-memory status
        const status = botStatusRegistry.get(botId);
        if (status) {
            return status;
        }
        // If not found in memory, check database
        const bot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            },
            include: {
                positions: {
                    where: {
                        status: 'Open'
                    }
                }
            }
        });
        if (!bot) {
            return null;
        }
        // Create default status based on bot.status field
        const defaultStatus = {
            id: bot.id,
            name: bot.name,
            symbol: bot.symbol,
            strategy: bot.strategy,
            timeframe: bot.timeframe,
            isActive: bot.isActive,
            status: {
                id: bot.id,
                name: bot.name,
                symbol: bot.symbol,
                strategy: bot.strategy,
                timeframe: bot.timeframe,
                isActive: bot.isActive,
                status: bot.status
            },
            lastUpdate: new Date().toISOString(),
            health: 'unknown',
            metrics: {},
            activePositions: bot.positions.length,
            errors: [],
            logs: []
        };
        // Store in memory for future requests
        botStatusRegistry.set(bot.id, defaultStatus);
        return defaultStatus;
    }
    catch (error) {
        console.error('Error getting bot status:', error);
        throw error;
    }
};
exports.getBotStatus = getBotStatus;
/**
 * Configure risk settings for a bot
 */
const configureBotRiskSettings = async (botId, userId, riskConfig) => {
    try {
        // Find bot first to check existence and ownership
        const bot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            },
            include: {
                riskSettings: true
            }
        });
        if (!bot) {
            throw new Error('Bot not found or access denied');
        }
        // Check if bot already has risk settings
        if (bot.riskSettings && bot.riskSettings.length > 0) {
            // Update existing risk settings
            const updatedSettings = await prismaClient_1.default.riskSettings.update({
                where: {
                    id: bot.riskSettings[0].id
                },
                data: {
                    ...riskConfig,
                    userId
                }
            });
            // Create audit log entry
            await (0, auditLog_1.createAuditLog)({
                userId,
                action: 'bot.risk.update',
                details: {
                    botId,
                    botName: bot.name,
                    riskSettingsId: updatedSettings.id,
                    updatedFields: Object.keys(riskConfig)
                }
            });
            return updatedSettings;
        }
        else {
            // Create new risk settings
            if (!riskConfig.name) {
                riskConfig.name = `${bot.name} Risk Settings`;
            }
            // Set required fields if not provided
            const defaultRiskSettings = {
                positionSizingMethod: 'FIXED_FRACTIONAL',
                riskPercentage: 1.0,
                maxPositionSize: 1000.0,
                stopLossType: 'PERCENTAGE',
                stopLossValue: 2.0,
                takeProfitType: 'PERCENTAGE',
                takeProfitValue: 3.0,
                maxRiskPerTrade: 2.0,
                maxRiskPerSymbol: 5.0,
                maxRiskPerDirection: 10.0,
                maxTotalRisk: 20.0,
                maxDrawdown: 15.0,
                maxPositions: 5,
                maxDailyLoss: 5.0,
                cooldownPeriod: 300,
                volatilityLookback: 14,
                circuitBreakerEnabled: true,
                maxDailyLossBreaker: 10.0,
                maxDrawdownBreaker: 20.0,
                volatilityMultiplier: 2.5,
                consecutiveLossesBreaker: 3,
                tradingPause: 3600,
                marketWideEnabled: true,
                enableManualOverride: true
            };
            const newSettings = await prismaClient_1.default.riskSettings.create({
                data: {
                    ...defaultRiskSettings,
                    ...riskConfig,
                    userId,
                    botId
                }
            });
            // Create audit log entry
            await (0, auditLog_1.createAuditLog)({
                userId,
                action: 'bot.risk.create',
                details: {
                    botId,
                    botName: bot.name,
                    riskSettingsId: newSettings.id
                }
            });
            return newSettings;
        }
    }
    catch (error) {
        console.error('Error configuring bot risk settings:', error);
        throw error;
    }
};
exports.configureBotRiskSettings = configureBotRiskSettings;
/**
 * Helper function to start a bot process
 */
async function startBotProcess(bot) {
    try {
        // Update in-memory status
        const status = botStatusRegistry.get(bot.id) || getDefaultBotStatus();
        status.isRunning = true;
        status.lastUpdate = new Date().toISOString();
        status.health = 'good';
        status.logs.push({
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Bot process started'
        });
        botStatusRegistry.set(bot.id, status);
        // Try to call ML service to start bot
        try {
            await axios_1.default.post(`${process.env.ML_SERVICE_URL || 'http://localhost:8000'}/api/bots/start`, {
                botId: bot.id,
                config: {
                    symbol: bot.symbol,
                    strategy: bot.strategy,
                    timeframe: bot.timeframe,
                    parameters: bot.parameters,
                    riskSettings: bot.riskSettings[0] || {}
                }
            });
        }
        catch (error) {
            console.error('Error calling ML service to start bot:', error);
            status.errors.push({
                timestamp: new Date().toISOString(),
                message: 'Failed to communicate with ML service',
                code: 'ML_SERVICE_ERROR'
            });
            botStatusRegistry.set(bot.id, status);
            return false;
        }
        return true;
    }
    catch (error) {
        console.error('Start bot process error:', error);
        return false;
    }
}
/**
 * Helper function to stop a bot process
 */
async function stopBotProcess(botId) {
    try {
        // Update in-memory status
        const status = botStatusRegistry.get(botId) || getDefaultBotStatus();
        status.isRunning = false;
        status.lastUpdate = new Date().toISOString();
        status.logs.push({
            timestamp: new Date().toISOString(),
            level: 'info',
            message: 'Bot process stopped'
        });
        botStatusRegistry.set(botId, status);
        // Try to call ML service to stop bot
        try {
            await axios_1.default.post(`${process.env.ML_SERVICE_URL || 'http://localhost:8000'}/api/bots/stop`, {
                botId
            });
        }
        catch (error) {
            console.error('Error calling ML service to stop bot:', error);
            status.errors.push({
                timestamp: new Date().toISOString(),
                message: 'Failed to communicate with ML service',
                code: 'ML_SERVICE_ERROR'
            });
            botStatusRegistry.set(botId, status);
            // Continue anyway, as we're stopping the bot
        }
        return true;
    }
    catch (error) {
        console.error('Stop bot process error:', error);
        return false;
    }
}
/**
 * Update bot health status
 * This function would be called by a health monitoring service
 */
const updateBotHealth = async (botId, healthData) => {
    try {
        // Get current status
        const status = botStatusRegistry.get(botId) || getDefaultBotStatus();
        // Update health and metrics
        status.health = healthData.health;
        status.metrics = {
            ...status.metrics,
            ...healthData.metrics
        };
        status.lastUpdate = new Date().toISOString();
        // Add errors if provided
        if (healthData.errors && healthData.errors.length > 0) {
            status.errors = [
                ...status.errors,
                ...healthData.errors
            ].slice(-50); // Keep only the last 50 errors
        }
        // Add logs if provided
        if (healthData.logs && healthData.logs.length > 0) {
            status.logs = [
                ...status.logs,
                ...healthData.logs
            ].slice(-100); // Keep only the last 100 logs
        }
        // Update registry
        botStatusRegistry.set(botId, status);
        return true;
    }
    catch (error) {
        console.error('Error updating bot health:', error);
        return false;
    }
};
exports.updateBotHealth = updateBotHealth;
/**
 * Run backtest for a bot
 */
const runBacktest = async (botId, userId, config) => {
    try {
        // Find bot first to check existence and ownership
        const bot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            }
        });
        if (!bot) {
            throw new Error('Bot not found or access denied');
        }
        // Call backtesting engine
        const backtestingEngine = await Promise.resolve().then(() => __importStar(require('./backtestingEngine')));
        const result = await backtestingEngine.runBacktest({
            botId,
            strategy: bot.strategy,
            parameters: bot.parameters,
            ...config
        });
        // Create audit log entry
        await (0, auditLog_1.createAuditLog)({
            userId,
            action: 'bot.backtest',
            details: {
                botId,
                symbol: config.symbol,
                timeframe: config.timeframe,
                performance: result.performance
            }
        });
        return result;
    }
    catch (error) {
        console.error('Error running backtest:', error);
        throw error;
    }
};
exports.runBacktest = runBacktest;
/**
 * Get backtest history for a bot
 */
const getBacktestHistory = async (botId, userId, limit = 10, offset = 0) => {
    try {
        // Find bot first to check existence and ownership
        const bot = await prismaClient_1.default.bot.findFirst({
            where: {
                id: botId,
                userId
            }
        });
        if (!bot) {
            throw new Error('Bot not found or access denied');
        }
        // For now, return mock data since we don't have a backtest table
        // TODO: Implement proper backtest storage in database
        const mockBacktests = [
            {
                id: `backtest_${Date.now()}`,
                botId,
                symbol: bot.symbol,
                timeframe: bot.timeframe,
                startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
                endDate: new Date(),
                performance: {
                    totalReturn: 1250.75,
                    totalReturnPercent: 12.51,
                    sharpeRatio: 1.85,
                    winRate: 68.5,
                    totalTrades: 127
                },
                createdAt: new Date()
            }
        ];
        return {
            backtests: mockBacktests.slice(offset, offset + limit),
            total: mockBacktests.length,
            limit,
            offset,
            hasMore: offset + limit < mockBacktests.length
        };
    }
    catch (error) {
        console.error('Error getting backtest history:', error);
        throw error;
    }
};
exports.getBacktestHistory = getBacktestHistory;
//# sourceMappingURL=botService.js.map