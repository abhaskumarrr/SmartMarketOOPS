"use strict";
/**
 * Trading Bot Controller
 * Handles trading bot configuration and control endpoints
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.updateBotHealth = exports.configureBotRiskSettings = exports.getBotStatus = exports.getBacktestHistory = exports.runBacktest = exports.pauseBot = exports.stopBot = exports.startBot = exports.deleteBot = exports.updateBot = exports.getBot = exports.getBots = exports.createBot = void 0;
const botService = __importStar(require("../services/trading/botService"));
const validator_1 = require("../utils/validator");
const express_validator_1 = require("express-validator");
/**
 * Create a new trading bot
 * @route POST /api/bots
 * @access Private
 */
const createBot = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.body)('name').notEmpty().withMessage('Name is required'),
            (0, express_validator_1.body)('symbol').notEmpty().withMessage('Symbol is required'),
            (0, express_validator_1.body)('strategy').notEmpty().withMessage('Strategy is required'),
            (0, express_validator_1.body)('timeframe').notEmpty().withMessage('Timeframe is required'),
            (0, express_validator_1.body)('parameters').optional().isObject().withMessage('Parameters must be an object')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botData = {
            name: req.body.name,
            symbol: req.body.symbol,
            strategy: req.body.strategy,
            timeframe: req.body.timeframe,
            parameters: req.body.parameters
        };
        const bot = await botService.createBot(userId, botData);
        return res.status(201).json({
            success: true,
            message: 'Bot created successfully',
            data: bot
        });
    }
    catch (error) {
        console.error('Create bot error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while creating bot',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.createBot = createBot;
/**
 * Get all trading bots for current user
 * @route GET /api/bots
 * @access Private
 */
const getBots = async (req, res) => {
    try {
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const bots = await botService.getBotsByUser(userId);
        return res.status(200).json({
            success: true,
            data: bots
        });
    }
    catch (error) {
        console.error('Get bots error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while fetching bots',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.getBots = getBots;
/**
 * Get a specific trading bot
 * @route GET /api/bots/:id
 * @access Private
 */
const getBot = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        const bot = await botService.getBotById(botId, userId);
        if (!bot) {
            return res.status(404).json({
                success: false,
                message: 'Bot not found'
            });
        }
        return res.status(200).json({
            success: true,
            data: bot
        });
    }
    catch (error) {
        console.error('Get bot error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while fetching bot',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.getBot = getBot;
/**
 * Update a trading bot
 * @route PUT /api/bots/:id
 * @access Private
 */
const updateBot = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required'),
            (0, express_validator_1.body)('name').optional().isString().withMessage('Name must be a string'),
            (0, express_validator_1.body)('symbol').optional().isString().withMessage('Symbol must be a string'),
            (0, express_validator_1.body)('strategy').optional().isString().withMessage('Strategy must be a string'),
            (0, express_validator_1.body)('timeframe').optional().isString().withMessage('Timeframe must be a string'),
            (0, express_validator_1.body)('parameters').optional().isObject().withMessage('Parameters must be an object')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        const updateData = {
            name: req.body.name,
            symbol: req.body.symbol,
            strategy: req.body.strategy,
            timeframe: req.body.timeframe,
            parameters: req.body.parameters
        };
        // Filter out undefined values
        Object.keys(updateData).forEach(key => {
            if (updateData[key] === undefined) {
                delete updateData[key];
            }
        });
        const updatedBot = await botService.updateBot(botId, userId, updateData);
        return res.status(200).json({
            success: true,
            message: 'Bot updated successfully',
            data: updatedBot
        });
    }
    catch (error) {
        // Handle specific errors
        if (error.message === 'Bot not found or access denied') {
            return res.status(404).json({
                success: false,
                message: 'Bot not found or access denied'
            });
        }
        if (error.message === 'Cannot update bot while it is active. Stop the bot first.') {
            return res.status(400).json({
                success: false,
                message: 'Cannot update bot while it is active. Stop the bot first.'
            });
        }
        console.error('Update bot error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while updating bot',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.updateBot = updateBot;
/**
 * Delete a trading bot
 * @route DELETE /api/bots/:id
 * @access Private
 */
const deleteBot = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        await botService.deleteBot(botId, userId);
        return res.status(200).json({
            success: true,
            message: 'Bot deleted successfully'
        });
    }
    catch (error) {
        // Handle specific errors
        if (error.message === 'Bot not found or access denied') {
            return res.status(404).json({
                success: false,
                message: 'Bot not found or access denied'
            });
        }
        console.error('Delete bot error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while deleting bot',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.deleteBot = deleteBot;
/**
 * Start a trading bot
 * @route POST /api/bots/:id/start
 * @access Private
 */
const startBot = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        await botService.startBot(botId, userId);
        return res.status(200).json({
            success: true,
            message: 'Bot started successfully'
        });
    }
    catch (error) {
        // Handle specific errors
        if (error.message === 'Bot not found or access denied') {
            return res.status(404).json({
                success: false,
                message: 'Bot not found or access denied'
            });
        }
        if (error.message === 'Bot is already running') {
            return res.status(400).json({
                success: false,
                message: 'Bot is already running'
            });
        }
        if (error.message === 'Bot has no risk settings. Please configure risk settings before starting.') {
            return res.status(400).json({
                success: false,
                message: 'Bot has no risk settings. Please configure risk settings before starting.'
            });
        }
        console.error('Start bot error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while starting bot',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.startBot = startBot;
/**
 * Stop a trading bot
 * @route POST /api/bots/:id/stop
 * @access Private
 */
const stopBot = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        await botService.stopBot(botId, userId);
        return res.status(200).json({
            success: true,
            message: 'Bot stopped successfully'
        });
    }
    catch (error) {
        // Handle specific errors
        if (error.message === 'Bot not found or access denied') {
            return res.status(404).json({
                success: false,
                message: 'Bot not found or access denied'
            });
        }
        if (error.message === 'Bot is not running') {
            return res.status(400).json({
                success: false,
                message: 'Bot is not running'
            });
        }
        console.error('Stop bot error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while stopping bot',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.stopBot = stopBot;
/**
 * Pause a trading bot
 * @route POST /api/bots/:id/pause
 * @access Private
 */
const pauseBot = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required'),
            (0, express_validator_1.body)('duration').optional().isInt().withMessage('Duration must be an integer')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        const duration = req.body.duration ? parseInt(req.body.duration) : undefined;
        await botService.pauseBot(botId, userId, duration);
        return res.status(200).json({
            success: true,
            message: `Bot paused successfully${duration ? ` for ${duration} seconds` : ''}`
        });
    }
    catch (error) {
        // Handle specific errors
        if (error.message === 'Bot not found or access denied') {
            return res.status(404).json({
                success: false,
                message: 'Bot not found or access denied'
            });
        }
        if (error.message === 'Bot is not running') {
            return res.status(400).json({
                success: false,
                message: 'Bot is not running'
            });
        }
        console.error('Pause bot error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while pausing bot',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.pauseBot = pauseBot;
/**
 * Run backtest for a bot
 * @route POST /api/bots/:id/backtest
 * @access Private
 */
const runBacktest = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required'),
            (0, express_validator_1.body)('symbol').notEmpty().withMessage('Symbol is required'),
            (0, express_validator_1.body)('timeframe').notEmpty().withMessage('Timeframe is required'),
            (0, express_validator_1.body)('startDate').isISO8601().withMessage('Valid start date is required'),
            (0, express_validator_1.body)('endDate').isISO8601().withMessage('Valid end date is required'),
            (0, express_validator_1.body)('initialCapital').isFloat({ min: 100 }).withMessage('Initial capital must be at least $100'),
            (0, express_validator_1.body)('leverage').isFloat({ min: 1, max: 100 }).withMessage('Leverage must be between 1 and 100'),
            (0, express_validator_1.body)('riskPerTrade').isFloat({ min: 0.1, max: 10 }).withMessage('Risk per trade must be between 0.1% and 10%'),
            (0, express_validator_1.body)('commission').isFloat({ min: 0, max: 1 }).withMessage('Commission must be between 0% and 1%')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        const backtestConfig = {
            symbol: req.body.symbol,
            timeframe: req.body.timeframe,
            startDate: new Date(req.body.startDate),
            endDate: new Date(req.body.endDate),
            initialCapital: req.body.initialCapital,
            leverage: req.body.leverage,
            riskPerTrade: req.body.riskPerTrade,
            commission: req.body.commission,
        };
        const result = await botService.runBacktest(botId, userId, backtestConfig);
        return res.status(200).json({
            success: true,
            message: 'Backtest completed successfully',
            data: result
        });
    }
    catch (error) {
        // Handle specific errors
        if (error.message === 'Bot not found or access denied') {
            return res.status(404).json({
                success: false,
                message: 'Bot not found or access denied'
            });
        }
        console.error('Run backtest error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while running backtest',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.runBacktest = runBacktest;
/**
 * Get backtest history for a bot
 * @route GET /api/bots/:id/backtests
 * @access Private
 */
const getBacktestHistory = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        const limit = parseInt(req.query.limit) || 10;
        const offset = parseInt(req.query.offset) || 0;
        const results = await botService.getBacktestHistory(botId, userId, limit, offset);
        return res.status(200).json({
            success: true,
            data: results
        });
    }
    catch (error) {
        console.error('Get backtest history error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while fetching backtest history',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.getBacktestHistory = getBacktestHistory;
/**
 * Get bot status
 * @route GET /api/bots/:id/status
 * @access Private
 */
const getBotStatus = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        const status = await botService.getBotStatus(botId, userId);
        return res.status(200).json({
            success: true,
            data: status
        });
    }
    catch (error) {
        // Handle specific errors
        if (error.message === 'Bot not found or access denied') {
            return res.status(404).json({
                success: false,
                message: 'Bot not found or access denied'
            });
        }
        console.error('Get bot status error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while fetching bot status',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.getBotStatus = getBotStatus;
/**
 * Configure risk settings for a bot
 * @route POST /api/bots/:id/risk
 * @access Private
 */
const configureBotRiskSettings = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required'),
            (0, express_validator_1.body)('name').optional().isString().withMessage('Name must be a string'),
            (0, express_validator_1.body)('description').optional().isString().withMessage('Description must be a string'),
            (0, express_validator_1.body)('positionSizingMethod').optional().isString().withMessage('Position sizing method must be a string'),
            (0, express_validator_1.body)('riskPercentage').optional().isFloat().withMessage('Risk percentage must be a number'),
            (0, express_validator_1.body)('maxPositionSize').optional().isFloat().withMessage('Max position size must be a number'),
            (0, express_validator_1.body)('stopLossType').optional().isString().withMessage('Stop loss type must be a string'),
            (0, express_validator_1.body)('stopLossValue').optional().isFloat().withMessage('Stop loss value must be a number'),
            (0, express_validator_1.body)('takeProfitType').optional().isString().withMessage('Take profit type must be a string'),
            (0, express_validator_1.body)('takeProfitValue').optional().isFloat().withMessage('Take profit value must be a number')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        const riskSettings = await botService.configureBotRiskSettings(botId, userId, req.body);
        return res.status(200).json({
            success: true,
            message: 'Risk settings configured successfully',
            data: riskSettings
        });
    }
    catch (error) {
        // Handle specific errors
        if (error.message === 'Bot not found or access denied') {
            return res.status(404).json({
                success: false,
                message: 'Bot not found or access denied'
            });
        }
        console.error('Configure risk settings error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while configuring risk settings',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.configureBotRiskSettings = configureBotRiskSettings;
/**
 * Update bot health status
 * @route POST /api/bots/:id/health
 * @access Private
 */
const updateBotHealth = async (req, res) => {
    try {
        // Validate request
        const validationRules = [
            (0, express_validator_1.param)('id').notEmpty().withMessage('Bot ID is required'),
            (0, express_validator_1.body)('health').isIn(['excellent', 'good', 'degraded', 'poor', 'critical', 'unknown']).withMessage('Valid health status is required'),
            (0, express_validator_1.body)('metrics').isObject().withMessage('Metrics must be an object')
        ];
        const validationErrors = await (0, validator_1.validateRequest)(req, validationRules);
        if (validationErrors) {
            return res.status(400).json({
                success: false,
                errors: validationErrors
            });
        }
        const userId = req.user?.id;
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User not authenticated'
            });
        }
        const botId = req.params.id;
        const healthData = {
            health: req.body.health,
            metrics: req.body.metrics,
            errors: req.body.errors,
            logs: req.body.logs
        };
        // Call the service function
        const success = await botService.updateBotHealth(botId, healthData);
        if (!success) {
            return res.status(400).json({
                success: false,
                message: 'Failed to update bot health'
            });
        }
        return res.status(200).json({
            success: true,
            message: 'Bot health updated successfully'
        });
    }
    catch (error) {
        console.error('Update bot health error:', error);
        return res.status(500).json({
            success: false,
            message: 'Server error while updating bot health',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.updateBotHealth = updateBotHealth;
//# sourceMappingURL=botController.js.map