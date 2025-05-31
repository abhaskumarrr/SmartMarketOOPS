"use strict";
/**
 * Strategy Controller
 * Handles HTTP requests related to trading strategies
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getExecutionResults = exports.getExecutionById = exports.getUserExecutions = exports.stopStrategyExecution = exports.startStrategyExecution = exports.validateStrategy = exports.deleteStrategy = exports.updateStrategy = exports.createStrategy = exports.getStrategyById = exports.getAllStrategies = void 0;
const logger_1 = require("../utils/logger");
const strategyExecutionService_1 = __importDefault(require("../services/trading/strategyExecutionService"));
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
// Create logger
const logger = (0, logger_1.createLogger)('StrategyController');
/**
 * Get all strategies
 * @route GET /api/strategies
 */
const getAllStrategies = async (req, res) => {
    try {
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get strategies from database
        const strategies = await prismaClient_1.default.tradingStrategy.findMany({
            where: {
                OR: [
                    { userId },
                    { isPublic: true }
                ]
            }
        });
        res.status(200).json(strategies);
    }
    catch (error) {
        logger.error('Error getting strategies', {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get strategies',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getAllStrategies = getAllStrategies;
/**
 * Get a strategy by ID
 * @route GET /api/strategies/:id
 */
const getStrategyById = async (req, res) => {
    try {
        const { id } = req.params;
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get strategy
        const strategy = await strategyExecutionService_1.default.getStrategy(id);
        // Check if user has access to strategy
        if (strategy.userId !== userId && !strategy.isPublic) {
            res.status(403).json({ error: 'Forbidden' });
            return;
        }
        res.status(200).json(strategy);
    }
    catch (error) {
        logger.error(`Error getting strategy ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        if (error instanceof Error && error.message.includes('not found')) {
            res.status(404).json({
                error: 'Strategy not found',
                details: error.message
            });
        }
        else {
            res.status(500).json({
                error: 'Failed to get strategy',
                details: error instanceof Error ? error.message : String(error)
            });
        }
    }
};
exports.getStrategyById = getStrategyById;
/**
 * Create a new strategy
 * @route POST /api/strategies
 */
const createStrategy = async (req, res) => {
    try {
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        const strategyData = req.body;
        // Add user ID to strategy data
        strategyData.userId = userId;
        // Validate strategy
        const validationResult = strategyExecutionService_1.default.validateStrategy(strategyData);
        if (!validationResult.isValid) {
            res.status(400).json({
                error: 'Invalid strategy configuration',
                validation: validationResult
            });
            return;
        }
        // Create strategy
        const strategy = await strategyExecutionService_1.default.createStrategy(strategyData);
        res.status(201).json(strategy);
    }
    catch (error) {
        logger.error('Error creating strategy', {
            error: error instanceof Error ? error.message : String(error),
            body: req.body
        });
        res.status(500).json({
            error: 'Failed to create strategy',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.createStrategy = createStrategy;
/**
 * Update a strategy
 * @route PUT /api/strategies/:id
 */
const updateStrategy = async (req, res) => {
    try {
        const { id } = req.params;
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get existing strategy
        const existingStrategy = await strategyExecutionService_1.default.getStrategy(id);
        // Check if user has access to strategy
        if (existingStrategy.userId !== userId) {
            res.status(403).json({ error: 'Forbidden' });
            return;
        }
        const updateData = req.body;
        // Validate strategy
        const validationResult = strategyExecutionService_1.default.validateStrategy({
            ...existingStrategy,
            ...updateData
        });
        if (!validationResult.isValid) {
            res.status(400).json({
                error: 'Invalid strategy configuration',
                validation: validationResult
            });
            return;
        }
        // Update strategy
        const strategy = await strategyExecutionService_1.default.updateStrategy(id, updateData);
        res.status(200).json(strategy);
    }
    catch (error) {
        logger.error(`Error updating strategy ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error),
            body: req.body
        });
        if (error instanceof Error && error.message.includes('not found')) {
            res.status(404).json({
                error: 'Strategy not found',
                details: error.message
            });
        }
        else {
            res.status(500).json({
                error: 'Failed to update strategy',
                details: error instanceof Error ? error.message : String(error)
            });
        }
    }
};
exports.updateStrategy = updateStrategy;
/**
 * Delete a strategy
 * @route DELETE /api/strategies/:id
 */
const deleteStrategy = async (req, res) => {
    try {
        const { id } = req.params;
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get existing strategy
        const existingStrategy = await strategyExecutionService_1.default.getStrategy(id);
        // Check if user has access to strategy
        if (existingStrategy.userId !== userId) {
            res.status(403).json({ error: 'Forbidden' });
            return;
        }
        // Delete strategy
        await strategyExecutionService_1.default.deleteStrategy(id);
        res.status(204).send();
    }
    catch (error) {
        logger.error(`Error deleting strategy ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        if (error instanceof Error && error.message.includes('not found')) {
            res.status(404).json({
                error: 'Strategy not found',
                details: error.message
            });
        }
        else {
            res.status(500).json({
                error: 'Failed to delete strategy',
                details: error instanceof Error ? error.message : String(error)
            });
        }
    }
};
exports.deleteStrategy = deleteStrategy;
/**
 * Validate a strategy configuration
 * @route POST /api/strategies/validate
 */
const validateStrategy = async (req, res) => {
    try {
        const strategyData = req.body;
        // Validate strategy
        const validationResult = strategyExecutionService_1.default.validateStrategy(strategyData);
        res.status(200).json(validationResult);
    }
    catch (error) {
        logger.error('Error validating strategy', {
            error: error instanceof Error ? error.message : String(error),
            body: req.body
        });
        res.status(500).json({
            error: 'Failed to validate strategy',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.validateStrategy = validateStrategy;
/**
 * Start executing a strategy
 * @route POST /api/strategies/:id/execute
 */
const startStrategyExecution = async (req, res) => {
    try {
        const { id } = req.params;
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get botId from request body if provided
        const { botId } = req.body;
        // Get existing strategy
        const existingStrategy = await strategyExecutionService_1.default.getStrategy(id);
        // Check if user has access to strategy
        if (existingStrategy.userId !== userId) {
            res.status(403).json({ error: 'Forbidden' });
            return;
        }
        // Start execution
        const execution = await strategyExecutionService_1.default.startExecution(id, userId, botId);
        res.status(201).json(execution);
    }
    catch (error) {
        logger.error(`Error starting execution of strategy ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error),
            body: req.body
        });
        res.status(500).json({
            error: 'Failed to start strategy execution',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.startStrategyExecution = startStrategyExecution;
/**
 * Stop executing a strategy
 * @route POST /api/executions/:id/stop
 */
const stopStrategyExecution = async (req, res) => {
    try {
        const { id } = req.params;
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get execution
        const execution = await prismaClient_1.default.strategyExecution.findUnique({
            where: { id }
        });
        if (!execution) {
            res.status(404).json({ error: 'Execution not found' });
            return;
        }
        // Check if user has access to execution
        if (execution.userId !== userId) {
            res.status(403).json({ error: 'Forbidden' });
            return;
        }
        // Stop execution
        const updatedExecution = await strategyExecutionService_1.default.stopExecution(id);
        res.status(200).json(updatedExecution);
    }
    catch (error) {
        logger.error(`Error stopping execution ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        if (error instanceof Error && error.message.includes('not found')) {
            res.status(404).json({
                error: 'Execution not found',
                details: error.message
            });
        }
        else {
            res.status(500).json({
                error: 'Failed to stop strategy execution',
                details: error instanceof Error ? error.message : String(error)
            });
        }
    }
};
exports.stopStrategyExecution = stopStrategyExecution;
/**
 * Get all executions for a user
 * @route GET /api/executions
 */
const getUserExecutions = async (req, res) => {
    try {
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get executions from database
        const executions = await prismaClient_1.default.strategyExecution.findMany({
            where: { userId }
        });
        res.status(200).json(executions);
    }
    catch (error) {
        logger.error('Error getting executions', {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get executions',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getUserExecutions = getUserExecutions;
/**
 * Get execution by ID
 * @route GET /api/executions/:id
 */
const getExecutionById = async (req, res) => {
    try {
        const { id } = req.params;
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get execution
        const execution = await prismaClient_1.default.strategyExecution.findUnique({
            where: { id }
        });
        if (!execution) {
            res.status(404).json({ error: 'Execution not found' });
            return;
        }
        // Check if user has access to execution
        if (execution.userId !== userId) {
            res.status(403).json({ error: 'Forbidden' });
            return;
        }
        res.status(200).json(execution);
    }
    catch (error) {
        logger.error(`Error getting execution ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get execution',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getExecutionById = getExecutionById;
/**
 * Get execution results
 * @route GET /api/executions/:id/results
 */
const getExecutionResults = async (req, res) => {
    try {
        const { id } = req.params;
        const userId = req.user?.id;
        if (!userId) {
            res.status(401).json({ error: 'Unauthorized' });
            return;
        }
        // Get execution
        const execution = await prismaClient_1.default.strategyExecution.findUnique({
            where: { id }
        });
        if (!execution) {
            res.status(404).json({ error: 'Execution not found' });
            return;
        }
        // Check if user has access to execution
        if (execution.userId !== userId) {
            res.status(403).json({ error: 'Forbidden' });
            return;
        }
        // Get execution results
        const results = await prismaClient_1.default.strategyExecutionResult.findMany({
            where: { executionId: id }
        });
        res.status(200).json(results);
    }
    catch (error) {
        logger.error(`Error getting execution results ${req.params.id}`, {
            error: error instanceof Error ? error.message : String(error)
        });
        res.status(500).json({
            error: 'Failed to get execution results',
            details: error instanceof Error ? error.message : String(error)
        });
    }
};
exports.getExecutionResults = getExecutionResults;
//# sourceMappingURL=strategyController.js.map