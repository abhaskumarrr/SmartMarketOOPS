"use strict";
/**
 * Strategy Execution Service
 * Executes trading strategies based on signals and risk parameters
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.StrategyExecutionService = void 0;
const uuid_1 = require("uuid");
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
const logger_1 = require("../../utils/logger");
const signalGenerationService_1 = __importDefault(require("./signalGenerationService"));
const riskManagementService_1 = __importDefault(require("./riskManagementService"));
const circuitBreakerService_1 = __importDefault(require("./circuitBreakerService"));
const strategy_1 = require("../../types/strategy");
const signals_1 = require("../../types/signals");
// Create logger
const logger = (0, logger_1.createLogger)('StrategyExecutionService');
/**
 * Strategy Execution Service class
 * Provides methods to execute and manage trading strategies
 */
class StrategyExecutionService {
    /**
     * Creates a new Strategy Execution Service instance
     */
    constructor() {
        // Store active strategy executions
        this.activeExecutions = new Map();
        logger.info('Strategy Execution Service initialized');
        // Setup interval to process active strategies
        setInterval(() => this.processActiveStrategies(), 60000); // Every minute
    }
    /**
     * Create a new strategy
     * @param config - Strategy configuration
     * @returns Created strategy configuration
     */
    async createStrategy(config) {
        try {
            logger.info(`Creating new strategy: ${config.name}`);
            // Generate new ID
            const id = (0, uuid_1.v4)();
            // Create strategy in database
            const strategy = await prismaClient_1.default.tradingStrategy.create({
                data: {
                    id,
                    name: config.name,
                    description: config.description || '',
                    type: config.type,
                    timeHorizon: config.timeHorizon,
                    symbols: config.symbols || [],
                    entryRules: config.entryRules || [],
                    exitRules: config.exitRules || [],
                    positionSizing: config.positionSizing || { method: 'FIXED_FRACTIONAL', parameters: { riskPercentage: 1.0 } },
                    riskManagement: config.riskManagement || {
                        maxPositionSize: 0.1,
                        maxDrawdown: 10.0,
                        maxOpenPositions: 5,
                        useCircuitBreakers: true
                    },
                    indicators: config.indicators || [],
                    isActive: config.isActive !== undefined ? config.isActive : false,
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString()
                }
            });
            return strategy;
        }
        catch (error) {
            const logData = {
                strategyName: config.name,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error creating strategy: ${config.name}`, logData);
            throw error;
        }
    }
    /**
     * Get a strategy by ID
     * @param id - Strategy ID
     * @returns Strategy configuration
     */
    async getStrategy(id) {
        try {
            logger.debug(`Getting strategy: ${id}`);
            const strategy = await prismaClient_1.default.tradingStrategy.findUnique({
                where: { id }
            });
            if (!strategy) {
                throw new Error(`Strategy with ID ${id} not found`);
            }
            return strategy;
        }
        catch (error) {
            const logData = {
                strategyId: id,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error getting strategy: ${id}`, logData);
            throw error;
        }
    }
    /**
     * Update a strategy
     * @param id - Strategy ID
     * @param updates - Strategy updates
     * @returns Updated strategy configuration
     */
    async updateStrategy(id, updates) {
        try {
            logger.info(`Updating strategy: ${id}`);
            // Get existing strategy
            const existingStrategy = await this.getStrategy(id);
            // Update strategy in database
            const updatedStrategy = await prismaClient_1.default.tradingStrategy.update({
                where: { id },
                data: {
                    ...updates,
                    updatedAt: new Date().toISOString()
                }
            });
            return updatedStrategy;
        }
        catch (error) {
            const logData = {
                strategyId: id,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error updating strategy: ${id}`, logData);
            throw error;
        }
    }
    /**
     * Delete a strategy
     * @param id - Strategy ID
     * @returns Success status
     */
    async deleteStrategy(id) {
        try {
            logger.info(`Deleting strategy: ${id}`);
            // Check if strategy has active executions
            const activeExecution = await prismaClient_1.default.strategyExecution.findFirst({
                where: {
                    strategyId: id,
                    status: 'ACTIVE'
                }
            });
            if (activeExecution) {
                throw new Error(`Cannot delete strategy ${id} with active executions`);
            }
            // Delete strategy from database
            await prismaClient_1.default.tradingStrategy.delete({
                where: { id }
            });
            return true;
        }
        catch (error) {
            const logData = {
                strategyId: id,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error deleting strategy: ${id}`, logData);
            throw error;
        }
    }
    /**
     * Validate a strategy configuration
     * @param config - Strategy configuration to validate
     * @returns Validation result
     */
    validateStrategy(config) {
        try {
            logger.debug('Validating strategy configuration');
            const errors = [];
            const warnings = [];
            const suggestions = [];
            // Check required fields
            if (!config.name) {
                errors.push('Strategy name is required');
            }
            if (!config.type) {
                errors.push('Strategy type is required');
            }
            if (!config.timeHorizon) {
                errors.push('Strategy time horizon is required');
            }
            if (!config.symbols || config.symbols.length === 0) {
                errors.push('At least one trading symbol is required');
            }
            // Check entry and exit rules
            if (!config.entryRules || config.entryRules.length === 0) {
                errors.push('At least one entry rule is required');
            }
            if (!config.exitRules || config.exitRules.length === 0) {
                errors.push('At least one exit rule is required');
            }
            // Check rule configurations
            if (config.entryRules) {
                config.entryRules.forEach((rule, index) => {
                    if (!rule.name) {
                        errors.push(`Entry rule at index ${index} must have a name`);
                    }
                    if (!rule.type) {
                        errors.push(`Entry rule at index ${index} must have a type`);
                    }
                    if (!rule.parameters) {
                        errors.push(`Entry rule at index ${index} must have parameters`);
                    }
                });
            }
            if (config.exitRules) {
                config.exitRules.forEach((rule, index) => {
                    if (!rule.name) {
                        errors.push(`Exit rule at index ${index} must have a name`);
                    }
                    if (!rule.type) {
                        errors.push(`Exit rule at index ${index} must have a type`);
                    }
                    if (!rule.parameters) {
                        errors.push(`Exit rule at index ${index} must have parameters`);
                    }
                    // Check if we have a stop loss rule
                    if (rule.type === strategy_1.ExitRuleType.STOP_LOSS) {
                        const hasStopLoss = true;
                    }
                });
                // Ensure we have a stop loss rule
                const hasStopLoss = config.exitRules.some(rule => rule.type === strategy_1.ExitRuleType.STOP_LOSS);
                if (!hasStopLoss) {
                    warnings.push('Strategy has no stop loss rule. This is risky.');
                    suggestions.push('Add a stop loss rule to protect capital.');
                }
            }
            // Check risk management
            if (!config.riskManagement) {
                warnings.push('No risk management configured');
                suggestions.push('Configure risk management settings to protect capital');
            }
            else {
                if (config.riskManagement.maxPositionSize > 0.5) {
                    warnings.push('Maximum position size is very high');
                    suggestions.push('Consider reducing maximum position size to protect capital');
                }
                if (config.riskManagement.maxDrawdown > 20) {
                    warnings.push('Maximum drawdown is very high');
                    suggestions.push('Consider reducing maximum drawdown to protect capital');
                }
            }
            return {
                isValid: errors.length === 0,
                errors,
                warnings,
                suggestions
            };
        }
        catch (error) {
            logger.error('Error validating strategy', { error: error instanceof Error ? error.message : String(error) });
            return {
                isValid: false,
                errors: [error instanceof Error ? error.message : String(error)],
                warnings: [],
                suggestions: []
            };
        }
    }
    /**
     * Start executing a strategy
     * @param strategyId - Strategy ID
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Execution instance
     */
    async startExecution(strategyId, userId, botId) {
        try {
            logger.info(`Starting execution of strategy ${strategyId} for user ${userId}`);
            // Check if trading is allowed
            const tradingAllowed = await circuitBreakerService_1.default.isTradingAllowed(userId, botId);
            if (!tradingAllowed.allowed) {
                throw new Error(`Trading is not allowed: ${tradingAllowed.reason}`);
            }
            // Get strategy configuration
            const strategy = await this.getStrategy(strategyId);
            // Create execution instance
            const executionId = (0, uuid_1.v4)();
            const execution = {
                id: executionId,
                strategyId,
                userId,
                botId,
                status: strategy_1.StrategyExecutionStatus.ACTIVE,
                startedAt: new Date().toISOString(),
                currentPositions: [],
                historicalPositions: [],
                performance: {
                    totalPnL: 0,
                    winRate: 0,
                    totalTrades: 0,
                    successfulTrades: 0,
                    failedTrades: 0,
                    averageHoldingTime: 0,
                    maxDrawdown: 0
                },
                logs: [
                    {
                        timestamp: new Date().toISOString(),
                        message: `Started execution of strategy ${strategy.name}`,
                        level: 'info'
                    }
                ],
                errors: []
            };
            // Store in database
            await prismaClient_1.default.strategyExecution.create({
                data: execution
            });
            // Store in memory
            this.activeExecutions.set(executionId, execution);
            // Return execution instance
            return execution;
        }
        catch (error) {
            const logData = {
                strategyId,
                userId,
                botId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error starting execution of strategy ${strategyId}`, logData);
            throw error;
        }
    }
    /**
     * Stop a strategy execution
     * @param executionId - Execution ID
     * @returns Updated execution instance
     */
    async stopExecution(executionId) {
        try {
            logger.info(`Stopping execution ${executionId}`);
            // Get execution from database
            const execution = await prismaClient_1.default.strategyExecution.findUnique({
                where: { id: executionId }
            });
            if (!execution) {
                throw new Error(`Execution with ID ${executionId} not found`);
            }
            // Update status
            const updatedExecution = await prismaClient_1.default.strategyExecution.update({
                where: { id: executionId },
                data: {
                    status: strategy_1.StrategyExecutionStatus.STOPPED,
                    stoppedAt: new Date().toISOString(),
                    logs: [
                        ...(execution.logs || []),
                        {
                            timestamp: new Date().toISOString(),
                            message: 'Execution stopped',
                            level: 'info'
                        }
                    ]
                }
            });
            // Remove from active executions
            this.activeExecutions.delete(executionId);
            return updatedExecution;
        }
        catch (error) {
            const logData = {
                executionId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error stopping execution ${executionId}`, logData);
            throw error;
        }
    }
    /**
     * Process a new trading signal with strategy rules
     * @param signal - Trading signal
     * @param executionId - Execution ID
     * @returns Execution result
     */
    async processSignal(signal, executionId) {
        try {
            logger.info(`Processing signal ${signal.id} for execution ${executionId}`);
            // Get execution
            const execution = await prismaClient_1.default.strategyExecution.findUnique({
                where: { id: executionId }
            });
            if (!execution) {
                throw new Error(`Execution with ID ${executionId} not found`);
            }
            // Get strategy configuration
            const strategy = await this.getStrategy(execution.strategyId);
            // Process entry rules
            const entryRuleResults = await this._processEntryRules(signal, strategy.entryRules);
            // Process exit rules if we have an open position
            const exitRuleResults = await this._processExitRules(signal, strategy.exitRules, execution);
            // Determine action based on rule results
            let action = 'NONE';
            // Check if we have an open position for this symbol
            const hasOpenPosition = execution.currentPositions.some(async (positionId) => {
                const position = await prismaClient_1.default.position.findUnique({
                    where: { id: positionId }
                });
                return position && position.symbol === signal.symbol;
            });
            // Determine action based on signal type and position status
            if (signal.type === signals_1.SignalType.ENTRY && !hasOpenPosition) {
                // Check if all required entry rules are satisfied
                const allRequiredEntrySatisfied = strategy.entryRules
                    .filter(rule => rule.isRequired)
                    .every(rule => {
                    const result = entryRuleResults.find(r => r.ruleId === rule.id);
                    return result && result.satisfied;
                });
                if (allRequiredEntrySatisfied) {
                    action = 'ENTRY';
                }
            }
            else if (signal.type === signals_1.SignalType.EXIT && hasOpenPosition) {
                // Check if any required exit rule is satisfied
                const anyRequiredExitSatisfied = strategy.exitRules
                    .filter(rule => rule.isRequired)
                    .some(rule => {
                    const result = exitRuleResults.find(r => r.ruleId === rule.id);
                    return result && result.satisfied;
                });
                if (anyRequiredExitSatisfied) {
                    action = 'EXIT';
                }
            }
            else if (signal.type === signals_1.SignalType.INCREASE && hasOpenPosition) {
                action = 'INCREASE';
            }
            else if (signal.type === signals_1.SignalType.DECREASE && hasOpenPosition) {
                action = 'DECREASE';
            }
            else if (signal.type === signals_1.SignalType.HOLD) {
                action = 'HOLD';
            }
            // Calculate position size if action is ENTRY or INCREASE
            let positionSize, stopLossPrice, takeProfitPrice;
            if (action === 'ENTRY' || action === 'INCREASE') {
                // Calculate position size using risk management service
                const positionSizeResult = await riskManagementService_1.default.calculatePositionSize({
                    userId: execution.userId,
                    botId: execution.botId,
                    symbol: signal.symbol,
                    direction: signal.direction === signals_1.SignalDirection.LONG ? 'long' : 'short',
                    entryPrice: signal.price,
                    stopLossPrice: signal.stopLoss,
                    confidence: signal.confidenceScore
                });
                positionSize = positionSizeResult.positionSize;
                stopLossPrice = signal.stopLoss;
                // Calculate take profit based on risk-reward ratio
                if (stopLossPrice && signal.direction === signals_1.SignalDirection.LONG) {
                    const riskAmount = signal.price - stopLossPrice;
                    takeProfitPrice = signal.price + (riskAmount * strategy.riskManagement.targetRiskRewardRatio || 2);
                }
                else if (stopLossPrice && signal.direction === signals_1.SignalDirection.SHORT) {
                    const riskAmount = stopLossPrice - signal.price;
                    takeProfitPrice = signal.price - (riskAmount * strategy.riskManagement.targetRiskRewardRatio || 2);
                }
            }
            // Create execution result
            const result = {
                executionId,
                signal,
                entryRuleResults,
                exitRuleResults,
                action,
                positionSize,
                entryPrice: signal.price,
                stopLossPrice,
                takeProfitPrice,
                confidence: signal.confidenceScore,
                notes: `Action determined: ${action}`,
                timestamp: new Date().toISOString()
            };
            // Store result in database
            await prismaClient_1.default.strategyExecutionResult.create({
                data: result
            });
            return result;
        }
        catch (error) {
            const logData = {
                signalId: signal.id,
                executionId,
                error: error instanceof Error ? error.message : String(error)
            };
            logger.error(`Error processing signal ${signal.id} for execution ${executionId}`, logData);
            throw error;
        }
    }
    /**
     * Process active strategies
     * @private
     */
    async processActiveStrategies() {
        try {
            // Get all active executions from database
            const activeExecutions = await prismaClient_1.default.strategyExecution.findMany({
                where: { status: strategy_1.StrategyExecutionStatus.ACTIVE }
            });
            logger.debug(`Processing ${activeExecutions.length} active strategies`);
            // Process each execution
            for (const execution of activeExecutions) {
                try {
                    // Check if trading is allowed
                    const tradingAllowed = await circuitBreakerService_1.default.isTradingAllowed(execution.userId, execution.botId);
                    if (!tradingAllowed.allowed) {
                        // Update execution status to PAUSED
                        await prismaClient_1.default.strategyExecution.update({
                            where: { id: execution.id },
                            data: {
                                status: strategy_1.StrategyExecutionStatus.PAUSED,
                                logs: [
                                    ...(execution.logs || []),
                                    {
                                        timestamp: new Date().toISOString(),
                                        message: `Execution paused: ${tradingAllowed.reason}`,
                                        level: 'warning'
                                    }
                                ]
                            }
                        });
                        continue;
                    }
                    // Get strategy configuration
                    const strategy = await this.getStrategy(execution.strategyId);
                    // Check for new signals for each symbol
                    for (const symbol of strategy.symbols) {
                        const latestSignal = await signalGenerationService_1.default.getLatestSignal(symbol);
                        if (latestSignal) {
                            // Process signal
                            await this.processSignal(latestSignal, execution.id);
                        }
                    }
                    // Update last executed timestamp
                    await prismaClient_1.default.strategyExecution.update({
                        where: { id: execution.id },
                        data: {
                            lastExecutedAt: new Date().toISOString()
                        }
                    });
                }
                catch (error) {
                    const logData = {
                        executionId: execution.id,
                        error: error instanceof Error ? error.message : String(error)
                    };
                    logger.error(`Error processing execution ${execution.id}`, logData);
                    // Log error in execution
                    await prismaClient_1.default.strategyExecution.update({
                        where: { id: execution.id },
                        data: {
                            errors: [
                                ...(execution.errors || []),
                                {
                                    timestamp: new Date().toISOString(),
                                    message: error instanceof Error ? error.message : String(error)
                                }
                            ]
                        }
                    });
                }
            }
        }
        catch (error) {
            logger.error('Error processing active strategies', {
                error: error instanceof Error ? error.message : String(error)
            });
        }
    }
    /**
     * Process entry rules for a signal
     * @private
     * @param signal - Trading signal
     * @param rules - Entry rules
     * @returns Rule results
     */
    async _processEntryRules(signal, rules) {
        const results = [];
        // Process each rule
        for (const rule of rules) {
            try {
                let satisfied = false;
                let details = {};
                // Process based on rule type
                switch (rule.type) {
                    case strategy_1.EntryRuleType.SIGNAL_BASED:
                        // Check if signal matches direction and strength
                        satisfied = ((!rule.direction || signal.direction === rule.direction) &&
                            signal.type === signals_1.SignalType.ENTRY &&
                            signal.confidenceScore >= (rule.parameters.minConfidence || 0));
                        details = {
                            signalType: signal.type,
                            signalDirection: signal.direction,
                            confidenceScore: signal.confidenceScore,
                            minRequiredConfidence: rule.parameters.minConfidence || 0
                        };
                        break;
                    case strategy_1.EntryRuleType.PRICE_BREAKOUT:
                        // Check for price breakout
                        const breakoutLevel = rule.parameters.level;
                        const isBreakout = rule.direction === signals_1.SignalDirection.LONG
                            ? signal.price > breakoutLevel
                            : signal.price < breakoutLevel;
                        satisfied = isBreakout;
                        details = {
                            currentPrice: signal.price,
                            breakoutLevel,
                            direction: rule.direction
                        };
                        break;
                    case strategy_1.EntryRuleType.ML_PREDICTION:
                        // ML prediction is already incorporated in the signal
                        satisfied = signal.confidenceScore >= (rule.parameters.threshold || 70);
                        details = {
                            confidenceScore: signal.confidenceScore,
                            threshold: rule.parameters.threshold || 70
                        };
                        break;
                    default:
                        satisfied = false;
                        details = { error: `Unsupported entry rule type: ${rule.type}` };
                }
                results.push({
                    ruleId: rule.id,
                    satisfied,
                    details
                });
            }
            catch (error) {
                logger.error(`Error processing entry rule ${rule.id}`, {
                    error: error instanceof Error ? error.message : String(error),
                    rule
                });
                results.push({
                    ruleId: rule.id,
                    satisfied: false,
                    details: { error: error instanceof Error ? error.message : String(error) }
                });
            }
        }
        return results;
    }
    /**
     * Process exit rules for a signal
     * @private
     * @param signal - Trading signal
     * @param rules - Exit rules
     * @param execution - Strategy execution
     * @returns Rule results
     */
    async _processExitRules(signal, rules, execution) {
        const results = [];
        // Get open positions for this symbol
        const openPositions = await Promise.all(execution.currentPositions.map(async (positionId) => {
            return await prismaClient_1.default.position.findUnique({
                where: { id: positionId }
            });
        }));
        const positionsForSymbol = openPositions.filter(p => p && p.symbol === signal.symbol);
        // Process each rule
        for (const rule of rules) {
            try {
                let satisfied = false;
                let details = {};
                // Process based on rule type
                switch (rule.type) {
                    case strategy_1.ExitRuleType.SIGNAL_BASED:
                        // Check if signal is an exit signal
                        satisfied = signal.type === signals_1.SignalType.EXIT &&
                            signal.confidenceScore >= (rule.parameters.minConfidence || 0);
                        details = {
                            signalType: signal.type,
                            confidenceScore: signal.confidenceScore,
                            minRequiredConfidence: rule.parameters.minConfidence || 0
                        };
                        break;
                    case strategy_1.ExitRuleType.STOP_LOSS:
                        // Check if current price has hit stop loss for any position
                        for (const position of positionsForSymbol) {
                            if (!position || !position.stopLossPrice)
                                continue;
                            const isStopLossHit = position.side.toLowerCase() === 'long'
                                ? signal.price <= position.stopLossPrice
                                : signal.price >= position.stopLossPrice;
                            if (isStopLossHit) {
                                satisfied = true;
                                details = {
                                    positionId: position.id,
                                    currentPrice: signal.price,
                                    stopLossPrice: position.stopLossPrice,
                                    side: position.side
                                };
                                break;
                            }
                        }
                        break;
                    case strategy_1.ExitRuleType.TAKE_PROFIT:
                        // Check if current price has hit take profit for any position
                        for (const position of positionsForSymbol) {
                            if (!position || !position.takeProfitPrice)
                                continue;
                            const isTakeProfitHit = position.side.toLowerCase() === 'long'
                                ? signal.price >= position.takeProfitPrice
                                : signal.price <= position.takeProfitPrice;
                            if (isTakeProfitHit) {
                                satisfied = true;
                                details = {
                                    positionId: position.id,
                                    currentPrice: signal.price,
                                    takeProfitPrice: position.takeProfitPrice,
                                    side: position.side
                                };
                                break;
                            }
                        }
                        break;
                    case strategy_1.ExitRuleType.TIME_BASED:
                        // Check if position has been open for longer than max duration
                        const maxDuration = rule.parameters.maxDurationHours * 60 * 60 * 1000; // Convert hours to milliseconds
                        for (const position of positionsForSymbol) {
                            if (!position)
                                continue;
                            const openDuration = Date.now() - new Date(position.openedAt).getTime();
                            if (openDuration > maxDuration) {
                                satisfied = true;
                                details = {
                                    positionId: position.id,
                                    openDuration: openDuration / (60 * 60 * 1000), // Convert to hours
                                    maxDuration: rule.parameters.maxDurationHours,
                                    openedAt: position.openedAt
                                };
                                break;
                            }
                        }
                        break;
                    default:
                        satisfied = false;
                        details = { error: `Unsupported exit rule type: ${rule.type}` };
                }
                results.push({
                    ruleId: rule.id,
                    satisfied,
                    details
                });
            }
            catch (error) {
                logger.error(`Error processing exit rule ${rule.id}`, {
                    error: error instanceof Error ? error.message : String(error),
                    rule
                });
                results.push({
                    ruleId: rule.id,
                    satisfied: false,
                    details: { error: error instanceof Error ? error.message : String(error) }
                });
            }
        }
        return results;
    }
}
exports.StrategyExecutionService = StrategyExecutionService;
// Create singleton instance
const strategyExecutionService = new StrategyExecutionService();
exports.default = strategyExecutionService;
//# sourceMappingURL=strategyExecutionService.js.map