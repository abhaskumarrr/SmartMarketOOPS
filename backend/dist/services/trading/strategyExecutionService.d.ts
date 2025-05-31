/**
 * Strategy Execution Service
 * Executes trading strategies based on signals and risk parameters
 */
import { StrategyConfig, StrategyExecution, StrategyExecutionResult, StrategyValidationResult } from '../../types/strategy';
import { TradingSignal } from '../../types/signals';
/**
 * Strategy Execution Service class
 * Provides methods to execute and manage trading strategies
 */
export declare class StrategyExecutionService {
    private activeExecutions;
    /**
     * Creates a new Strategy Execution Service instance
     */
    constructor();
    /**
     * Create a new strategy
     * @param config - Strategy configuration
     * @returns Created strategy configuration
     */
    createStrategy(config: Partial<StrategyConfig>): Promise<StrategyConfig>;
    /**
     * Get a strategy by ID
     * @param id - Strategy ID
     * @returns Strategy configuration
     */
    getStrategy(id: string): Promise<StrategyConfig>;
    /**
     * Update a strategy
     * @param id - Strategy ID
     * @param updates - Strategy updates
     * @returns Updated strategy configuration
     */
    updateStrategy(id: string, updates: Partial<StrategyConfig>): Promise<StrategyConfig>;
    /**
     * Delete a strategy
     * @param id - Strategy ID
     * @returns Success status
     */
    deleteStrategy(id: string): Promise<boolean>;
    /**
     * Validate a strategy configuration
     * @param config - Strategy configuration to validate
     * @returns Validation result
     */
    validateStrategy(config: Partial<StrategyConfig>): StrategyValidationResult;
    /**
     * Start executing a strategy
     * @param strategyId - Strategy ID
     * @param userId - User ID
     * @param botId - Optional Bot ID
     * @returns Execution instance
     */
    startExecution(strategyId: string, userId: string, botId?: string): Promise<StrategyExecution>;
    /**
     * Stop a strategy execution
     * @param executionId - Execution ID
     * @returns Updated execution instance
     */
    stopExecution(executionId: string): Promise<StrategyExecution>;
    /**
     * Process a new trading signal with strategy rules
     * @param signal - Trading signal
     * @param executionId - Execution ID
     * @returns Execution result
     */
    processSignal(signal: TradingSignal, executionId: string): Promise<StrategyExecutionResult>;
    /**
     * Process active strategies
     * @private
     */
    private processActiveStrategies;
    /**
     * Process entry rules for a signal
     * @private
     * @param signal - Trading signal
     * @param rules - Entry rules
     * @returns Rule results
     */
    private _processEntryRules;
    /**
     * Process exit rules for a signal
     * @private
     * @param signal - Trading signal
     * @param rules - Exit rules
     * @param execution - Strategy execution
     * @returns Rule results
     */
    private _processExitRules;
}
declare const strategyExecutionService: StrategyExecutionService;
export default strategyExecutionService;
