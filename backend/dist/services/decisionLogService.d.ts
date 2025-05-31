/**
 * Decision Log Service
 * Logs important decisions made by the system or users
 * Used for auditing, debugging, and analyzing trading strategy behavior
 */
import { DecisionLog, IDecisionLogService, DecisionLogQueryParams, CreateDecisionLogRequest } from '../types/auditLog';
/**
 * Decision importance levels
 */
export type DecisionImportance = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
/**
 * Decision source types
 */
export type DecisionSource = 'System' | 'User' | 'Strategy' | 'ML' | 'RiskManagement';
/**
 * Decision action types
 */
export type DecisionActionType = 'OrderExecution' | 'RiskAdjustment' | 'StrategySwitch' | 'PositionClose' | 'BotControl' | 'MarketAnalysis' | 'SystemAlert';
/**
 * Decision log entry
 */
export interface DecisionLogEntry {
    source: DecisionSource;
    actionType: DecisionActionType;
    decision: string;
    reasonDetails?: string;
    userId?: string;
    botId?: string;
    strategyId?: string;
    symbol?: string;
    orderId?: string;
    positionId?: string;
    importance: DecisionImportance;
    metadata?: Record<string, any>;
    tags?: string[];
}
/**
 * Decision Log Service class
 * Handles logging of trading decisions and actions
 */
export declare class DecisionLogService implements IDecisionLogService {
    /**
     * Create a new decision log
     * @param data - Decision log creation request
     * @returns Created decision log
     */
    createDecisionLog(data: CreateDecisionLogRequest): Promise<DecisionLog>;
    /**
     * Get a decision log by ID
     * @param id - Decision log ID
     * @returns Decision log
     */
    getDecisionLog(id: string): Promise<DecisionLog>;
    /**
     * Update a decision log
     * @param id - Decision log ID
     * @param data - Updated decision log data
     * @returns Updated decision log
     */
    updateDecisionLog(id: string, data: Partial<DecisionLog>): Promise<DecisionLog>;
    /**
     * Delete a decision log
     * @param id - Decision log ID
     * @returns Whether the deletion was successful
     */
    deleteDecisionLog(id: string): Promise<boolean>;
    /**
     * Query decision logs based on parameters
     * @param params - Query parameters
     * @returns Matching decision logs
     */
    queryDecisionLogs(params: DecisionLogQueryParams): Promise<DecisionLog[]>;
    /**
     * Count decision logs based on parameters
     * @param params - Query parameters
     * @returns Number of matching decision logs
     */
    countDecisionLogs(params: DecisionLogQueryParams): Promise<number>;
    /**
     * Map a decision log from database to response format
     * @param dbLog - Decision log from database
     * @returns Mapped decision log
     */
    private mapDecisionLogFromDb;
}
export declare const decisionLogService: DecisionLogService;
export default decisionLogService;
/**
 * Standalone function to create a decision log
 * Wrapper around the decisionLogService.createDecisionLog method
 * @param data - Decision log entry data
 * @returns Promise resolving to the created decision log
 */
export declare const createDecisionLog: (data: DecisionLogEntry) => Promise<DecisionLog>;
