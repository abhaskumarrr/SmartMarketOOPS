/**
 * Strategy Controller
 * Handles HTTP requests related to trading strategies
 */
import { Response } from 'express';
import { AuthenticatedRequest } from '../middleware/authMiddleware';
/**
 * Get all strategies
 * @route GET /api/strategies
 */
export declare const getAllStrategies: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get a strategy by ID
 * @route GET /api/strategies/:id
 */
export declare const getStrategyById: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Create a new strategy
 * @route POST /api/strategies
 */
export declare const createStrategy: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Update a strategy
 * @route PUT /api/strategies/:id
 */
export declare const updateStrategy: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Delete a strategy
 * @route DELETE /api/strategies/:id
 */
export declare const deleteStrategy: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Validate a strategy configuration
 * @route POST /api/strategies/validate
 */
export declare const validateStrategy: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Start executing a strategy
 * @route POST /api/strategies/:id/execute
 */
export declare const startStrategyExecution: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Stop executing a strategy
 * @route POST /api/executions/:id/stop
 */
export declare const stopStrategyExecution: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get all executions for a user
 * @route GET /api/executions
 */
export declare const getUserExecutions: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get execution by ID
 * @route GET /api/executions/:id
 */
export declare const getExecutionById: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get execution results
 * @route GET /api/executions/:id/results
 */
export declare const getExecutionResults: (req: AuthenticatedRequest, res: Response) => Promise<void>;
