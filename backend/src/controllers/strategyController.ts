/**
 * Strategy Controller
 * Handles HTTP requests related to trading strategies
 */

import { Response } from 'express';
import { createLogger } from '../utils/logger';
import strategyExecutionService from '../services/trading/strategyExecutionService';
import {
  StrategyConfig,
  StrategyExecution,
  StrategyExecutionResult,
  StrategyValidationResult
} from '../types/strategy';
import { AuthenticatedRequest } from '../middleware/authMiddleware';
import prisma from '../utils/prismaClient';

// Create logger
const logger = createLogger('StrategyController');

/**
 * Get all strategies
 * @route GET /api/strategies
 */
export const getAllStrategies = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    // Get strategies from database
    const strategies = await prisma.tradingStrategy.findMany({
      where: {
        OR: [
          { userId },
          { isPublic: true }
        ]
      }
    });
    
    res.status(200).json(strategies);
  } catch (error) {
    logger.error('Error getting strategies', {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get strategies',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get a strategy by ID
 * @route GET /api/strategies/:id
 */
export const getStrategyById = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    // Get strategy
    const strategy = await strategyExecutionService.getStrategy(id);
    
    // Check if user has access to strategy
    if (strategy.userId !== userId && !strategy.isPublic) {
      res.status(403).json({ error: 'Forbidden' });
      return;
    }
    
    res.status(200).json(strategy);
  } catch (error) {
    logger.error(`Error getting strategy ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    if (error instanceof Error && error.message.includes('not found')) {
      res.status(404).json({
        error: 'Strategy not found',
        details: error.message
      });
    } else {
      res.status(500).json({
        error: 'Failed to get strategy',
        details: error instanceof Error ? error.message : String(error)
      });
    }
  }
};

/**
 * Create a new strategy
 * @route POST /api/strategies
 */
export const createStrategy = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
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
    const validationResult = strategyExecutionService.validateStrategy(strategyData);
    
    if (!validationResult.isValid) {
      res.status(400).json({
        error: 'Invalid strategy configuration',
        validation: validationResult
      });
      return;
    }
    
    // Create strategy
    const strategy = await strategyExecutionService.createStrategy(strategyData);
    
    res.status(201).json(strategy);
  } catch (error) {
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

/**
 * Update a strategy
 * @route PUT /api/strategies/:id
 */
export const updateStrategy = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    // Get existing strategy
    const existingStrategy = await strategyExecutionService.getStrategy(id);
    
    // Check if user has access to strategy
    if (existingStrategy.userId !== userId) {
      res.status(403).json({ error: 'Forbidden' });
      return;
    }
    
    const updateData = req.body;
    
    // Validate strategy
    const validationResult = strategyExecutionService.validateStrategy({
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
    const strategy = await strategyExecutionService.updateStrategy(id, updateData);
    
    res.status(200).json(strategy);
  } catch (error) {
    logger.error(`Error updating strategy ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error),
      body: req.body
    });
    
    if (error instanceof Error && error.message.includes('not found')) {
      res.status(404).json({
        error: 'Strategy not found',
        details: error.message
      });
    } else {
      res.status(500).json({
        error: 'Failed to update strategy',
        details: error instanceof Error ? error.message : String(error)
      });
    }
  }
};

/**
 * Delete a strategy
 * @route DELETE /api/strategies/:id
 */
export const deleteStrategy = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    // Get existing strategy
    const existingStrategy = await strategyExecutionService.getStrategy(id);
    
    // Check if user has access to strategy
    if (existingStrategy.userId !== userId) {
      res.status(403).json({ error: 'Forbidden' });
      return;
    }
    
    // Delete strategy
    await strategyExecutionService.deleteStrategy(id);
    
    res.status(204).send();
  } catch (error) {
    logger.error(`Error deleting strategy ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    if (error instanceof Error && error.message.includes('not found')) {
      res.status(404).json({
        error: 'Strategy not found',
        details: error.message
      });
    } else {
      res.status(500).json({
        error: 'Failed to delete strategy',
        details: error instanceof Error ? error.message : String(error)
      });
    }
  }
};

/**
 * Validate a strategy configuration
 * @route POST /api/strategies/validate
 */
export const validateStrategy = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const strategyData = req.body;
    
    // Validate strategy
    const validationResult = strategyExecutionService.validateStrategy(strategyData);
    
    res.status(200).json(validationResult);
  } catch (error) {
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

/**
 * Start executing a strategy
 * @route POST /api/strategies/:id/execute
 */
export const startStrategyExecution = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
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
    const existingStrategy = await strategyExecutionService.getStrategy(id);
    
    // Check if user has access to strategy
    if (existingStrategy.userId !== userId) {
      res.status(403).json({ error: 'Forbidden' });
      return;
    }
    
    // Start execution
    const execution = await strategyExecutionService.startExecution(id, userId, botId);
    
    res.status(201).json(execution);
  } catch (error) {
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

/**
 * Stop executing a strategy
 * @route POST /api/executions/:id/stop
 */
export const stopStrategyExecution = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    // Get execution
    const execution = await prisma.strategyExecution.findUnique({
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
    const updatedExecution = await strategyExecutionService.stopExecution(id);
    
    res.status(200).json(updatedExecution);
  } catch (error) {
    logger.error(`Error stopping execution ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    if (error instanceof Error && error.message.includes('not found')) {
      res.status(404).json({
        error: 'Execution not found',
        details: error.message
      });
    } else {
      res.status(500).json({
        error: 'Failed to stop strategy execution',
        details: error instanceof Error ? error.message : String(error)
      });
    }
  }
};

/**
 * Get all executions for a user
 * @route GET /api/executions
 */
export const getUserExecutions = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    // Get executions from database
    const executions = await prisma.strategyExecution.findMany({
      where: { userId }
    });
    
    res.status(200).json(executions);
  } catch (error) {
    logger.error('Error getting executions', {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get executions',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get execution by ID
 * @route GET /api/executions/:id
 */
export const getExecutionById = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    // Get execution
    const execution = await prisma.strategyExecution.findUnique({
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
  } catch (error) {
    logger.error(`Error getting execution ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get execution',
      details: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get execution results
 * @route GET /api/executions/:id/results
 */
export const getExecutionResults = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    // Get execution
    const execution = await prisma.strategyExecution.findUnique({
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
    const results = await prisma.strategyExecutionResult.findMany({
      where: { executionId: id }
    });
    
    res.status(200).json(results);
  } catch (error) {
    logger.error(`Error getting execution results ${req.params.id}`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      error: 'Failed to get execution results',
      details: error instanceof Error ? error.message : String(error)
    });
  }
}; 