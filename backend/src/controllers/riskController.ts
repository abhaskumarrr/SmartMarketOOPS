/**
 * Risk Management Controller
 * Handles risk management API endpoints
 */

import { Request, Response } from 'express';
import prisma from '../utils/prismaClient';
import { createLogger, LogData } from '../utils/logger';
import riskManagementService from '../services/trading/riskManagementService';
import riskAssessmentService from '../services/trading/riskAssessmentService';
import circuitBreakerService from '../services/trading/circuitBreakerService';

// Create logger
const logger = createLogger('RiskController');

// Extend Request type to include user
interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    [key: string]: any;
  };
}

/**
 * Get risk settings for a user or bot
 * @param req - Express request
 * @param res - Express response
 */
export const getRiskSettings = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const { botId } = req.query;
    
    const settings = await riskManagementService.getRiskSettings(
      userId,
      botId as string | undefined
    );
    
    res.status(200).json(settings);
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error getting risk settings', logData);
    
    res.status(500).json({
      error: 'Failed to get risk settings',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Create or update risk settings
 * @param req - Express request
 * @param res - Express response
 */
export const saveRiskSettings = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const settings = req.body;
    
    // Ensure settings have the correct user ID
    settings.userId = userId;
    
    const savedSettings = await riskManagementService.saveRiskSettings(settings);
    
    res.status(200).json(savedSettings);
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error saving risk settings', logData);
    
    res.status(500).json({
      error: 'Failed to save risk settings',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Calculate position size
 * @param req - Express request
 * @param res - Express response
 */
export const calculatePositionSize = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const {
      botId,
      symbol,
      direction,
      entryPrice,
      stopLossPrice,
      stopLossPercentage,
      confidence
    } = req.body;
    
    // Validate required fields
    if (!symbol || !direction || !entryPrice) {
      res.status(400).json({
        error: 'Missing required fields',
        message: 'Symbol, direction, and entryPrice are required'
      });
      return;
    }
    
    // Validate direction
    if (direction !== 'long' && direction !== 'short') {
      res.status(400).json({
        error: 'Invalid direction',
        message: 'Direction must be either "long" or "short"'
      });
      return;
    }
    
    // Validate that at least one stop loss parameter is provided
    if (!stopLossPrice && !stopLossPercentage) {
      logger.warn('No stop loss parameters provided for position size calculation', {
        userId,
        symbol,
        direction
      });
    }
    
    const result = await riskManagementService.calculatePositionSize({
      userId,
      botId,
      symbol,
      direction,
      entryPrice: parseFloat(entryPrice),
      stopLossPrice: stopLossPrice ? parseFloat(stopLossPrice) : undefined,
      stopLossPercentage: stopLossPercentage ? parseFloat(stopLossPercentage) : undefined,
      confidence: confidence ? parseInt(confidence, 10) : undefined
    });
    
    res.status(200).json(result);
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error calculating position size', logData);
    
    res.status(500).json({
      error: 'Failed to calculate position size',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Generate risk report
 * @param req - Express request
 * @param res - Express response
 */
export const generateRiskReport = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const report = await riskAssessmentService.generateRiskReport(userId);
    
    res.status(200).json(report);
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error generating risk report', logData);
    
    res.status(500).json({
      error: 'Failed to generate risk report',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Analyze trade risk
 * @param req - Express request
 * @param res - Express response
 */
export const analyzeTradeRisk = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const { positionId } = req.params;
    
    if (!positionId) {
      res.status(400).json({
        error: 'Missing position ID',
        message: 'Position ID is required'
      });
      return;
    }
    
    const analysis = await riskAssessmentService.analyzeTradeRisk(userId, positionId);
    
    res.status(200).json(analysis);
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error analyzing trade risk', logData);
    
    res.status(500).json({
      error: 'Failed to analyze trade risk',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Get risk alerts
 * @param req - Express request
 * @param res - Express response
 */
export const getRiskAlerts = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const { acknowledged, level, limit } = req.query;
    
    // Build query
    const query: any = {
      where: {
        userId
      },
      orderBy: {
        timestamp: 'desc'
      }
    };
    
    // Add filters
    if (acknowledged !== undefined) {
      query.where.acknowledged = acknowledged === 'true';
    }
    
    if (level) {
      query.where.level = level;
    }
    
    // Add limit
    if (limit) {
      query.take = parseInt(limit as string, 10);
    }
    
    // Execute query
    const alerts = await prisma.riskAlert.findMany(query);
    
    res.status(200).json(alerts);
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error getting risk alerts', logData);
    
    res.status(500).json({
      error: 'Failed to get risk alerts',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Acknowledge risk alert
 * @param req - Express request
 * @param res - Express response
 */
export const acknowledgeRiskAlert = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const { alertId } = req.params;
    
    if (!alertId) {
      res.status(400).json({
        error: 'Missing alert ID',
        message: 'Alert ID is required'
      });
      return;
    }
    
    // Check if alert exists and belongs to the user
    const alert = await prisma.riskAlert.findFirst({
      where: {
        id: alertId,
        userId
      }
    });
    
    if (!alert) {
      res.status(404).json({
        error: 'Alert not found',
        message: 'Alert not found or does not belong to the user'
      });
      return;
    }
    
    // Update alert
    const updatedAlert = await prisma.riskAlert.update({
      where: {
        id: alertId
      },
      data: {
        acknowledged: true
      }
    });
    
    res.status(200).json(updatedAlert);
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error acknowledging risk alert', logData);
    
    res.status(500).json({
      error: 'Failed to acknowledge risk alert',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Check circuit breaker status
 * @param req - Express request
 * @param res - Express response
 */
export const getCircuitBreakerStatus = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const { botId } = req.query;
    
    const status = circuitBreakerService.getCircuitBreakerStatus(
      userId,
      botId as string | undefined
    );
    
    res.status(200).json({
      isTripped: status?.isTripped || false,
      reason: status?.reason,
      trippedAt: status?.trippedAt,
      cooldownUntil: status?.cooldownUntil,
      resetable: status?.resetable
    });
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error getting circuit breaker status', logData);
    
    res.status(500).json({
      error: 'Failed to get circuit breaker status',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Reset circuit breaker
 * @param req - Express request
 * @param res - Express response
 */
export const resetCircuitBreaker = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const { botId } = req.body;
    
    const success = await circuitBreakerService.resetCircuitBreaker(
      userId,
      botId
    );
    
    if (success) {
      res.status(200).json({
        success: true,
        message: 'Circuit breaker reset successfully'
      });
    } else {
      res.status(400).json({
        success: false,
        message: 'Failed to reset circuit breaker, may not be resetable'
      });
    }
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error resetting circuit breaker', logData);
    
    res.status(500).json({
      error: 'Failed to reset circuit breaker',
      message: error instanceof Error ? error.message : String(error)
    });
  }
};

/**
 * Check if trading is allowed
 * @param req - Express request
 * @param res - Express response
 */
export const checkTradingAllowed = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Unauthorized' });
      return;
    }
    
    const { botId, symbol } = req.query;
    
    const result = await circuitBreakerService.isTradingAllowed(
      userId,
      botId as string | undefined,
      symbol as string | undefined
    );
    
    res.status(200).json(result);
  } catch (error) {
    const logData: LogData = {
      userId: req.user?.id,
      error: error instanceof Error ? error.message : String(error)
    };
    
    logger.error('Error checking if trading is allowed', logData);
    
    res.status(500).json({
      error: 'Failed to check if trading is allowed',
      message: error instanceof Error ? error.message : String(error)
    });
  }
}; 