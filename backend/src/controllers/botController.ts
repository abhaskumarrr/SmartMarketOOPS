/**
 * Trading Bot Controller
 * Handles trading bot configuration and control endpoints
 */

import { Request, Response } from 'express';
import * as botService from '../services/trading/botService';
import { validateRequest } from '../utils/validator';
import { body, param } from 'express-validator';

/**
 * Create a new trading bot
 * @route POST /api/bots
 * @access Private
 */
export const createBot = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      body('name').notEmpty().withMessage('Name is required'),
      body('symbol').notEmpty().withMessage('Symbol is required'),
      body('strategy').notEmpty().withMessage('Strategy is required'),
      body('timeframe').notEmpty().withMessage('Timeframe is required'),
      body('parameters').optional().isObject().withMessage('Parameters must be an object')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    console.error('Create bot error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while creating bot',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Get all trading bots for current user
 * @route GET /api/bots
 * @access Private
 */
export const getBots = async (req: Request, res: Response) => {
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
  } catch (error) {
    console.error('Get bots error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while fetching bots',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Get a specific trading bot
 * @route GET /api/bots/:id
 * @access Private
 */
export const getBot = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    console.error('Get bot error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while fetching bot',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Update a trading bot
 * @route PUT /api/bots/:id
 * @access Private
 */
export const updateBot = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required'),
      body('name').optional().isString().withMessage('Name must be a string'),
      body('symbol').optional().isString().withMessage('Symbol must be a string'),
      body('strategy').optional().isString().withMessage('Strategy must be a string'),
      body('timeframe').optional().isString().withMessage('Timeframe must be a string'),
      body('parameters').optional().isObject().withMessage('Parameters must be an object')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
      if (updateData[key as keyof typeof updateData] === undefined) {
        delete updateData[key as keyof typeof updateData];
      }
    });

    const updatedBot = await botService.updateBot(botId, userId, updateData);

    return res.status(200).json({
      success: true,
      message: 'Bot updated successfully',
      data: updatedBot
    });
  } catch (error) {
    // Handle specific errors
    if ((error as Error).message === 'Bot not found or access denied') {
      return res.status(404).json({
        success: false,
        message: 'Bot not found or access denied'
      });
    }
    
    if ((error as Error).message === 'Cannot update bot while it is active. Stop the bot first.') {
      return res.status(400).json({
        success: false,
        message: 'Cannot update bot while it is active. Stop the bot first.'
      });
    }
    
    console.error('Update bot error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while updating bot',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Delete a trading bot
 * @route DELETE /api/bots/:id
 * @access Private
 */
export const deleteBot = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    // Handle specific errors
    if ((error as Error).message === 'Bot not found or access denied') {
      return res.status(404).json({
        success: false,
        message: 'Bot not found or access denied'
      });
    }
    
    console.error('Delete bot error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while deleting bot',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Start a trading bot
 * @route POST /api/bots/:id/start
 * @access Private
 */
export const startBot = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    // Handle specific errors
    if ((error as Error).message === 'Bot not found or access denied') {
      return res.status(404).json({
        success: false,
        message: 'Bot not found or access denied'
      });
    }
    
    if ((error as Error).message === 'Bot is already running') {
      return res.status(400).json({
        success: false,
        message: 'Bot is already running'
      });
    }
    
    if ((error as Error).message === 'Bot has no risk settings. Please configure risk settings before starting.') {
      return res.status(400).json({
        success: false,
        message: 'Bot has no risk settings. Please configure risk settings before starting.'
      });
    }
    
    console.error('Start bot error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while starting bot',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Stop a trading bot
 * @route POST /api/bots/:id/stop
 * @access Private
 */
export const stopBot = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    // Handle specific errors
    if ((error as Error).message === 'Bot not found or access denied') {
      return res.status(404).json({
        success: false,
        message: 'Bot not found or access denied'
      });
    }
    
    if ((error as Error).message === 'Bot is not running') {
      return res.status(400).json({
        success: false,
        message: 'Bot is not running'
      });
    }
    
    console.error('Stop bot error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while stopping bot',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Pause a trading bot
 * @route POST /api/bots/:id/pause
 * @access Private
 */
export const pauseBot = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required'),
      body('duration').optional().isInt().withMessage('Duration must be an integer')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    // Handle specific errors
    if ((error as Error).message === 'Bot not found or access denied') {
      return res.status(404).json({
        success: false,
        message: 'Bot not found or access denied'
      });
    }
    
    if ((error as Error).message === 'Bot is not running') {
      return res.status(400).json({
        success: false,
        message: 'Bot is not running'
      });
    }
    
    console.error('Pause bot error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while pausing bot',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Run backtest for a bot
 * @route POST /api/bots/:id/backtest
 * @access Private
 */
export const runBacktest = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required'),
      body('symbol').notEmpty().withMessage('Symbol is required'),
      body('timeframe').notEmpty().withMessage('Timeframe is required'),
      body('startDate').isISO8601().withMessage('Valid start date is required'),
      body('endDate').isISO8601().withMessage('Valid end date is required'),
      body('initialCapital').isFloat({ min: 100 }).withMessage('Initial capital must be at least $100'),
      body('leverage').isFloat({ min: 1, max: 100 }).withMessage('Leverage must be between 1 and 100'),
      body('riskPerTrade').isFloat({ min: 0.1, max: 10 }).withMessage('Risk per trade must be between 0.1% and 10%'),
      body('commission').isFloat({ min: 0, max: 1 }).withMessage('Commission must be between 0% and 1%')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    // Handle specific errors
    if ((error as Error).message === 'Bot not found or access denied') {
      return res.status(404).json({
        success: false,
        message: 'Bot not found or access denied'
      });
    }

    console.error('Run backtest error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while running backtest',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Get backtest history for a bot
 * @route GET /api/bots/:id/backtests
 * @access Private
 */
export const getBacktestHistory = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
    const limit = parseInt(req.query.limit as string) || 10;
    const offset = parseInt(req.query.offset as string) || 0;

    const results = await botService.getBacktestHistory(botId, userId, limit, offset);

    return res.status(200).json({
      success: true,
      data: results
    });
  } catch (error) {
    console.error('Get backtest history error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while fetching backtest history',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Get bot status
 * @route GET /api/bots/:id/status
 * @access Private
 */
export const getBotStatus = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    // Handle specific errors
    if ((error as Error).message === 'Bot not found or access denied') {
      return res.status(404).json({
        success: false,
        message: 'Bot not found or access denied'
      });
    }
    
    console.error('Get bot status error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while fetching bot status',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Configure risk settings for a bot
 * @route POST /api/bots/:id/risk
 * @access Private
 */
export const configureBotRiskSettings = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required'),
      body('name').optional().isString().withMessage('Name must be a string'),
      body('description').optional().isString().withMessage('Description must be a string'),
      body('positionSizingMethod').optional().isString().withMessage('Position sizing method must be a string'),
      body('riskPercentage').optional().isFloat().withMessage('Risk percentage must be a number'),
      body('maxPositionSize').optional().isFloat().withMessage('Max position size must be a number'),
      body('stopLossType').optional().isString().withMessage('Stop loss type must be a string'),
      body('stopLossValue').optional().isFloat().withMessage('Stop loss value must be a number'),
      body('takeProfitType').optional().isString().withMessage('Take profit type must be a string'),
      body('takeProfitValue').optional().isFloat().withMessage('Take profit value must be a number')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    // Handle specific errors
    if ((error as Error).message === 'Bot not found or access denied') {
      return res.status(404).json({
        success: false,
        message: 'Bot not found or access denied'
      });
    }
    
    console.error('Configure risk settings error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while configuring risk settings',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Update bot health status
 * @route POST /api/bots/:id/health
 * @access Private
 */
export const updateBotHealth = async (req: Request, res: Response) => {
  try {
    // Validate request
    const validationRules = [
      param('id').notEmpty().withMessage('Bot ID is required'),
      body('health').isIn(['excellent', 'good', 'degraded', 'poor', 'critical', 'unknown']).withMessage('Valid health status is required'),
      body('metrics').isObject().withMessage('Metrics must be an object')
    ];

    const validationErrors = await validateRequest(req, validationRules);
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
  } catch (error) {
    console.error('Update bot health error:', error);
    return res.status(500).json({
      success: false,
      message: 'Server error while updating bot health',
      error: process.env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
}; 