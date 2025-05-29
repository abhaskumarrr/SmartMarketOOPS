/**
 * Signal Controller
 * Handles API endpoints for trading signal generation and retrieval
 */

import { Request, Response } from 'express';
import { v4 as uuidv4 } from 'uuid';
import signalGenerationService from '../services/trading/signalGenerationService';
import { SignalFilterCriteria, SignalGenerationOptions, SignalDirection, SignalType, SignalStrength, SignalTimeframe } from '../types/signals';
import { createLogger } from '../utils/logger';

// Create logger
const logger = createLogger('SignalController');

/**
 * Generate trading signals for a symbol
 * @route POST /api/signals/generate
 */
export const generateSignals = async (req: Request, res: Response): Promise<void> => {
  try {
    const { symbol, features, options } = req.body;
    
    // Validate request body
    if (!symbol) {
      res.status(400).json({
        success: false,
        error: {
          code: 'MISSING_SYMBOL',
          message: 'Symbol is required'
        },
        timestamp: new Date().toISOString()
      });
      return;
    }
    
    if (!features || Object.keys(features).length === 0) {
      res.status(400).json({
        success: false,
        error: {
          code: 'MISSING_FEATURES',
          message: 'Market features are required'
        },
        timestamp: new Date().toISOString()
      });
      return;
    }
    
    // Generate signals
    const signals = await signalGenerationService.generateSignals(
      symbol, 
      features, 
      options as SignalGenerationOptions
    );
    
    res.status(200).json({
      success: true,
      data: {
        signals,
        count: signals.length
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error generating signals', {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      success: false,
      error: {
        code: 'SIGNAL_GENERATION_ERROR',
        message: error instanceof Error ? error.message : 'Failed to generate signals'
      },
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * Get signals based on filter criteria
 * @route GET /api/signals
 */
export const getSignals = async (req: Request, res: Response): Promise<void> => {
  try {
    const {
      symbol,
      types,
      directions,
      minStrength,
      timeframes,
      minConfidenceScore,
      fromTimestamp,
      toTimestamp,
      status = 'active'
    } = req.query;
    
    // Build filter criteria
    const criteria: SignalFilterCriteria = {};
    
    if (symbol) {
      criteria.symbol = String(symbol);
    }
    
    if (types) {
      criteria.types = (Array.isArray(types) ? types : [types])
        .map(t => String(t).toUpperCase())
        .filter(t => Object.values(SignalType).includes(t as SignalType)) as SignalType[];
    }
    
    if (directions) {
      criteria.directions = (Array.isArray(directions) ? directions : [directions])
        .map(d => String(d).toUpperCase())
        .filter(d => Object.values(SignalDirection).includes(d as SignalDirection)) as SignalDirection[];
    }
    
    if (minStrength && Object.values(SignalStrength).includes(String(minStrength).toUpperCase() as SignalStrength)) {
      criteria.minStrength = String(minStrength).toUpperCase() as SignalStrength;
    }
    
    if (timeframes) {
      criteria.timeframes = (Array.isArray(timeframes) ? timeframes : [timeframes])
        .map(t => String(t).toUpperCase())
        .filter(t => Object.values(SignalTimeframe).includes(t as SignalTimeframe)) as SignalTimeframe[];
    }
    
    if (minConfidenceScore !== undefined) {
      const score = parseInt(String(minConfidenceScore), 10);
      if (!isNaN(score)) {
        criteria.minConfidenceScore = score;
      }
    }
    
    if (fromTimestamp) {
      criteria.fromTimestamp = String(fromTimestamp);
    }
    
    if (toTimestamp) {
      criteria.toTimestamp = String(toTimestamp);
    }
    
    if (status) {
      criteria.status = String(status) as 'active' | 'expired' | 'validated' | 'invalidated' | 'all';
    }
    
    // Get signals
    const signals = await signalGenerationService.getSignals(criteria);
    
    res.status(200).json({
      success: true,
      data: {
        signals,
        count: signals.length,
        criteria
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error getting signals', {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      success: false,
      error: {
        code: 'SIGNAL_RETRIEVAL_ERROR',
        message: error instanceof Error ? error.message : 'Failed to retrieve signals'
      },
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * Get latest signal for a symbol
 * @route GET /api/signals/:symbol/latest
 */
export const getLatestSignal = async (req: Request, res: Response): Promise<void> => {
  try {
    const { symbol } = req.params;
    
    if (!symbol) {
      res.status(400).json({
        success: false,
        error: {
          code: 'MISSING_SYMBOL',
          message: 'Symbol is required'
        },
        timestamp: new Date().toISOString()
      });
      return;
    }
    
    // Get latest signal
    const signal = await signalGenerationService.getLatestSignal(symbol);
    
    if (!signal) {
      res.status(404).json({
        success: false,
        error: {
          code: 'SIGNAL_NOT_FOUND',
          message: `No signals found for symbol ${symbol}`
        },
        timestamp: new Date().toISOString()
      });
      return;
    }
    
    res.status(200).json({
      success: true,
      data: signal,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error(`Error getting latest signal`, {
      error: error instanceof Error ? error.message : String(error)
    });
    
    res.status(500).json({
      success: false,
      error: {
        code: 'SIGNAL_RETRIEVAL_ERROR',
        message: error instanceof Error ? error.message : 'Failed to retrieve latest signal'
      },
      timestamp: new Date().toISOString()
    });
  }
}; 