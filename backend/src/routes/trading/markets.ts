import express from 'express';
import { DeltaExchangeUnified } from '../../services/DeltaExchangeUnified';
import { validateMarketLookupParams } from '../../schemas/orderValidation';
import logger from '../../utils/logger';

const router = express.Router();

// Create DeltaExchangeUnified instance with test credentials
const deltaExchange = new DeltaExchangeUnified({
  apiKey: process.env.DELTA_API_KEY || '',
  apiSecret: process.env.DELTA_API_SECRET || '',
  testnet: process.env.DELTA_TESTNET === 'true'
});

// Initialize the DeltaExchangeUnified client if not already initialized
(async () => {
  try {
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
      logger.info('Delta Exchange client initialized successfully for markets API');
    }
  } catch (error) {
    logger.error('Failed to initialize Delta Exchange client for markets API:', error);
  }
})();

/**
 * @route GET /api/markets
 * @description Get all available markets
 * @access Public
 */
router.get('/', async (req, res) => {
  try {
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }
    
    const markets = await deltaExchange.getMarkets();
    
    return res.status(200).json({
      success: true,
      data: markets,
      message: 'Markets retrieved successfully'
    });
  } catch (error) {
    logger.error('Error fetching markets:', error);
    
    return res.status(500).json({
      success: false,
      error: 'Failed to Fetch Markets',
      message: error.message || 'An unexpected error occurred'
    });
  }
});

/**
 * @route GET /api/markets/lookup
 * @description Look up a specific market by symbol
 * @access Public
 */
router.get('/lookup', async (req, res) => {
  try {
    const { symbol } = req.query;
    
    if (!symbol || typeof symbol !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Missing Symbol',
        message: 'Symbol query parameter is required'
      });
    }
    
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }
    
    const market = await deltaExchange.getMarketBySymbol(symbol);
    
    if (!market) {
      return res.status(404).json({
        success: false,
        error: 'Market Not Found',
        message: `No market found for symbol: ${symbol}`
      });
    }
    
    return res.status(200).json({
      success: true,
      data: market,
      message: 'Market retrieved successfully'
    });
  } catch (error) {
    logger.error(`Error looking up market for symbol ${req.query.symbol}:`, error);
    
    return res.status(500).json({
      success: false,
      error: 'Failed to Lookup Market',
      message: error.message || 'An unexpected error occurred'
    });
  }
});

/**
 * @route POST /api/markets/lookup
 * @description Look up a specific market by symbol (using POST)
 * @access Public
 */
router.post('/lookup', async (req, res) => {
  try {
    const { error, value } = validateMarketLookupParams(req.body);
    
    if (error) {
      logger.warn('Invalid market lookup parameters:', error.message);
      return res.status(400).json({
        success: false,
        error: 'Validation Error',
        message: error.message
      });
    }
    
    const { symbol } = value;
    
    // Ensure Delta Exchange client is initialized
    if (!deltaExchange.isInitialized()) {
      await deltaExchange.initialize();
    }
    
    const market = await deltaExchange.getMarketBySymbol(symbol);
    
    if (!market) {
      return res.status(404).json({
        success: false,
        error: 'Market Not Found',
        message: `No market found for symbol: ${symbol}`
      });
    }
    
    return res.status(200).json({
      success: true,
      data: market,
      message: 'Market retrieved successfully'
    });
  } catch (error) {
    logger.error('Error looking up market:', error);
    
    return res.status(500).json({
      success: false,
      error: 'Failed to Lookup Market',
      message: error.message || 'An unexpected error occurred'
    });
  }
});

export default router; 