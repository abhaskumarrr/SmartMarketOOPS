/**
 * Market Data Routes
 * Real-time market data endpoints using Delta Exchange via CCXT
 */

import express from 'express';
import { marketDataService } from '../services/marketDataService';
import { accurateMarketDataService } from '../services/accurateMarketDataService';
// import { logger } from '../utils/logger';

// Simple console logger for now
const logger = {
  info: (message: string, ...args: any[]) => console.log(`[INFO] ${message}`, ...args),
  error: (message: string, ...args: any[]) => console.error(`[ERROR] ${message}`, ...args),
  warn: (message: string, ...args: any[]) => console.warn(`[WARN] ${message}`, ...args),
  debug: (message: string, ...args: any[]) => console.log(`[DEBUG] ${message}`, ...args)
};

const router = express.Router();

/**
 * GET /api/market-data
 * Get market data for all supported symbols
 */
router.get('/', async (req, res) => {
  try {
    const symbols = marketDataService.getSupportedSymbols();
    const marketData = await marketDataService.getMultipleMarketData(symbols);

    res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      source: marketDataService.isReady() ? 'delta_exchange' : 'mock'
    });
  } catch (error) {
    logger.error('Error fetching market data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/market-data/accurate/:symbol
 * Get ACCURATE market data for a specific symbol using multiple sources
 */
router.get('/accurate/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const marketData = await accurateMarketDataService.getMarketData(symbol.toUpperCase());

    if (!marketData) {
      return res.status(404).json({
        success: false,
        error: 'Symbol not found',
        message: `Accurate market data not available for symbol: ${symbol}`
      });
    }

    res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      source: marketData.source,
      validated: marketData.isValidated,
      note: 'This data is cross-validated from multiple reliable exchanges'
    });
  } catch (error) {
    logger.error(`Error fetching accurate market data for ${req.params.symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch accurate market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/market-data/accurate
 * Get ACCURATE market data for all supported symbols
 */
router.get('/accurate', async (req, res) => {
  try {
    const symbols = accurateMarketDataService.getSupportedSymbols();
    const marketData = await accurateMarketDataService.getMultipleMarketData(symbols);

    res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      note: 'This data is cross-validated from multiple reliable exchanges',
      sources_used: [...new Set(marketData.map(d => d.source))]
    });
  } catch (error) {
    logger.error('Error fetching accurate market data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch accurate market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/market-data/status
 * Get market data service status
 */
router.get('/status', async (req, res) => {
  try {
    const isReady = marketDataService.isReady();
    const supportedSymbols = marketDataService.getSupportedSymbols();

    res.json({
      success: true,
      data: {
        status: isReady ? 'connected' : 'disconnected',
        source: isReady ? 'delta_exchange' : 'mock',
        supportedSymbols,
        timestamp: Date.now()
      }
    });
  } catch (error) {
    logger.error('Error getting market data status:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get status',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/market-data/:symbol
 * Get market data for a specific symbol
 */
router.get('/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const marketData = await marketDataService.getMarketData(symbol.toUpperCase());

    if (!marketData) {
      return res.status(404).json({
        success: false,
        error: 'Symbol not found',
        message: `Market data not available for symbol: ${symbol}`
      });
    }

    res.json({
      success: true,
      data: marketData,
      timestamp: Date.now(),
      source: marketDataService.isReady() ? 'delta_exchange' : 'mock'
    });
  } catch (error) {
    logger.error(`Error fetching market data for ${req.params.symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch market data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/market-data/:symbol/orderbook
 * Get order book data for a specific symbol
 */
router.get('/:symbol/orderbook', async (req, res) => {
  try {
    const { symbol } = req.params;
    const limit = parseInt(req.query.limit as string) || 10;

    const orderBook = await marketDataService.getOrderBook(symbol.toUpperCase(), limit);

    if (!orderBook) {
      return res.status(404).json({
        success: false,
        error: 'Order book not available',
        message: `Order book data not available for symbol: ${symbol}`
      });
    }

    res.json({
      success: true,
      data: orderBook,
      timestamp: Date.now(),
      source: 'delta_exchange'
    });
  } catch (error) {
    logger.error(`Error fetching order book for ${req.params.symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch order book',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/market-data/:symbol/trades
 * Get recent trades for a specific symbol
 */
router.get('/:symbol/trades', async (req, res) => {
  try {
    const { symbol } = req.params;
    const limit = parseInt(req.query.limit as string) || 50;

    const trades = await marketDataService.getRecentTrades(symbol.toUpperCase(), limit);

    res.json({
      success: true,
      data: trades,
      timestamp: Date.now(),
      source: 'delta_exchange'
    });
  } catch (error) {
    logger.error(`Error fetching trades for ${req.params.symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch trades',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/market-data/refresh
 * Force refresh market data service connection
 */
router.post('/refresh', async (req, res) => {
  try {
    // Cleanup and reinitialize the market data service
    await marketDataService.cleanup();

    // The service will automatically reinitialize on next request
    res.json({
      success: true,
      message: 'Market data service refresh initiated',
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error refreshing market data service:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to refresh service',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
