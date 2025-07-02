/**
 * Working Trading Routes (TypeScript)
 * Delta Exchange India trading integration endpoints
 */

import express, { Request, Response, NextFunction } from 'express';
import { protect as auth } from '../middleware/auth';
import env from '../config/environment';
import { DeltaExchangeUnified, DeltaOrderRequest } from '../services/DeltaExchangeUnified';
import { logger } from '../utils/logger';

const router = express.Router();

// Initialize Delta Exchange Unified service
const deltaExchange = new DeltaExchangeUnified({
  apiKey: env.DELTA_EXCHANGE_API_KEY,
  apiSecret: env.DELTA_EXCHANGE_API_SECRET,
  testnet: env.DELTA_EXCHANGE_TESTNET
});

// A simple middleware to ensure the Delta client is initialized.
const ensureDeltaInitialized = (req: Request, res: Response, next: NextFunction) => {
  if (deltaExchange.isInitialized()) {
    return next();
  }
  // If not initialized, return an error
  res.status(503).json({
    success: false,
    error: 'Delta Exchange service is not initialized. Please try again later.'
  });
};

// Apply authentication middleware to all routes
router.use(auth);
router.use(ensureDeltaInitialized);

/**
 * GET /api/trading/status
 * Get trading service status
 */
router.get('/status', (req: Request, res: Response, next: NextFunction) => {
  try {
    res.json({
      success: true,
      data: {
        status: 'connected',
        exchange: 'delta_exchange_india',
        environment: env.DELTA_EXCHANGE_TESTNET ? 'testnet' : 'production',
        supportedSymbols: ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD'],
        timestamp: Date.now()
      }
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/trading/products
 * Get all available trading products
 */
router.get('/products', (req: Request, res: Response, next: NextFunction) => {
  try {
    const products = deltaExchange.getAllProducts();
    res.json({
      success: true,
      data: products,
      meta: {
        total: products.length,
        exchange: 'delta_exchange_india',
        timestamp: Date.now()
      }
    });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/trading/market-data/:symbol
 * Get real-time market data for a given symbol
 */
router.get('/market-data/:symbol', (req: Request, res: Response, next: NextFunction) => {
  const { symbol } = req.params;
  deltaExchange.getMarketData(symbol.toUpperCase())
    .then(marketData => {
      if (!marketData) {
        return res.status(404).json({
          success: false,
          error: 'Symbol not found'
        });
      }
      res.json({
        success: true,
        data: marketData,
        timestamp: Date.now(),
        source: 'delta_exchange_india'
      });
    })
    .catch(next);
});

/**
 * POST /api/trading/orders
 * Place a new order
 */
router.post('/orders', (req: Request, res: Response, next: NextFunction) => {
  const orderRequest: DeltaOrderRequest = req.body;
  
  // Basic validation before placing the order
  if (!orderRequest.product_id || !orderRequest.size || !orderRequest.side) {
    const error: any = new Error('Invalid order request: product_id, size, and side are required.');
    error.status = 400;
    return next(error);
  }

  deltaExchange.placeOrder(orderRequest)
    .then(response => {
      res.json({ success: true, data: response });
    })
    .catch(next);
});

export default router;