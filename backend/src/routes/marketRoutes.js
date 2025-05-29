/**
 * Market Routes
 * Provides endpoints for market data
 */

const express = require('express');
const router = express.Router();
const { auth } = require('../middleware/auth');
const marketController = require('../controllers/marketController');

/**
 * Get real-time price data
 * @route GET /api/market/price
 */
router.get('/price', auth, marketController.getRealTimePrice);

/**
 * Get historical OHLCV data
 * @route GET /api/market/history
 */
router.get('/history', auth, marketController.getHistoricalData);

/**
 * Get market pairs data (available trading pairs)
 * @route GET /api/market/pairs
 */
router.get('/pairs', auth, marketController.getMarketPairs);

module.exports = router; 