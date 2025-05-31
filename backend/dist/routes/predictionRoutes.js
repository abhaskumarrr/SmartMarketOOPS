/**
 * Prediction Routes
 * Provides endpoints for market predictions and analytics
 */
const express = require('express');
const router = express.Router();
const { auth } = require('../middleware/auth');
const predictionController = require('../controllers/predictionController');
/**
 * Get the latest price predictions
 * @route GET /api/predictions/latest
 */
router.get('/latest', auth, predictionController.getLatestPredictions);
/**
 * Get historical predictions performance
 * @route GET /api/predictions/history
 */
router.get('/history', auth, predictionController.getPredictionHistory);
/**
 * Get current market analysis
 * @route GET /api/market/analysis
 */
router.get('/market/analysis', auth, predictionController.getMarketAnalysis);
module.exports = router;
//# sourceMappingURL=predictionRoutes.js.map