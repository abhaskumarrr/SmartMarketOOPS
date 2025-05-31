/**
 * Prediction Controller
 * Handles market predictions and analytics
 */
const prisma = require('../utils/prismaClient');
const { createError } = require('../middleware/errorHandler');
/**
 * Get latest price predictions for a specific symbol
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getLatestPredictions = async (req, res, next) => {
    try {
        const { symbol = 'BTCUSD' } = req.query;
        // TODO: In production, fetch this from ML service or database
        // For now, return mock data
        const mockPredictions = {
            timestamp: new Date().toISOString(),
            symbol: symbol,
            timeframe: '1h',
            prediction: {
                direction: 'up',
                confidence: 0.78,
                priceTarget: 45780.50,
                timeTarget: new Date(Date.now() + 3600000).toISOString()
            },
            model: {
                name: 'CNNLSTM',
                version: 'v_20250523_023634'
            }
        };
        res.json({
            success: true,
            data: mockPredictions
        });
    }
    catch (error) {
        next(createError(`Failed to fetch predictions: ${error.message}`, 500));
    }
};
/**
 * Get historical predictions and their performance
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getPredictionHistory = async (req, res, next) => {
    try {
        const { symbol = 'BTCUSD', limit = 10, offset = 0 } = req.query;
        // TODO: In production, fetch this from database
        // For now, return mock data
        const mockHistory = {
            total: 100,
            accuracy: 0.72,
            predictions: [
                {
                    timestamp: new Date(Date.now() - 86400000).toISOString(),
                    symbol: symbol,
                    predicted: 'up',
                    actual: 'up',
                    confidence: 0.82,
                    profit: 1.2
                },
                {
                    timestamp: new Date(Date.now() - 172800000).toISOString(),
                    symbol: symbol,
                    predicted: 'down',
                    actual: 'down',
                    confidence: 0.65,
                    profit: 0.8
                }
            ]
        };
        res.json({
            success: true,
            data: mockHistory
        });
    }
    catch (error) {
        next(createError(`Failed to fetch prediction history: ${error.message}`, 500));
    }
};
/**
 * Get current market analysis
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getMarketAnalysis = async (req, res, next) => {
    try {
        const { symbol = 'BTCUSD' } = req.query;
        // TODO: In production, fetch this from analysis service
        // For now, return mock data
        const mockAnalysis = {
            timestamp: new Date().toISOString(),
            marketCondition: 'neutral',
            volatilityIndex: 0.43,
            sentiment: {
                overall: 'bullish',
                source: 'combined',
                socialMedia: 0.62,
                news: 0.58,
                technicals: 0.64
            },
            keyLevels: {
                [symbol]: {
                    support: [43500, 42800, 41200],
                    resistance: [46200, 47500, 49000]
                }
            }
        };
        res.json({
            success: true,
            data: mockAnalysis
        });
    }
    catch (error) {
        next(createError(`Failed to fetch market analysis: ${error.message}`, 500));
    }
};
module.exports = {
    getLatestPredictions,
    getPredictionHistory,
    getMarketAnalysis
};
//# sourceMappingURL=predictionController.js.map