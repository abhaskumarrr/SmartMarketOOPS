"use strict";
/**
 * Market Data Routes
 * Real-time market data endpoints using Delta Exchange via CCXT
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const marketDataService_1 = require("../services/marketDataService");
const accurateMarketDataService_1 = require("../services/accurateMarketDataService");
// import { logger } from '../utils/logger';
// Simple console logger for now
const logger = {
    info: (message, ...args) => console.log(`[INFO] ${message}`, ...args),
    error: (message, ...args) => console.error(`[ERROR] ${message}`, ...args),
    warn: (message, ...args) => console.warn(`[WARN] ${message}`, ...args),
    debug: (message, ...args) => console.log(`[DEBUG] ${message}`, ...args)
};
const router = express_1.default.Router();
/**
 * GET /api/market-data
 * Get market data for all supported symbols
 */
router.get('/', async (req, res) => {
    try {
        const symbols = marketDataService_1.marketDataService.getSupportedSymbols();
        const marketData = await marketDataService_1.marketDataService.getMultipleMarketData(symbols);
        res.json({
            success: true,
            data: marketData,
            timestamp: Date.now(),
            source: marketDataService_1.marketDataService.isReady() ? 'delta_exchange' : 'mock'
        });
    }
    catch (error) {
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
        const marketData = await accurateMarketDataService_1.accurateMarketDataService.getMarketData(symbol.toUpperCase());
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
    }
    catch (error) {
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
        const symbols = accurateMarketDataService_1.accurateMarketDataService.getSupportedSymbols();
        const marketData = await accurateMarketDataService_1.accurateMarketDataService.getMultipleMarketData(symbols);
        res.json({
            success: true,
            data: marketData,
            timestamp: Date.now(),
            note: 'This data is cross-validated from multiple reliable exchanges',
            sources_used: [...new Set(marketData.map(d => d.source))]
        });
    }
    catch (error) {
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
        const isReady = marketDataService_1.marketDataService.isReady();
        const supportedSymbols = marketDataService_1.marketDataService.getSupportedSymbols();
        res.json({
            success: true,
            data: {
                status: isReady ? 'connected' : 'disconnected',
                source: isReady ? 'delta_exchange' : 'mock',
                supportedSymbols,
                timestamp: Date.now()
            }
        });
    }
    catch (error) {
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
        const marketData = await marketDataService_1.marketDataService.getMarketData(symbol.toUpperCase());
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
            source: marketDataService_1.marketDataService.isReady() ? 'delta_exchange' : 'mock'
        });
    }
    catch (error) {
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
        const limit = parseInt(req.query.limit) || 10;
        const orderBook = await marketDataService_1.marketDataService.getOrderBook(symbol.toUpperCase(), limit);
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
    }
    catch (error) {
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
        const limit = parseInt(req.query.limit) || 50;
        const trades = await marketDataService_1.marketDataService.getRecentTrades(symbol.toUpperCase(), limit);
        res.json({
            success: true,
            data: trades,
            timestamp: Date.now(),
            source: 'delta_exchange'
        });
    }
    catch (error) {
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
        await marketDataService_1.marketDataService.cleanup();
        // The service will automatically reinitialize on next request
        res.json({
            success: true,
            message: 'Market data service refresh initiated',
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger.error('Error refreshing market data service:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to refresh service',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
exports.default = router;
//# sourceMappingURL=marketDataRoutes.js.map