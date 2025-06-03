"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const marketDataService_1 = require("../services/marketDataService");
const logger_1 = require("../utils/logger");
const router = express_1.default.Router();
// In-memory storage for paper trading data (in production, use Redis or database)
let paperTradingState = {
    portfolio: {
        balance: 2000,
        totalPnL: 0,
        positions: [],
        trades: [],
        isActive: false
    },
    config: {
        initialCapital: 2000,
        leverage: 3,
        riskPerTrade: 0.02,
        assets: ['ETH/USDT', 'BTC/USDT'],
        stopLossPercentage: 0.025
    },
    lastUpdate: new Date()
};
// Mock current position data (replace with actual data from running system)
const mockCurrentPosition = {
    id: 'pos_1733270421000',
    symbol: 'ETH/USDT',
    side: 'buy',
    size: 0.6203,
    entryPrice: 2579.39,
    stopLoss: 2514.91,
    takeProfitLevels: [
        { percentage: 25, ratio: 2.0, price: 2643.37, executed: false },
        { percentage: 50, ratio: 5.0, price: 2901.35, executed: false },
        { percentage: 25, ratio: 5.0, price: 2901.35, executed: false }
    ],
    openTime: new Date(),
    status: 'open'
};
paperTradingState.portfolio.positions = [mockCurrentPosition];
// Get portfolio overview
router.get('/portfolio', async (req, res) => {
    try {
        // Get current prices for unrealized P&L calculation
        const positions = [];
        for (const position of paperTradingState.portfolio.positions) {
            if (position.status === 'open') {
                try {
                    const currentPrice = await getCurrentPrice(position.symbol);
                    const unrealizedPnL = position.side === 'buy'
                        ? (currentPrice - position.entryPrice) * position.size
                        : (position.entryPrice - currentPrice) * position.size;
                    positions.push({
                        ...position,
                        currentPrice,
                        unrealizedPnL
                    });
                }
                catch (error) {
                    logger_1.logger.error(`Error getting price for ${position.symbol}:`, error);
                    positions.push({
                        ...position,
                        currentPrice: position.entryPrice,
                        unrealizedPnL: 0
                    });
                }
            }
            else {
                positions.push(position);
            }
        }
        const totalUnrealizedPnL = positions
            .filter(p => p.status === 'open')
            .reduce((sum, p) => sum + (p.unrealizedPnL || 0), 0);
        res.json({
            success: true,
            data: {
                ...paperTradingState.portfolio,
                positions,
                totalUnrealizedPnL,
                currentBalance: paperTradingState.portfolio.balance + paperTradingState.portfolio.totalPnL + totalUnrealizedPnL,
                lastUpdate: new Date()
            }
        });
    }
    catch (error) {
        logger_1.logger.error('Error getting portfolio:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});
// Get real-time market data
router.get('/market-data', async (req, res) => {
    try {
        const symbols = ['ETH/USDT', 'BTC/USDT'];
        const marketData = {};
        for (const symbol of symbols) {
            try {
                const ticker = await marketDataService_1.marketDataService.getMarketData(symbol);
                marketData[symbol] = {
                    symbol,
                    price: ticker?.price || 0,
                    change: ticker?.change || '0.00',
                    changePercent: ticker?.changePercent || 0,
                    volume: ticker?.volume || 0,
                    high24h: ticker?.high24h || ticker?.price || 0,
                    low24h: ticker?.low24h || ticker?.price || 0,
                    timestamp: new Date()
                };
            }
            catch (error) {
                logger_1.logger.error(`Error getting market data for ${symbol}:`, error);
                marketData[symbol] = {
                    symbol,
                    price: 0,
                    change: '0.00',
                    changePercent: 0,
                    volume: 0,
                    high24h: 0,
                    low24h: 0,
                    timestamp: new Date()
                };
            }
        }
        res.json({
            success: true,
            data: marketData
        });
    }
    catch (error) {
        logger_1.logger.error('Error getting market data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});
// Get trading system status
router.get('/status', async (req, res) => {
    try {
        const status = {
            isConnected: true,
            isTrading: paperTradingState.portfolio.isActive,
            exchange: 'Delta Exchange Indian Testnet',
            lastUpdate: paperTradingState.lastUpdate,
            config: paperTradingState.config,
            nextCycleIn: 30 - (Date.now() % 30000) / 1000, // Seconds until next 30s cycle
            systemHealth: 'healthy'
        };
        res.json({
            success: true,
            data: status
        });
    }
    catch (error) {
        logger_1.logger.error('Error getting status:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});
// Get historical price data for charts
router.get('/chart-data/:symbol', async (req, res) => {
    try {
        const { symbol } = req.params;
        const { timeframe = '1m', limit = '100' } = req.query;
        const limitNum = parseInt(limit) || 100;
        // Generate mock historical data (in production, fetch from exchange)
        const now = Date.now();
        const interval = timeframe === '1m' ? 60000 : 300000; // 1min or 5min
        const data = [];
        let basePrice = symbol === 'ETH/USDT' ? 2567 : 105444;
        for (let i = limitNum; i >= 0; i--) {
            const timestamp = now - (i * interval);
            const volatility = 0.002; // 0.2% volatility
            const change = (Math.random() - 0.5) * volatility;
            const open = basePrice;
            const close = basePrice * (1 + change);
            const high = Math.max(open, close) * (1 + Math.random() * 0.001);
            const low = Math.min(open, close) * (1 - Math.random() * 0.001);
            const volume = Math.random() * 1000;
            data.push({
                timestamp,
                open: parseFloat(open.toFixed(2)),
                high: parseFloat(high.toFixed(2)),
                low: parseFloat(low.toFixed(2)),
                close: parseFloat(close.toFixed(2)),
                volume: parseFloat(volume.toFixed(2))
            });
            basePrice = close;
        }
        res.json({
            success: true,
            data: {
                symbol,
                timeframe,
                candles: data
            }
        });
    }
    catch (error) {
        logger_1.logger.error('Error getting chart data:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});
// Update paper trading state (called by trading system)
router.post('/update', async (req, res) => {
    try {
        const { portfolio, config } = req.body;
        if (portfolio) {
            paperTradingState.portfolio = { ...paperTradingState.portfolio, ...portfolio };
        }
        if (config) {
            paperTradingState.config = { ...paperTradingState.config, ...config };
        }
        paperTradingState.lastUpdate = new Date();
        res.json({
            success: true,
            message: 'Paper trading state updated'
        });
    }
    catch (error) {
        logger_1.logger.error('Error updating paper trading state:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});
// Helper function to get current price
async function getCurrentPrice(symbol) {
    try {
        // Use the symbol directly - the market data service handles mapping
        const marketData = await marketDataService_1.marketDataService.getMarketData(symbol);
        return marketData?.price || 0;
    }
    catch (error) {
        logger_1.logger.error(`Error fetching price for ${symbol}:`, error);
        return 0;
    }
}
exports.default = router;
//# sourceMappingURL=paperTradingRoutes.js.map