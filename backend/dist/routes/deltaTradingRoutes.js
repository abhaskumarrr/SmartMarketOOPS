"use strict";
/**
 * Delta Exchange Trading Routes
 * Production-ready trading bot management endpoints for Delta Exchange India testnet
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const DeltaBotManager_1 = require("../services/DeltaBotManager");
const logger_1 = require("../utils/logger");
const websocketServer_1 = require("../sockets/websocketServer");
const router = express_1.default.Router();
// Global Delta Bot Manager instance
let botManager = null;
/**
 * Initialize Delta Bot Manager
 */
const initializeBotManager = async () => {
    if (!botManager) {
        botManager = new DeltaBotManager_1.DeltaBotManager();
        await botManager.initialize();
        // Set up event listeners for WebSocket broadcasting
        setupBotManagerEventListeners();
    }
    return botManager;
};
/**
 * Set up event listeners for WebSocket broadcasting
 */
const setupBotManagerEventListeners = () => {
    if (!botManager)
        return;
    botManager.on('botStatusChanged', (data) => {
        (0, websocketServer_1.broadcastToClients)('botStatusChanged', data);
    });
    botManager.on('botTradeExecuted', (data) => {
        (0, websocketServer_1.broadcastToClients)('botTradeExecuted', data);
    });
    botManager.on('botPositionClosed', (data) => {
        (0, websocketServer_1.broadcastToClients)('botPositionClosed', data);
    });
    botManager.on('botError', (data) => {
        (0, websocketServer_1.broadcastToClients)('botError', data);
    });
};
/**
 * GET /api/delta-trading/health
 * Public health check endpoint (no auth required)
 */
router.get('/health', async (req, res) => {
    try {
        res.json({
            success: true,
            message: 'Delta Exchange trading service is healthy',
            data: {
                exchange: 'delta_exchange_india',
                environment: process.env.DELTA_EXCHANGE_TESTNET === 'true' ? 'testnet' : 'production',
                api_configured: !!(process.env.DELTA_EXCHANGE_API_KEY && process.env.DELTA_EXCHANGE_API_SECRET),
                timestamp: Date.now()
            }
        });
    }
    catch (error) {
        logger_1.logger.error('Error in health check:', error);
        res.status(500).json({
            success: false,
            error: 'Health check failed',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * GET /api/delta-trading/test-connection
 * Test Delta Exchange connection (public endpoint)
 */
router.get('/test-connection', async (req, res) => {
    try {
        const manager = await initializeBotManager();
        res.json({
            success: true,
            message: 'Delta Exchange connection successful',
            data: {
                exchange: 'delta_exchange_india',
                environment: process.env.DELTA_EXCHANGE_TESTNET === 'true' ? 'testnet' : 'production',
                api_key_configured: !!process.env.DELTA_EXCHANGE_API_KEY,
                api_secret_configured: !!process.env.DELTA_EXCHANGE_API_SECRET
            },
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error('Error testing Delta Exchange connection:', error);
        res.status(500).json({
            success: false,
            error: 'Delta Exchange connection failed',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
// Apply authentication middleware to all other routes
// TODO: Re-enable authentication for production
// router.use(auth);
/**
 * GET /api/delta-trading/status
 * Get Delta Exchange trading service status
 */
router.get('/status', async (req, res) => {
    try {
        const manager = await initializeBotManager();
        const status = manager.getManagerStatus();
        res.json({
            success: true,
            data: {
                ...status,
                exchange: 'delta_exchange_india',
                environment: process.env.DELTA_EXCHANGE_TESTNET === 'true' ? 'testnet' : 'production',
                timestamp: Date.now()
            }
        });
    }
    catch (error) {
        logger_1.logger.error('Error getting Delta trading status:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get trading status',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * POST /api/delta-trading/bots
 * Create a new trading bot
 */
router.post('/bots', async (req, res) => {
    try {
        const manager = await initializeBotManager();
        const config = req.body;
        // Validate required fields
        if (!config.id || !config.name || !config.symbol) {
            return res.status(400).json({
                success: false,
                error: 'Invalid bot configuration',
                message: 'Missing required fields: id, name, symbol'
            });
        }
        // Set default values
        const botConfig = {
            id: config.id,
            name: config.name,
            symbol: config.symbol.toUpperCase(),
            strategy: config.strategy || 'momentum',
            capital: config.capital || 1000,
            leverage: config.leverage || 3,
            riskPerTrade: config.riskPerTrade || 2,
            maxPositions: config.maxPositions || 3,
            stopLoss: config.stopLoss || 5,
            takeProfit: config.takeProfit || 10,
            enabled: config.enabled !== false,
            testnet: process.env.DELTA_EXCHANGE_TESTNET === 'true'
        };
        const botId = await manager.createBot(botConfig);
        res.json({
            success: true,
            data: { botId, config: botConfig },
            message: 'Trading bot created successfully',
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error('Error creating bot:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to create bot',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * GET /api/delta-trading/bots
 * Get all trading bots
 */
router.get('/bots', async (req, res) => {
    try {
        const manager = await initializeBotManager();
        const bots = manager.getAllBotStatuses();
        res.json({
            success: true,
            data: bots,
            meta: {
                total: bots.length,
                running: bots.filter(b => b.status === 'running').length,
                stopped: bots.filter(b => b.status === 'stopped').length,
                paused: bots.filter(b => b.status === 'paused').length,
                error: bots.filter(b => b.status === 'error').length
            },
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error('Error getting bots:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get bots',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * GET /api/delta-trading/bots/:botId
 * Get specific bot status
 */
router.get('/bots/:botId', async (req, res) => {
    try {
        const { botId } = req.params;
        const manager = await initializeBotManager();
        const status = manager.getBotStatus(botId);
        res.json({
            success: true,
            data: status,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error(`Error getting bot ${req.params.botId}:`, error);
        res.status(404).json({
            success: false,
            error: 'Bot not found',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * POST /api/delta-trading/bots/:botId/start
 * Start a trading bot
 */
router.post('/bots/:botId/start', async (req, res) => {
    try {
        const { botId } = req.params;
        const manager = await initializeBotManager();
        await manager.startBot(botId);
        res.json({
            success: true,
            message: `Bot ${botId} started successfully`,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error(`Error starting bot ${req.params.botId}:`, error);
        res.status(500).json({
            success: false,
            error: 'Failed to start bot',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * POST /api/delta-trading/bots/:botId/stop
 * Stop a trading bot
 */
router.post('/bots/:botId/stop', async (req, res) => {
    try {
        const { botId } = req.params;
        const manager = await initializeBotManager();
        await manager.stopBot(botId);
        res.json({
            success: true,
            message: `Bot ${botId} stopped successfully`,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error(`Error stopping bot ${req.params.botId}:`, error);
        res.status(500).json({
            success: false,
            error: 'Failed to stop bot',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * POST /api/delta-trading/bots/:botId/pause
 * Pause a trading bot
 */
router.post('/bots/:botId/pause', async (req, res) => {
    try {
        const { botId } = req.params;
        const manager = await initializeBotManager();
        manager.pauseBot(botId);
        res.json({
            success: true,
            message: `Bot ${botId} paused successfully`,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error(`Error pausing bot ${req.params.botId}:`, error);
        res.status(500).json({
            success: false,
            error: 'Failed to pause bot',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * POST /api/delta-trading/bots/:botId/resume
 * Resume a trading bot
 */
router.post('/bots/:botId/resume', async (req, res) => {
    try {
        const { botId } = req.params;
        const manager = await initializeBotManager();
        manager.resumeBot(botId);
        res.json({
            success: true,
            message: `Bot ${botId} resumed successfully`,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error(`Error resuming bot ${req.params.botId}:`, error);
        res.status(500).json({
            success: false,
            error: 'Failed to resume bot',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * DELETE /api/delta-trading/bots/:botId
 * Remove a trading bot
 */
router.delete('/bots/:botId', async (req, res) => {
    try {
        const { botId } = req.params;
        const manager = await initializeBotManager();
        await manager.removeBot(botId);
        res.json({
            success: true,
            message: `Bot ${botId} removed successfully`,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error(`Error removing bot ${req.params.botId}:`, error);
        res.status(500).json({
            success: false,
            error: 'Failed to remove bot',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * GET /api/delta-trading/bots/:botId/performance
 * Get bot performance metrics
 */
router.get('/bots/:botId/performance', async (req, res) => {
    try {
        const { botId } = req.params;
        const manager = await initializeBotManager();
        const performance = manager.getBotPerformance(botId);
        res.json({
            success: true,
            data: performance,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error(`Error getting bot performance ${req.params.botId}:`, error);
        res.status(404).json({
            success: false,
            error: 'Bot not found',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * PUT /api/delta-trading/bots/:botId/config
 * Update bot configuration
 */
router.put('/bots/:botId/config', async (req, res) => {
    try {
        const { botId } = req.params;
        const manager = await initializeBotManager();
        const newConfig = req.body;
        manager.updateBotConfig(botId, newConfig);
        res.json({
            success: true,
            message: `Bot ${botId} configuration updated successfully`,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error(`Error updating bot config ${req.params.botId}:`, error);
        res.status(500).json({
            success: false,
            error: 'Failed to update bot configuration',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * POST /api/delta-trading/emergency-stop
 * Emergency stop all bots
 */
router.post('/emergency-stop', async (req, res) => {
    try {
        const manager = await initializeBotManager();
        await manager.emergencyStopAll();
        res.json({
            success: true,
            message: 'Emergency stop executed for all bots',
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error('Error in emergency stop:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to execute emergency stop',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * GET /api/delta-trading/performance
 * Get overall performance summary
 */
router.get('/performance', async (req, res) => {
    try {
        const manager = await initializeBotManager();
        const performance = manager.getAllBotsPerformance();
        res.json({
            success: true,
            data: performance,
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error('Error getting overall performance:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get performance data',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
/**
 * GET /api/delta-trading/products
 * Get available Delta Exchange products
 */
router.get('/products', async (req, res) => {
    try {
        const manager = await initializeBotManager();
        // Access the Delta service through the manager (we'll need to expose this)
        // For now, return the common perpetual products
        const products = [
            {
                id: 27,
                symbol: 'BTCUSD',
                description: 'Bitcoin Perpetual',
                contract_type: 'perpetual_futures',
                state: 'live',
                underlying_asset: { symbol: 'BTC' },
                quoting_asset: { symbol: 'USD' },
                settling_asset: { symbol: 'INR' }
            },
            {
                id: 3136,
                symbol: 'ETHUSD',
                description: 'Ethereum Perpetual',
                contract_type: 'perpetual_futures',
                state: 'live',
                underlying_asset: { symbol: 'ETH' },
                quoting_asset: { symbol: 'USD' },
                settling_asset: { symbol: 'INR' }
            }
        ];
        res.json({
            success: true,
            data: products,
            meta: {
                total: products.length,
                exchange: 'delta_exchange_india',
                environment: 'testnet'
            },
            timestamp: Date.now()
        });
    }
    catch (error) {
        logger_1.logger.error('Error getting products:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to get products',
            message: error instanceof Error ? error.message : 'Unknown error'
        });
    }
});
exports.default = router;
//# sourceMappingURL=deltaTradingRoutes.js.map