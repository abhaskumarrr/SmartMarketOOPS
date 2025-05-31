/**
 * Delta Exchange API Controller
 * Handles Delta Exchange API interactions
 */
const prisma = require('../utils/prismaClient');
const { decrypt } = require('../utils/encryption');
const { createError } = require('../middleware/errorHandler');
const { DeltaExchangeService, createDefaultService, createService } = require('../services/deltaExchangeService');
/**
 * Helper function to get Delta API service for authenticated user
 * @param {Object} user - User object from request
 * @returns {Promise<DeltaExchangeService>} - DeltaApiService instance
 */
async function getDeltaApiService(user) {
    // Get the most recent API key for user
    const apiKeyRecord = await prisma.apiKey.findFirst({
        where: { userId: user.id },
        orderBy: { createdAt: 'desc' }
    });
    if (!apiKeyRecord) {
        // If no user-specific key is found, use the default (environment) credentials
        return createDefaultService();
    }
    // Decrypt API key data
    const apiKeyData = decrypt(apiKeyRecord.encryptedData);
    // Return Delta API service instance
    return createService(apiKeyData.apiKey, apiKeyData.apiSecret, apiKeyData.testnet);
}
/**
 * Get available products (markets)
 * @route GET /api/delta/products
 * @access Private
 */
const getProducts = async (req, res, next) => {
    try {
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchMarkets();
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch products: ${error.message}`, 500));
    }
};
/**
 * Get order book for a product
 * @route GET /api/delta/products/:id/orderbook
 * @access Private
 */
const getOrderBook = async (req, res, next) => {
    try {
        const { id } = req.params;
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchOrderBook(id);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch order book: ${error.message}`, 500));
    }
};
/**
 * Get recent trades for a product
 * @route GET /api/delta/products/:id/trades
 * @access Private
 */
const getRecentTrades = async (req, res, next) => {
    try {
        const { id } = req.params;
        const { limit } = req.query;
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchTrades(id, limit ? parseInt(limit) : 100);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch recent trades: ${error.message}`, 500));
    }
};
/**
 * Get account balance
 * @route GET /api/delta/balance
 * @access Private
 */
const getAccountBalance = async (req, res, next) => {
    try {
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchBalance();
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch account balance: ${error.message}`, 500));
    }
};
/**
 * Get positions
 * @route GET /api/delta/positions
 * @access Private
 */
const getPositions = async (req, res, next) => {
    try {
        const { symbol } = req.query;
        const deltaApi = await getDeltaApiService(req.user);
        let data;
        if (symbol) {
            data = await deltaApi.fetchPosition(symbol);
        }
        else {
            // For all positions, we can call fetchPosition without a symbol
            data = await deltaApi.request('GET', '/positions');
        }
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch positions: ${error.message}`, 500));
    }
};
/**
 * Get orders
 * @route GET /api/delta/orders
 * @access Private
 */
const getOrders = async (req, res, next) => {
    try {
        const { status, symbol } = req.query;
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchOpenOrders(symbol, { status: status || 'open' });
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch orders: ${error.message}`, 500));
    }
};
/**
 * Create order
 * @route POST /api/delta/orders
 * @access Private
 */
const createOrder = async (req, res, next) => {
    try {
        const { product_id, side, size, limit_price, order_type = 'limit_order', ...restOrderData } = req.body;
        // Validate required fields
        if (!product_id || !side || !size) {
            return next(createError('Missing required order fields: product_id, side, size', 400));
        }
        // For limit orders, validate limit_price
        if (order_type === 'limit_order' && !limit_price) {
            return next(createError('Missing required field for limit order: limit_price', 400));
        }
        const deltaApi = await getDeltaApiService(req.user);
        let data;
        const orderType = order_type === 'market_order' ? 'market' : 'limit';
        const orderSize = parseFloat(size);
        const orderPrice = limit_price ? parseFloat(limit_price) : undefined;
        data = await deltaApi.createOrder(product_id, orderType, side, orderSize, orderPrice, restOrderData);
        res.status(201).json({
            success: true,
            data
        });
    }
    catch (error) {
        // Try to parse error from Delta API
        let errorMessage = error.message;
        let statusCode = 500;
        try {
            const parsedError = JSON.parse(error.message);
            errorMessage = parsedError.message || error.message;
            statusCode = parsedError.status || 500;
        }
        catch (e) {
            // If can't parse, use original error
        }
        next(createError(`Failed to create order: ${errorMessage}`, statusCode));
    }
};
/**
 * Cancel order
 * @route DELETE /api/delta/orders/:id
 * @access Private
 */
const cancelOrder = async (req, res, next) => {
    try {
        const { id } = req.params;
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.cancelOrder(id);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to cancel order: ${error.message}`, 500));
    }
};
/**
 * Cancel all orders
 * @route DELETE /api/delta/orders
 * @access Private
 */
const cancelAllOrders = async (req, res, next) => {
    try {
        const filters = req.body; // Can include product_id, etc.
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.request('DELETE', '/orders', filters);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to cancel all orders: ${error.message}`, 500));
    }
};
/**
 * Get order history
 * @route GET /api/delta/orders/history
 * @access Private
 */
const getOrderHistory = async (req, res, next) => {
    try {
        const { symbol, limit, offset, status } = req.query;
        const params = {
            ...(limit ? { limit: parseInt(limit) } : {}),
            ...(offset ? { offset: parseInt(offset) } : {}),
            ...(status ? { status } : {})
        };
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchOrderHistory(symbol, params);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch order history: ${error.message}`, 500));
    }
};
/**
 * Get trade history
 * @route GET /api/delta/fills
 * @access Private
 */
const getTradeHistory = async (req, res, next) => {
    try {
        const { symbol, limit, offset, order_id } = req.query;
        const params = {
            ...(limit ? { limit: parseInt(limit) } : {}),
            ...(offset ? { offset: parseInt(offset) } : {}),
            ...(order_id ? { order_id } : {})
        };
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchMyTrades(symbol, params);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch trade history: ${error.message}`, 500));
    }
};
/**
 * Get comprehensive market data
 * @route GET /api/delta/market-data
 * @access Private
 */
const getMarketData = async (req, res, next) => {
    try {
        const { symbol = 'BTCUSDT' } = req.query;
        // This is a custom endpoint that combines multiple Delta API calls
        // to provide comprehensive market data in a single request
        // Validate symbol
        if (!symbol) {
            return next(createError('Symbol is required', 400));
        }
        const deltaApi = await getDeltaApiService(req.user);
        // Execute multiple requests in parallel
        const [ticker, orderbook, recentTrades, fundingRate] = await Promise.all([
            deltaApi.fetchTicker(symbol),
            deltaApi.fetchOrderBook(symbol),
            deltaApi.fetchTrades(symbol, 10),
            deltaApi.fetchFundingRates([symbol])
        ]);
        res.status(200).json({
            success: true,
            data: {
                symbol,
                ticker,
                orderbook,
                recentTrades,
                fundingRate
            }
        });
    }
    catch (error) {
        next(createError(`Failed to fetch market data: ${error.message}`, 500));
    }
};
/**
 * Get leverage settings
 * @route GET /api/delta/leverage
 * @access Private
 */
const getLeverage = async (req, res, next) => {
    try {
        const { symbol } = req.query;
        if (!symbol) {
            return next(createError('Symbol is required', 400));
        }
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchLeverage(symbol);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch leverage: ${error.message}`, 500));
    }
};
/**
 * Set leverage
 * @route POST /api/delta/leverage
 * @access Private
 */
const setLeverage = async (req, res, next) => {
    try {
        const { symbol, leverage } = req.body;
        if (!symbol || !leverage) {
            return next(createError('Symbol and leverage are required', 400));
        }
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.setLeverage(parseFloat(leverage), symbol);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to set leverage: ${error.message}`, 500));
    }
};
/**
 * Get available currencies
 * @route GET /api/delta/currencies
 * @access Private
 */
const getCurrencies = async (req, res, next) => {
    try {
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.fetchCurrencies();
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to fetch currencies: ${error.message}`, 500));
    }
};
/**
 * Close all positions
 * @route POST /api/delta/positions/close-all
 * @access Private
 */
const closeAllPositions = async (req, res, next) => {
    try {
        const params = req.body;
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.closeAllPositions(params);
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to close all positions: ${error.message}`, 500));
    }
};
/**
 * Add margin to position
 * @route POST /api/delta/positions/add-margin
 * @access Private
 */
const addMargin = async (req, res, next) => {
    try {
        const { symbol, amount } = req.body;
        if (!symbol || !amount) {
            return next(createError('Symbol and amount are required', 400));
        }
        const deltaApi = await getDeltaApiService(req.user);
        const data = await deltaApi.addMargin(symbol, parseFloat(amount));
        res.status(200).json({
            success: true,
            data
        });
    }
    catch (error) {
        next(createError(`Failed to add margin: ${error.message}`, 500));
    }
};
module.exports = {
    getProducts,
    getOrderBook,
    getRecentTrades,
    getAccountBalance,
    getPositions,
    createOrder,
    cancelOrder,
    getOrders,
    cancelAllOrders,
    getOrderHistory,
    getTradeHistory,
    getMarketData,
    getLeverage,
    setLeverage,
    getCurrencies,
    closeAllPositions,
    addMargin
};
//# sourceMappingURL=deltaApiController.js.map