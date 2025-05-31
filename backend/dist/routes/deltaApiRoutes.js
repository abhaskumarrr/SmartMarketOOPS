/**
 * Delta Exchange API Routes
 * Endpoints for interacting with the Delta Exchange API
 */
const express = require('express');
const router = express.Router();
const { getProducts, getOrderBook, getRecentTrades, getAccountBalance, getPositions, createOrder, cancelOrder, getOrders, cancelAllOrders, getOrderHistory, getTradeHistory, getMarketData, getLeverage, setLeverage, getCurrencies, closeAllPositions, addMargin } = require('../controllers/deltaApiController');
const { auth } = require('../middleware/auth');
// All routes require authentication
router.use(auth);
// Market data endpoints
router.get('/products', getProducts);
router.get('/products/:id/orderbook', getOrderBook);
router.get('/products/:id/trades', getRecentTrades);
router.get('/market-data', getMarketData);
router.get('/currencies', getCurrencies);
// Account endpoints
router.get('/balance', getAccountBalance);
router.get('/positions', getPositions);
// Position management
router.post('/positions/close-all', closeAllPositions);
router.post('/positions/add-margin', addMargin);
// Leverage management
router.get('/leverage', getLeverage);
router.post('/leverage', setLeverage);
// Order endpoints
router.get('/orders', getOrders);
router.post('/orders', createOrder);
router.delete('/orders/:id', cancelOrder);
router.delete('/orders', cancelAllOrders);
router.get('/orders/history', getOrderHistory);
router.get('/fills', getTradeHistory);
module.exports = router;
//# sourceMappingURL=deltaApiRoutes.js.map