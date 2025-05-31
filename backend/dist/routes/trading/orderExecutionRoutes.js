"use strict";
/**
 * Order Execution Routes
 * Routes for order execution
 */
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const orderExecutionController_1 = require("../../controllers/trading/orderExecutionController");
const authMiddleware_1 = require("../../middleware/authMiddleware");
const router = (0, express_1.Router)();
// Apply authentication middleware to all routes
router.use(authMiddleware_1.authenticateJWT);
// Execute order
router.post('/execute', orderExecutionController_1.orderExecutionController.executeOrder);
// Cancel order
router.post('/:id/cancel', orderExecutionController_1.orderExecutionController.cancelOrder);
// Get order details
router.get('/:id', orderExecutionController_1.orderExecutionController.getOrder);
// Get user orders
router.get('/user/:userId', orderExecutionController_1.orderExecutionController.getUserOrders);
// Get position orders
router.get('/position/:positionId', orderExecutionController_1.orderExecutionController.getPositionOrders);
// Get strategy orders
router.get('/strategy/:strategyId', orderExecutionController_1.orderExecutionController.getStrategyOrders);
// Get bot orders
router.get('/bot/:botId', orderExecutionController_1.orderExecutionController.getBotOrders);
// Get signal orders
router.get('/signal/:signalId', orderExecutionController_1.orderExecutionController.getSignalOrders);
exports.default = router;
//# sourceMappingURL=orderExecutionRoutes.js.map