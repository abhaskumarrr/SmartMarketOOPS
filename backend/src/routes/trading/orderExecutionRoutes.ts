/**
 * Order Execution Routes
 * Routes for order execution
 */

import { Router } from 'express';
import { orderExecutionController } from '../../controllers/trading/orderExecutionController';
import { authenticateJWT } from '../../middleware/authMiddleware';

const router = Router();

// Apply authentication middleware to all routes
router.use(authenticateJWT);

// Execute order
router.post('/execute', orderExecutionController.executeOrder);

// Cancel order
router.post('/:id/cancel', orderExecutionController.cancelOrder);

// Get order details
router.get('/:id', orderExecutionController.getOrder);

// Get user orders
router.get('/user/:userId', orderExecutionController.getUserOrders);

// Get position orders
router.get('/position/:positionId', orderExecutionController.getPositionOrders);

// Get strategy orders
router.get('/strategy/:strategyId', orderExecutionController.getStrategyOrders);

// Get bot orders
router.get('/bot/:botId', orderExecutionController.getBotOrders);

// Get signal orders
router.get('/signal/:signalId', orderExecutionController.getSignalOrders);

export default router; 