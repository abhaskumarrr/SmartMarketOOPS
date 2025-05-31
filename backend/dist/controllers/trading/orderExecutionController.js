"use strict";
/**
 * Order Execution Controller
 * Handles HTTP requests for order execution
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.orderExecutionController = void 0;
const uuid_1 = require("uuid");
const logger_1 = require("../../utils/logger");
// Import the order execution service
const orderExecutionService_1 = require("../../services/trading/orderExecutionService");
const smartOrderRouter_1 = __importDefault(require("../../services/trading/smartOrderRouter"));
const orderExecution_1 = require("../../types/orderExecution");
// Create logger
const logger = (0, logger_1.createLogger)('OrderExecutionController');
// Create order execution service instance
const orderExecutionService = new orderExecutionService_1.OrderExecutionService();
/**
 * Validates order request parameters
 * @param req - Express request
 * @returns Whether the request is valid and any error messages
 */
function validateOrderRequest(req) {
    const errors = [];
    // Required fields
    if (!req.body.symbol) {
        errors.push('Symbol is required');
    }
    if (!req.body.side) {
        errors.push('Side is required');
    }
    else if (!Object.values(orderExecution_1.OrderSide).includes(req.body.side)) {
        errors.push(`Side must be one of: ${Object.values(orderExecution_1.OrderSide).join(', ')}`);
    }
    if (!req.body.type) {
        errors.push('Order type is required');
    }
    else if (!Object.values(orderExecution_1.OrderType).includes(req.body.type)) {
        errors.push(`Order type must be one of: ${Object.values(orderExecution_1.OrderType).join(', ')}`);
    }
    if (req.body.quantity === undefined) {
        errors.push('Quantity is required');
    }
    else if (isNaN(Number(req.body.quantity)) || Number(req.body.quantity) <= 0) {
        errors.push('Quantity must be a positive number');
    }
    // Type-specific validation
    if (req.body.type === orderExecution_1.OrderType.LIMIT && req.body.price === undefined) {
        errors.push('Price is required for limit orders');
    }
    if (req.body.type === orderExecution_1.OrderType.STOP && req.body.stopPrice === undefined) {
        errors.push('Stop price is required for stop orders');
    }
    if (req.body.type === orderExecution_1.OrderType.STOP_LIMIT &&
        (req.body.price === undefined || req.body.stopPrice === undefined)) {
        errors.push('Price and stop price are required for stop-limit orders');
    }
    if (req.body.type === orderExecution_1.OrderType.TRAILING_STOP && req.body.trailingOffset === undefined) {
        errors.push('Trailing offset is required for trailing stop orders');
    }
    // Optional fields validation
    if (req.body.timeInForce && !Object.values(orderExecution_1.TimeInForce).includes(req.body.timeInForce)) {
        errors.push(`Time in force must be one of: ${Object.values(orderExecution_1.TimeInForce).join(', ')}`);
    }
    if (req.body.source && !Object.values(orderExecution_1.ExecutionSource).includes(req.body.source)) {
        errors.push(`Source must be one of: ${Object.values(orderExecution_1.ExecutionSource).join(', ')}`);
    }
    return {
        valid: errors.length === 0,
        errors
    };
}
/**
 * Maps request to execution request format
 * @param req - Express request
 * @returns Order execution request
 */
function mapToExecutionRequest(req) {
    // Extract user ID from authenticated request
    const userId = req.user?.id || '';
    return {
        symbol: req.body.symbol,
        type: req.body.type,
        side: req.body.side,
        quantity: Number(req.body.quantity),
        price: req.body.price !== undefined ? Number(req.body.price) : undefined,
        stopPrice: req.body.stopPrice !== undefined ? Number(req.body.stopPrice) : undefined,
        timeInForce: req.body.timeInForce || orderExecution_1.TimeInForce.GTC,
        clientOrderId: req.body.clientOrderId || (0, uuid_1.v4)(),
        positionId: req.body.positionId,
        strategyId: req.body.strategyId,
        botId: req.body.botId,
        signalId: req.body.signalId,
        source: req.body.source || orderExecution_1.ExecutionSource.MANUAL,
        userId,
        exchangeId: req.body.exchangeId || 'auto',
        reduceOnly: req.body.reduceOnly || false,
        postOnly: req.body.postOnly || false,
        leverage: req.body.leverage !== undefined ? Number(req.body.leverage) : undefined,
        marginType: req.body.marginType,
        takeProfitPrice: req.body.takeProfitPrice !== undefined ? Number(req.body.takeProfitPrice) : undefined,
        stopLossPrice: req.body.stopLossPrice !== undefined ? Number(req.body.stopLossPrice) : undefined,
        trailingOffset: req.body.trailingOffset !== undefined ? Number(req.body.trailingOffset) : undefined,
        notes: req.body.notes,
        metadata: req.body.metadata
    };
}
/**
 * Maps request to execution options
 * @param req - Express request
 * @returns Order execution options
 */
function mapToExecutionOptions(req) {
    return {
        validateOnly: req.body.validateOnly || false,
        dryRun: req.body.dryRun || false,
        smartRouting: req.body.smartRouting !== undefined ? req.body.smartRouting : true,
        retry: req.body.retry,
        timeout: req.body.timeout,
        slicer: req.body.slicer,
        notifications: req.body.notifications
    };
}
/**
 * Order Execution Controller
 */
exports.orderExecutionController = {
    /**
     * Execute an order
     * POST /api/orders/execute
     */
    async executeOrder(req, res) {
        try {
            logger.info('Execute order request received', { body: req.body });
            // Validate request
            const validation = validateOrderRequest(req);
            if (!validation.valid) {
                logger.warn('Invalid order request', { errors: validation.errors, body: req.body });
                res.status(400).json({
                    success: false,
                    error: {
                        code: 'INVALID_REQUEST',
                        message: 'Invalid order request',
                        details: validation.errors
                    },
                    timestamp: new Date().toISOString()
                });
                return;
            }
            // Map request to execution request
            const request = mapToExecutionRequest(req);
            const options = mapToExecutionOptions(req);
            // Route to best exchange if auto-routing is enabled
            if (request.exchangeId === 'auto' && options.smartRouting) {
                request.exchangeId = await smartOrderRouter_1.default.route(request);
            }
            // Handle order slicing for large orders
            if (options.slicer?.enabled) {
                const slicedOrders = await smartOrderRouter_1.default.splitOrder(request);
                const results = [];
                // Execute each slice
                for (const slicedOrder of slicedOrders) {
                    const result = await orderExecutionService.executeOrder(slicedOrder, options);
                    results.push(result);
                }
                res.status(200).json({
                    success: true,
                    data: {
                        orders: results,
                        totalSlices: results.length
                    },
                    timestamp: new Date().toISOString()
                });
            }
            else {
                // Execute single order
                const result = await orderExecutionService.executeOrder(request, options);
                res.status(200).json({
                    success: true,
                    data: result,
                    timestamp: new Date().toISOString()
                });
            }
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Order execution failed: ${errorMessage}`, {
                error: errorMessage,
                body: req.body
            });
            res.status(500).json({
                success: false,
                error: {
                    code: 'EXECUTION_FAILED',
                    message: 'Order execution failed',
                    details: errorMessage
                },
                timestamp: new Date().toISOString()
            });
        }
    },
    /**
     * Cancel an order
     * POST /api/orders/:id/cancel
     */
    async cancelOrder(req, res) {
        try {
            const orderId = req.params.id;
            const exchangeId = req.query.exchangeId || 'delta';
            const userId = req.user?.id || '';
            logger.info(`Cancel order request received for order ${orderId}`);
            const result = await orderExecutionService.cancelOrder(orderId, exchangeId, userId);
            res.status(200).json({
                success: result,
                data: {
                    orderId,
                    cancelled: result
                },
                timestamp: new Date().toISOString()
            });
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Order cancellation failed: ${errorMessage}`, {
                error: errorMessage,
                orderId: req.params.id
            });
            res.status(500).json({
                success: false,
                error: {
                    code: 'CANCELLATION_FAILED',
                    message: 'Order cancellation failed',
                    details: errorMessage
                },
                timestamp: new Date().toISOString()
            });
        }
    },
    /**
     * Get order details
     * GET /api/orders/:id
     */
    async getOrder(req, res) {
        try {
            const orderId = req.params.id;
            const exchangeId = req.query.exchangeId || 'delta';
            const userId = req.user?.id || '';
            logger.info(`Get order request received for order ${orderId}`);
            const order = await orderExecutionService.getOrder(orderId, exchangeId, userId);
            res.status(200).json({
                success: true,
                data: order,
                timestamp: new Date().toISOString()
            });
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Get order failed: ${errorMessage}`, {
                error: errorMessage,
                orderId: req.params.id
            });
            res.status(500).json({
                success: false,
                error: {
                    code: 'GET_ORDER_FAILED',
                    message: 'Failed to get order details',
                    details: errorMessage
                },
                timestamp: new Date().toISOString()
            });
        }
    },
    /**
     * Get user orders
     * GET /api/orders
     */
    async getUserOrders(req, res) {
        try {
            const userId = req.user?.id || '';
            const status = req.query.status;
            logger.info(`Get user orders request received for user ${userId}`);
            const orders = await orderExecutionService.getOrdersByUser(userId, status);
            res.status(200).json({
                success: true,
                data: orders,
                timestamp: new Date().toISOString()
            });
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Get user orders failed: ${errorMessage}`, {
                error: errorMessage,
                userId: req.user?.id
            });
            res.status(500).json({
                success: false,
                error: {
                    code: 'GET_ORDERS_FAILED',
                    message: 'Failed to get user orders',
                    details: errorMessage
                },
                timestamp: new Date().toISOString()
            });
        }
    },
    /**
     * Get orders by position
     * GET /api/positions/:id/orders
     */
    async getPositionOrders(req, res) {
        try {
            const positionId = req.params.id;
            logger.info(`Get position orders request received for position ${positionId}`);
            const orders = await orderExecutionService.getOrdersByPosition(positionId);
            res.status(200).json({
                success: true,
                data: orders,
                timestamp: new Date().toISOString()
            });
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Get position orders failed: ${errorMessage}`, {
                error: errorMessage,
                positionId: req.params.id
            });
            res.status(500).json({
                success: false,
                error: {
                    code: 'GET_POSITION_ORDERS_FAILED',
                    message: 'Failed to get position orders',
                    details: errorMessage
                },
                timestamp: new Date().toISOString()
            });
        }
    },
    /**
     * Get orders by strategy
     * GET /api/strategies/:id/orders
     */
    async getStrategyOrders(req, res) {
        try {
            const strategyId = req.params.id;
            logger.info(`Get strategy orders request received for strategy ${strategyId}`);
            const orders = await orderExecutionService.getOrdersByStrategy(strategyId);
            res.status(200).json({
                success: true,
                data: orders,
                timestamp: new Date().toISOString()
            });
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Get strategy orders failed: ${errorMessage}`, {
                error: errorMessage,
                strategyId: req.params.id
            });
            res.status(500).json({
                success: false,
                error: {
                    code: 'GET_STRATEGY_ORDERS_FAILED',
                    message: 'Failed to get strategy orders',
                    details: errorMessage
                },
                timestamp: new Date().toISOString()
            });
        }
    },
    /**
     * Get orders by bot
     * GET /api/bots/:id/orders
     */
    async getBotOrders(req, res) {
        try {
            const botId = req.params.id;
            logger.info(`Get bot orders request received for bot ${botId}`);
            const orders = await orderExecutionService.getOrdersByBot(botId);
            res.status(200).json({
                success: true,
                data: orders,
                timestamp: new Date().toISOString()
            });
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Get bot orders failed: ${errorMessage}`, {
                error: errorMessage,
                botId: req.params.id
            });
            res.status(500).json({
                success: false,
                error: {
                    code: 'GET_BOT_ORDERS_FAILED',
                    message: 'Failed to get bot orders',
                    details: errorMessage
                },
                timestamp: new Date().toISOString()
            });
        }
    },
    /**
     * Get orders by signal
     * GET /api/signals/:id/orders
     */
    async getSignalOrders(req, res) {
        try {
            const signalId = req.params.id;
            logger.info(`Get signal orders request received for signal ${signalId}`);
            const orders = await orderExecutionService.getOrdersBySignal(signalId);
            res.status(200).json({
                success: true,
                data: orders,
                timestamp: new Date().toISOString()
            });
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Get signal orders failed: ${errorMessage}`, {
                error: errorMessage,
                signalId: req.params.id
            });
            res.status(500).json({
                success: false,
                error: {
                    code: 'GET_SIGNAL_ORDERS_FAILED',
                    message: 'Failed to get signal orders',
                    details: errorMessage
                },
                timestamp: new Date().toISOString()
            });
        }
    }
};
exports.default = exports.orderExecutionController;
//# sourceMappingURL=orderExecutionController.js.map