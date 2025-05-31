"use strict";
/**
 * Order Execution Service
 * Handles order execution across different exchanges
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OrderExecutionService = void 0;
const prismaClient_1 = __importDefault(require("../../utils/prismaClient"));
const logger_1 = require("../../utils/logger");
const orderExecution_1 = require("../../types/orderExecution");
// Import risk management service
const riskManagementService_1 = require("./riskManagementService");
// Import deltaExchangeService with require since it's a JS file
const { createDefaultService } = require('../../services/deltaExchangeService');
// Create logger
const logger = (0, logger_1.createLogger)('OrderExecutionService');
/**
 * Order Execution Service class
 * Handles order execution across different exchanges
 */
class OrderExecutionService {
    /**
     * Create a new Order Execution Service
     */
    constructor() {
        // Map of exchange connectors
        this.exchangeConnectors = new Map();
        logger.info('Order Execution Service initialized');
        // Initialize exchange connectors
        this.registerExchangeConnector('delta', createDefaultService());
        // Initialize risk management service
        this.riskManagementService = new riskManagementService_1.RiskManagementService();
    }
    /**
     * Register an exchange connector
     * @param exchangeId - Exchange identifier
     * @param connector - Exchange connector
     */
    registerExchangeConnector(exchangeId, connector) {
        this.exchangeConnectors.set(exchangeId, connector);
        logger.info(`Registered exchange connector: ${exchangeId}`);
    }
    /**
     * Validate an order request
     * @param request - Order execution request
     * @returns Validation error message or null if valid
     */
    validateOrderRequest(request) {
        // Check required fields
        if (!request.symbol) {
            return 'Symbol is required';
        }
        if (!request.side) {
            return 'Side is required';
        }
        if (!request.type) {
            return 'Order type is required';
        }
        if (request.quantity <= 0) {
            return 'Quantity must be positive';
        }
        // Type-specific validation
        if (request.type === orderExecution_1.OrderType.LIMIT && request.price === undefined) {
            return 'Price is required for limit orders';
        }
        if (request.type === orderExecution_1.OrderType.STOP && request.stopPrice === undefined) {
            return 'Stop price is required for stop orders';
        }
        if (request.type === orderExecution_1.OrderType.STOP_LIMIT &&
            (request.price === undefined || request.stopPrice === undefined)) {
            return 'Price and stop price are required for stop-limit orders';
        }
        if (request.type === orderExecution_1.OrderType.TRAILING_STOP && request.trailingOffset === undefined) {
            return 'Trailing offset is required for trailing stop orders';
        }
        return null;
    }
    /**
     * Perform risk check for an order
     * @param request - Order execution request
     * @returns Result of risk check
     */
    async performRiskCheck(request) {
        try {
            // Get risk settings for user/bot
            const riskSettings = await this.riskManagementService.getRiskSettings(request.userId, request.botId);
            // Check position sizing
            const positionSizingRequest = {
                userId: request.userId,
                botId: request.botId,
                symbol: request.symbol,
                direction: request.side === orderExecution_1.OrderSide.BUY ? 'long' : 'short',
                entryPrice: request.price || 0, // Will be updated with market price if 0
                stopLossPrice: request.stopLossPrice,
                stopLossPercentage: undefined, // Will be calculated if not provided
                confidence: 50 // Default confidence
            };
            // For market orders, get current price
            if (request.type === orderExecution_1.OrderType.MARKET || positionSizingRequest.entryPrice === 0) {
                const connector = this.exchangeConnectors.get(request.exchangeId);
                if (connector) {
                    const ticker = await connector.fetchTicker(request.symbol);
                    positionSizingRequest.entryPrice = request.side === orderExecution_1.OrderSide.BUY
                        ? ticker.askPrice || ticker.lastPrice
                        : ticker.bidPrice || ticker.lastPrice;
                }
            }
            // Calculate maximum position size based on risk
            const positionSizing = await this.riskManagementService.calculatePositionSize(positionSizingRequest);
            // For simplicity, we'll assume the position sizing result has these properties
            // In a real implementation, you would properly type this
            const sizeLimit = positionSizing.recommendedSize || positionSizing.maxPositionSize || 0;
            const leverageLimit = positionSizing.leverage || 10; // Default max leverage
            const accountBalance = positionSizing.accountBalance || 0;
            const riskAmount = positionSizing.riskAmount || 0;
            // Check if requested quantity exceeds max size
            if (request.quantity > sizeLimit) {
                return {
                    passed: false,
                    reason: `Order quantity exceeds maximum allowed size (${sizeLimit})`,
                    details: {
                        maxSize: sizeLimit,
                        requestedSize: request.quantity,
                        riskAmount,
                        accountBalance
                    }
                };
            }
            // Check leverage
            if (request.leverage && request.leverage > leverageLimit) {
                return {
                    passed: false,
                    reason: `Requested leverage (${request.leverage}x) exceeds maximum allowed (${leverageLimit}x)`,
                    details: {
                        maxLeverage: leverageLimit,
                        requestedLeverage: request.leverage
                    }
                };
            }
            // Additional risk checks could be added here
            return { passed: true };
        }
        catch (error) {
            logger.error(`Error during risk check: ${error instanceof Error ? error.message : String(error)}`, {
                error,
                request
            });
            // Fail safe - if risk check errors, reject the order
            return {
                passed: false,
                reason: 'Risk check failed due to internal error',
                details: { error: error instanceof Error ? error.message : String(error) }
            };
        }
    }
    /**
     * Map internal order request to exchange-specific format
     * @param request - Order execution request
     * @returns Exchange-specific request
     */
    mapToExchangeRequest(request) {
        // Convert order type to exchange format
        let exchangeType = request.type.toLowerCase();
        // Handle special cases
        if (request.type === orderExecution_1.OrderType.STOP_LIMIT) {
            exchangeType = 'stop_limit';
        }
        else if (request.type === orderExecution_1.OrderType.TRAILING_STOP) {
            exchangeType = 'trailing_stop';
        }
        // Convert side to exchange format
        const exchangeSide = request.side.toLowerCase();
        // Build params object for exchange-specific parameters
        const params = {};
        // Add time in force if specified
        if (request.timeInForce) {
            params.timeInForce = request.timeInForce;
        }
        // Add client order ID if specified
        if (request.clientOrderId) {
            params.clientOrderId = request.clientOrderId;
        }
        // Add leverage if specified
        if (request.leverage) {
            params.leverage = request.leverage;
        }
        // Add margin type if specified
        if (request.marginType) {
            params.marginType = request.marginType;
        }
        // Add reduce only flag if specified
        if (request.reduceOnly) {
            params.reduceOnly = request.reduceOnly;
        }
        // Add post only flag if specified
        if (request.postOnly) {
            params.postOnly = request.postOnly;
        }
        // Add stop loss if specified
        if (request.stopLossPrice) {
            params.stopLoss = {
                price: request.stopLossPrice
            };
        }
        // Add take profit if specified
        if (request.takeProfitPrice) {
            params.takeProfit = {
                price: request.takeProfitPrice
            };
        }
        // Add trailing offset if specified
        if (request.trailingOffset) {
            params.trailingOffset = request.trailingOffset;
        }
        return {
            type: exchangeType,
            side: exchangeSide,
            quantity: request.quantity,
            price: request.price,
            stopPrice: request.stopPrice,
            params
        };
    }
    /**
     * Map exchange response to order execution result
     * @param response - Exchange response
     * @param request - Original order request
     * @returns Order execution result
     */
    mapExchangeResponseToResult(response, request) {
        // Get current timestamp
        const now = new Date();
        const nowString = now.toISOString();
        // Map status
        let status = orderExecution_1.OrderExecutionStatus.SUBMITTED;
        if (response.status === 'filled') {
            status = orderExecution_1.OrderExecutionStatus.FILLED;
        }
        else if (response.status === 'partially_filled') {
            status = orderExecution_1.OrderExecutionStatus.PARTIALLY_FILLED;
        }
        else if (response.status === 'canceled' || response.status === 'cancelled') {
            status = orderExecution_1.OrderExecutionStatus.CANCELLED;
        }
        else if (response.status === 'rejected') {
            status = orderExecution_1.OrderExecutionStatus.REJECTED;
        }
        else if (response.status === 'expired') {
            status = orderExecution_1.OrderExecutionStatus.EXPIRED;
        }
        // Calculate filled and remaining quantities
        const filledQuantity = response.filled || 0;
        const remainingQuantity = Math.max(0, request.quantity - filledQuantity);
        // Use helper function to create result with proper typing
        return this.createOrderResult({
            status,
            symbol: request.symbol,
            type: request.type,
            side: request.side,
            quantity: request.quantity,
            price: request.price,
            stopPrice: request.stopPrice,
            avgFillPrice: response.average || response.price || request.price,
            filledQuantity,
            remainingQuantity,
            fee: response.fee?.cost,
            feeCurrency: response.fee?.currency,
            clientOrderId: request.clientOrderId || response.clientOrderId,
            exchangeOrderId: response.id,
            positionId: request.positionId,
            strategyId: request.strategyId,
            botId: request.botId,
            signalId: request.signalId,
            source: request.source,
            userId: request.userId,
            exchangeId: request.exchangeId,
            exchangeTimestamp: response.timestamp ? new Date(response.timestamp).toISOString() : undefined,
            submittedAt: nowString,
            updatedAt: nowString,
            raw: response
        });
    }
    /**
     * Store order in database
     * @param order - Order execution result
     */
    async storeOrder(order) {
        try {
            logger.info(`Storing order: ${order.id}`, { order });
            // Map error to database format
            const errorData = order.error ? {
                errorCode: order.error.code,
                errorMessage: order.error.message,
                errorDetails: order.error.details ? order.error.details : undefined
            } : {};
            // Convert timestamps to Date objects for Prisma
            const submittedAt = new Date(order.submittedAt);
            const updatedAt = new Date(order.updatedAt);
            const completedAt = order.completedAt ? new Date(order.completedAt) : undefined;
            // Store the order in the database
            await prismaClient_1.default.order.create({
                data: {
                    id: order.id,
                    status: order.status,
                    symbol: order.symbol,
                    type: order.type,
                    side: order.side,
                    quantity: order.quantity,
                    price: order.price,
                    stopPrice: order.stopPrice,
                    avgFillPrice: order.avgFillPrice,
                    filledQuantity: order.filledQuantity,
                    remainingQuantity: order.remainingQuantity,
                    fee: order.fee,
                    feeCurrency: order.feeCurrency,
                    clientOrderId: order.clientOrderId,
                    exchangeOrderId: order.exchangeOrderId,
                    source: order.source,
                    exchangeId: order.exchangeId,
                    submittedAt,
                    updatedAt,
                    completedAt,
                    raw: order.raw ? order.raw : undefined,
                    ...errorData,
                    // Connect relations
                    user: {
                        connect: {
                            id: order.userId
                        }
                    },
                    ...(order.positionId ? {
                        position: {
                            connect: {
                                id: order.positionId
                            }
                        }
                    } : {}),
                    ...(order.strategyId ? {
                        strategy: {
                            connect: {
                                id: order.strategyId
                            }
                        }
                    } : {}),
                    ...(order.botId ? {
                        bot: {
                            connect: {
                                id: order.botId
                            }
                        }
                    } : {}),
                    ...(order.signalId ? {
                        signal: {
                            connect: {
                                id: order.signalId
                            }
                        }
                    } : {})
                }
            });
            logger.info(`Order stored successfully: ${order.id}`);
        }
        catch (error) {
            logger.error(`Failed to store order: ${error instanceof Error ? error.message : String(error)}`, {
                error,
                order
            });
            // Don't throw error - just log it, so the order execution can still succeed
        }
    }
    /**
     * Execute an order
     * @param request - Order execution request
     * @param options - Order execution options
     * @returns Order execution result
     */
    async executeOrder(request, options = {}) {
        try {
            logger.info(`Executing order for ${request.symbol}`, { request, options });
            // Validate request
            const validationError = this.validateOrderRequest(request);
            if (validationError) {
                logger.warn(`Order validation failed: ${validationError}`, { request });
                return this.createOrderResult({
                    status: orderExecution_1.OrderExecutionStatus.REJECTED,
                    symbol: request.symbol,
                    type: request.type,
                    side: request.side,
                    quantity: request.quantity,
                    filledQuantity: 0,
                    remainingQuantity: request.quantity,
                    clientOrderId: request.clientOrderId,
                    source: request.source,
                    userId: request.userId,
                    exchangeId: request.exchangeId,
                    error: {
                        code: 'VALIDATION_FAILED',
                        message: validationError
                    }
                });
            }
            // Perform risk check
            const riskCheck = await this.performRiskCheck(request);
            if (!riskCheck.passed) {
                logger.warn(`Risk check failed: ${riskCheck.reason}`, { request, riskCheck });
                return this.createOrderResult({
                    status: orderExecution_1.OrderExecutionStatus.REJECTED,
                    symbol: request.symbol,
                    type: request.type,
                    side: request.side,
                    quantity: request.quantity,
                    filledQuantity: 0,
                    remainingQuantity: request.quantity,
                    clientOrderId: request.clientOrderId,
                    source: request.source,
                    userId: request.userId,
                    exchangeId: request.exchangeId,
                    error: {
                        code: 'RISK_CHECK_FAILED',
                        message: riskCheck.reason,
                        details: riskCheck.details
                    }
                });
            }
            // Check if it's dry run or validate only
            if (options.validateOnly || options.dryRun) {
                logger.info(`Skipping actual execution (${options.validateOnly ? 'validateOnly' : 'dryRun'})`, { request });
                return this.createOrderResult({
                    status: options.validateOnly ? orderExecution_1.OrderExecutionStatus.PENDING : orderExecution_1.OrderExecutionStatus.SUBMITTED,
                    symbol: request.symbol,
                    type: request.type,
                    side: request.side,
                    quantity: request.quantity,
                    price: request.price,
                    stopPrice: request.stopPrice,
                    clientOrderId: request.clientOrderId,
                    positionId: request.positionId,
                    strategyId: request.strategyId,
                    botId: request.botId,
                    signalId: request.signalId,
                    source: request.source,
                    userId: request.userId,
                    exchangeId: request.exchangeId
                });
            }
            // Get exchange connector
            const connector = this.exchangeConnectors.get(request.exchangeId);
            if (!connector) {
                throw new Error(`Unknown exchange: ${request.exchangeId}`);
            }
            // Map internal order type to exchange order type
            const exchangeRequest = this.mapToExchangeRequest(request);
            // Retry logic
            const maxAttempts = options.retry?.maxAttempts || 1;
            const retryInterval = options.retry?.interval || 1000;
            let lastError = null;
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    // Execute order on exchange
                    const response = await connector.createOrder(request.symbol, exchangeRequest.type, exchangeRequest.side, exchangeRequest.quantity, exchangeRequest.price, exchangeRequest.params);
                    // Map exchange response to result
                    const result = this.mapExchangeResponseToResult(response, request);
                    // Store order in database
                    await this.storeOrder(result);
                    logger.info(`Order executed successfully`, { result });
                    return result;
                }
                catch (error) {
                    lastError = error;
                    if (attempt < maxAttempts) {
                        logger.warn(`Order execution attempt ${attempt} failed, retrying...`, {
                            error: error instanceof Error ? error.message : String(error),
                            request
                        });
                        await new Promise(resolve => setTimeout(resolve, retryInterval));
                    }
                }
            }
            // If we reached here, all attempts failed
            const errorMessage = lastError instanceof Error ? lastError.message : String(lastError);
            logger.error(`Order execution failed after ${maxAttempts} attempts: ${errorMessage}`, {
                error: errorMessage,
                request
            });
            const timestamp = new Date().toISOString();
            return this.createOrderResult({
                status: orderExecution_1.OrderExecutionStatus.REJECTED,
                symbol: request.symbol,
                type: request.type,
                side: request.side,
                quantity: request.quantity,
                filledQuantity: 0,
                remainingQuantity: request.quantity,
                clientOrderId: request.clientOrderId,
                source: request.source,
                userId: request.userId,
                exchangeId: request.exchangeId,
                error: {
                    code: 'EXECUTION_FAILED',
                    message: errorMessage,
                    details: lastError
                }
            });
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Unexpected error during order execution: ${errorMessage}`, {
                error: errorMessage,
                request
            });
            const timestamp = new Date().toISOString();
            return this.createOrderResult({
                status: orderExecution_1.OrderExecutionStatus.REJECTED,
                symbol: request.symbol,
                type: request.type,
                side: request.side,
                quantity: request.quantity,
                filledQuantity: 0,
                remainingQuantity: request.quantity,
                clientOrderId: request.clientOrderId,
                source: request.source,
                userId: request.userId,
                exchangeId: request.exchangeId,
                error: {
                    code: 'UNEXPECTED_ERROR',
                    message: errorMessage
                }
            });
        }
    }
    /**
     * Cancel an order
     * @param orderId - Order ID
     * @param exchangeId - Exchange ID
     * @param userId - User ID
     * @returns Whether cancellation was successful
     */
    async cancelOrder(orderId, exchangeId, userId) {
        try {
            logger.info(`Cancelling order ${orderId} on ${exchangeId} for user ${userId}`);
            // Get order from database
            // For now, just use a hardcoded response
            // In a real implementation, this would query the database
            // const order = await prisma.order.findUnique({ where: { id: orderId } });
            // For now, assume the order exists and has symbol 'BTC/USDT'
            const symbol = 'BTC/USDT';
            // Get exchange connector
            const connector = this.exchangeConnectors.get(exchangeId);
            if (!connector) {
                throw new Error(`Unknown exchange: ${exchangeId}`);
            }
            // Cancel order on exchange
            const result = await connector.cancelOrder(orderId, { symbol });
            // Update order status in database
            // In a real implementation, this would update the database
            // await prisma.order.update({
            //   where: { id: orderId },
            //   data: { status: OrderExecutionStatus.CANCELLED, updatedAt: new Date() }
            // });
            logger.info(`Order ${orderId} cancelled successfully`);
            return true;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to cancel order ${orderId}: ${errorMessage}`, {
                error: errorMessage,
                orderId,
                exchangeId,
                userId
            });
            return false;
        }
    }
    /**
     * Get order details
     * @param orderId - Order ID
     * @param exchangeId - Exchange ID
     * @param userId - User ID
     * @returns Order execution result
     */
    async getOrder(orderId, exchangeId, userId) {
        try {
            logger.info(`Getting order ${orderId} from ${exchangeId} for user ${userId}`);
            // Get exchange connector
            const connector = this.exchangeConnectors.get(exchangeId);
            if (!connector) {
                throw new Error(`Unknown exchange: ${exchangeId}`);
            }
            // Fetch order from exchange
            const response = await connector.fetchOrder(orderId);
            // Map response to result
            // Since we don't have the original request, we'll create a partial one based on the response
            const partialRequest = {
                symbol: response.symbol,
                type: this.mapExchangeOrderTypeToInternal(response.type),
                side: response.side === 'buy' ? orderExecution_1.OrderSide.BUY : orderExecution_1.OrderSide.SELL,
                quantity: response.amount,
                price: response.price,
                clientOrderId: response.clientOrderId,
                userId,
                exchangeId
            };
            const result = this.mapExchangeResponseToResult(response, partialRequest);
            logger.info(`Order ${orderId} fetched successfully`);
            return result;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to get order ${orderId}: ${errorMessage}`, {
                error: errorMessage,
                orderId,
                exchangeId,
                userId
            });
            throw new Error(`Failed to get order: ${errorMessage}`);
        }
    }
    /**
     * Get orders by user
     * @param userId - User ID
     * @param status - Optional order status filter
     * @returns Array of order execution results
     */
    async getOrdersByUser(userId, status) {
        try {
            logger.info(`Getting orders for user ${userId}${status ? ` with status ${status.join(', ')}` : ''}`);
            // Get exchange connectors for user
            // In a real implementation, this would query the database to get the user's exchanges
            const exchangeIds = ['delta']; // Hardcoded for now
            const orders = [];
            for (const exchangeId of exchangeIds) {
                const connector = this.exchangeConnectors.get(exchangeId);
                if (!connector) {
                    logger.warn(`Unknown exchange: ${exchangeId}`);
                    continue;
                }
                try {
                    // Fetch open orders from exchange
                    const openOrders = await connector.fetchOpenOrders();
                    // Fetch order history from exchange
                    const orderHistory = await connector.fetchOrderHistory();
                    // Combine and map results
                    const allOrders = [...openOrders, ...orderHistory];
                    for (const order of allOrders) {
                        const partialRequest = {
                            symbol: order.symbol,
                            type: this.mapExchangeOrderTypeToInternal(order.type),
                            side: order.side === 'buy' ? orderExecution_1.OrderSide.BUY : orderExecution_1.OrderSide.SELL,
                            quantity: order.amount,
                            price: order.price,
                            clientOrderId: order.clientOrderId,
                            userId,
                            exchangeId
                        };
                        const result = this.mapExchangeResponseToResult(order, partialRequest);
                        // Apply status filter if provided
                        if (!status || status.includes(result.status)) {
                            orders.push(result);
                        }
                    }
                }
                catch (error) {
                    const errorMessage = error instanceof Error ? error.message : String(error);
                    logger.warn(`Failed to fetch orders from ${exchangeId}: ${errorMessage}`, {
                        error: errorMessage,
                        exchangeId,
                        userId
                    });
                    // Continue with other exchanges
                }
            }
            logger.info(`Retrieved ${orders.length} orders for user ${userId}${status ? ` with status ${status.join(', ')}` : ''}`);
            return orders;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to get orders for user ${userId}: ${errorMessage}`, {
                error: errorMessage,
                userId
            });
            throw new Error(`Failed to get orders: ${errorMessage}`);
        }
    }
}
exports.OrderExecutionService = OrderExecutionService;
//# sourceMappingURL=orderExecutionService.js.map