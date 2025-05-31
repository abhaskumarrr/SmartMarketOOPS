"use strict";
/**
 * Smart Order Router
 * Routes orders to the best exchange based on price and liquidity
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SmartOrderRouter = void 0;
const uuid_1 = require("uuid");
const logger_1 = require("../../utils/logger");
const orderExecution_1 = require("../../types/orderExecution");
// Import deltaExchangeService properly with require
const { createDefaultService } = require('../../services/deltaExchangeService');
// Create logger
const logger = (0, logger_1.createLogger)('SmartOrderRouter');
/**
 * Smart Order Router
 * Routes orders to the best exchange based on price and liquidity
 */
class SmartOrderRouter {
    /**
     * Create a new Smart Order Router
     */
    constructor() {
        // Map of exchange connectors
        this.exchangeConnectors = new Map();
        // Exchange priority - used as fallback when prices are identical
        this.exchangePriority = ['delta', 'binance', 'kraken'];
        logger.info('Smart Order Router initialized');
        // Register exchange connectors
        this.registerExchangeConnector('delta', createDefaultService());
        // Refresh exchange data periodically
        setInterval(() => this.refreshExchangeData(), 60000); // Every minute
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
     * Route an order to the best exchange
     * @param request - Order execution request
     * @returns Exchange ID
     */
    async route(request) {
        try {
            logger.info(`Routing order for ${request.symbol}`, { request });
            // If exchange ID is already specified, use it
            if (request.exchangeId && request.exchangeId !== 'auto') {
                return request.exchangeId;
            }
            // Get best exchange
            const bestExchange = await this.getBestExchange(request.symbol, request.side, request.quantity);
            logger.info(`Best exchange for ${request.symbol}: ${bestExchange.exchangeId}`, {
                bestExchange,
                request
            });
            return bestExchange.exchangeId;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Order routing failed: ${errorMessage}`, {
                error: errorMessage,
                request
            });
            // Fallback to default exchange
            return this.exchangePriority[0];
        }
    }
    /**
     * Get the best exchange for an order
     * @param symbol - Trading symbol
     * @param side - Order side
     * @param quantity - Order quantity
     * @returns Best exchange information
     */
    async getBestExchange(symbol, side, quantity) {
        try {
            logger.info(`Finding best exchange for ${symbol}`, { symbol, side, quantity });
            // Get all available exchanges
            const exchanges = Array.from(this.exchangeConnectors.keys());
            // Get prices from all exchanges
            const exchangePrices = await Promise.all(exchanges.map(async (exchangeId) => {
                try {
                    const connector = this.exchangeConnectors.get(exchangeId);
                    const ticker = await connector.getTicker(symbol);
                    // Get the relevant price based on side
                    const price = side === orderExecution_1.OrderSide.BUY ? ticker.askPrice : ticker.bidPrice;
                    // Get fee
                    const fee = await this.getExchangeFee(exchangeId, symbol, side);
                    // Check if exchange has enough liquidity
                    const hasLiquidity = await this.checkExchangeLiquidity(exchangeId, symbol, side, quantity);
                    return {
                        exchangeId,
                        price,
                        fee,
                        hasLiquidity
                    };
                }
                catch (error) {
                    logger.warn(`Failed to get price from ${exchangeId}`, {
                        error: error instanceof Error ? error.message : String(error),
                        exchangeId,
                        symbol
                    });
                    return null;
                }
            }));
            // Filter out null values and exchanges with insufficient liquidity
            const validExchanges = exchangePrices
                .filter(exchange => exchange !== null && exchange.hasLiquidity)
                .sort((a, b) => {
                // Calculate effective price (including fee)
                const aEffectivePrice = this.calculateEffectivePrice(a.price, a.fee, side);
                const bEffectivePrice = this.calculateEffectivePrice(b.price, b.fee, side);
                // Sort based on side (BUY: lowest price first, SELL: highest price first)
                if (side === orderExecution_1.OrderSide.BUY) {
                    return aEffectivePrice - bEffectivePrice;
                }
                else {
                    return bEffectivePrice - aEffectivePrice;
                }
            });
            // If no valid exchanges, use default
            if (validExchanges.length === 0) {
                throw new Error('No valid exchanges found');
            }
            // Return best exchange
            return {
                exchangeId: validExchanges[0].exchangeId,
                price: validExchanges[0].price,
                fee: validExchanges[0].fee
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to get best exchange: ${errorMessage}`, {
                error: errorMessage,
                symbol,
                side,
                quantity
            });
            // Fallback to default exchange with zero price and fee
            return {
                exchangeId: this.exchangePriority[0],
                price: 0,
                fee: 0
            };
        }
    }
    /**
     * Split a large order into smaller orders
     * @param request - Order execution request
     * @returns Array of split orders
     */
    async splitOrder(request) {
        try {
            logger.info(`Splitting order for ${request.symbol}`, { request });
            // Check if order is large enough to split
            const marketImpact = await this.estimateMarketImpact(request.symbol, request.quantity, request.side);
            // If market impact is low, don't split
            if (marketImpact.priceImpact < 0.001) { // 0.1%
                return [request];
            }
            // Calculate number of slices
            const numSlices = Math.min(Math.ceil(request.quantity / marketImpact.maxSingleOrderSize), 10 // Maximum of 10 slices
            );
            // If only one slice, don't split
            if (numSlices <= 1) {
                return [request];
            }
            // Calculate slice size
            const sliceSize = request.quantity / numSlices;
            // Create split orders
            const splitOrders = [];
            for (let i = 0; i < numSlices; i++) {
                // Clone request
                const splitOrder = {
                    ...request,
                    quantity: sliceSize,
                    clientOrderId: `${request.clientOrderId || (0, uuid_1.v4)()}_${i + 1}`
                };
                splitOrders.push(splitOrder);
            }
            return splitOrders;
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Order splitting failed: ${errorMessage}`, {
                error: errorMessage,
                request
            });
            // Fallback to original order
            return [request];
        }
    }
    /**
     * Estimate market impact of a large order
     * @param symbol - Trading symbol
     * @param quantity - Order quantity
     * @param side - Order side
     * @returns Market impact estimation
     * @private
     */
    async estimateMarketImpact(symbol, quantity, side) {
        try {
            // Analyze order books across exchanges
            const exchanges = Array.from(this.exchangeConnectors.keys());
            // Get order books from all exchanges
            const orderBooks = await Promise.all(exchanges.map(async (exchangeId) => {
                try {
                    const connector = this.exchangeConnectors.get(exchangeId);
                    return await connector.getOrderBook(symbol, 20); // Get top 20 levels
                }
                catch (error) {
                    logger.warn(`Failed to get order book from ${exchangeId}`, {
                        error: error instanceof Error ? error.message : String(error),
                        exchangeId,
                        symbol
                    });
                    return null;
                }
            }));
            // Filter out null values
            const validOrderBooks = orderBooks.filter(book => book !== null);
            // If no valid order books, return default values
            if (validOrderBooks.length === 0) {
                return {
                    priceImpact: 0.01, // 1%
                    recommendedSlices: 5,
                    maxSingleOrderSize: quantity / 5
                };
            }
            // Calculate market depth
            let totalDepth = 0;
            let weightedPrice = 0;
            let availableQuantity = 0;
            for (const book of validOrderBooks) {
                // Get the relevant side of the book
                const levels = side === orderExecution_1.OrderSide.BUY ? book.asks : book.bids;
                // Calculate depth
                for (const [price, size] of levels) {
                    availableQuantity += size;
                    weightedPrice += price * size;
                    if (availableQuantity >= quantity) {
                        break;
                    }
                }
                totalDepth += availableQuantity;
            }
            // Calculate average price
            const avgPrice = weightedPrice / availableQuantity;
            // Get best price
            const bestPrice = side === orderExecution_1.OrderSide.BUY
                ? validOrderBooks[0].asks[0][0]
                : validOrderBooks[0].bids[0][0];
            // Calculate price impact
            const priceImpact = Math.abs(avgPrice - bestPrice) / bestPrice;
            // Calculate recommended slices
            const recommendedSlices = Math.max(1, Math.min(10, Math.ceil(priceImpact * 100)));
            // Calculate max single order size
            const maxSingleOrderSize = Math.max(quantity / recommendedSlices, totalDepth / 20 // 5% of total depth
            );
            return {
                priceImpact,
                recommendedSlices,
                maxSingleOrderSize
            };
        }
        catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to estimate market impact: ${errorMessage}`, {
                error: errorMessage,
                symbol,
                quantity,
                side
            });
            // Return default values
            return {
                priceImpact: 0.01, // 1%
                recommendedSlices: 5,
                maxSingleOrderSize: quantity / 5
            };
        }
    }
    /**
     * Check if an exchange has enough liquidity for an order
     * @param exchangeId - Exchange ID
     * @param symbol - Trading symbol
     * @param side - Order side
     * @param quantity - Order quantity
     * @returns Whether the exchange has enough liquidity
     * @private
     */
    async checkExchangeLiquidity(exchangeId, symbol, side, quantity) {
        try {
            const connector = this.exchangeConnectors.get(exchangeId);
            const orderBook = await connector.getOrderBook(symbol, 20);
            // Get the relevant side of the book
            const levels = side === orderExecution_1.OrderSide.BUY ? orderBook.asks : orderBook.bids;
            // Calculate available quantity
            let availableQuantity = 0;
            for (const [_, size] of levels) {
                availableQuantity += size;
                if (availableQuantity >= quantity) {
                    return true;
                }
            }
            // If not enough liquidity, check if it's at least 80% of the order
            return availableQuantity >= quantity * 0.8;
        }
        catch (error) {
            logger.warn(`Failed to check liquidity on ${exchangeId}`, {
                error: error instanceof Error ? error.message : String(error),
                exchangeId,
                symbol
            });
            return false;
        }
    }
    /**
     * Get exchange fee for a trading pair
     * @param exchangeId - Exchange ID
     * @param symbol - Trading symbol
     * @param side - Order side
     * @returns Fee percentage
     * @private
     */
    async getExchangeFee(exchangeId, symbol, side) {
        // For now, return hardcoded fees
        // In a real implementation, this would fetch from the exchange or a database
        switch (exchangeId) {
            case 'delta':
                return 0.0007; // 0.07%
            case 'binance':
                return 0.001; // 0.1%
            case 'kraken':
                return 0.0016; // 0.16%
            default:
                return 0.002; // 0.2% default
        }
    }
    /**
     * Calculate effective price including fees
     * @param price - Raw price
     * @param fee - Fee percentage
     * @param side - Order side
     * @returns Effective price
     * @private
     */
    calculateEffectivePrice(price, fee, side) {
        // For buy orders, effective price is higher due to fees
        // For sell orders, effective price is lower due to fees
        return side === orderExecution_1.OrderSide.BUY
            ? price * (1 + fee)
            : price * (1 - fee);
    }
    /**
     * Refresh exchange data
     * @private
     */
    async refreshExchangeData() {
        try {
            // Refresh exchange info for each connector
            for (const [exchangeId, connector] of this.exchangeConnectors.entries()) {
                try {
                    if (typeof connector.refreshExchangeInfo === 'function') {
                        await connector.refreshExchangeInfo();
                    }
                }
                catch (error) {
                    logger.warn(`Failed to refresh data for ${exchangeId}`, {
                        error: error instanceof Error ? error.message : String(error),
                        exchangeId
                    });
                }
            }
        }
        catch (error) {
            logger.error('Failed to refresh exchange data', {
                error: error instanceof Error ? error.message : String(error)
            });
        }
    }
}
exports.SmartOrderRouter = SmartOrderRouter;
// Create singleton instance
const smartOrderRouter = new SmartOrderRouter();
exports.default = smartOrderRouter;
//# sourceMappingURL=smartOrderRouter.js.map