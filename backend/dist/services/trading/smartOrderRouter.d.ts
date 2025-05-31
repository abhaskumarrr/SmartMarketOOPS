/**
 * Smart Order Router
 * Routes orders to the best exchange based on price and liquidity
 */
import { OrderExecutionRequest, OrderSide, ISmartOrderRouter } from '../../types/orderExecution';
/**
 * Smart Order Router
 * Routes orders to the best exchange based on price and liquidity
 */
export declare class SmartOrderRouter implements ISmartOrderRouter {
    private exchangeConnectors;
    private exchangePriority;
    /**
     * Create a new Smart Order Router
     */
    constructor();
    /**
     * Register an exchange connector
     * @param exchangeId - Exchange identifier
     * @param connector - Exchange connector
     */
    registerExchangeConnector(exchangeId: string, connector: any): void;
    /**
     * Route an order to the best exchange
     * @param request - Order execution request
     * @returns Exchange ID
     */
    route(request: OrderExecutionRequest): Promise<string>;
    /**
     * Get the best exchange for an order
     * @param symbol - Trading symbol
     * @param side - Order side
     * @param quantity - Order quantity
     * @returns Best exchange information
     */
    getBestExchange(symbol: string, side: OrderSide, quantity: number): Promise<{
        exchangeId: string;
        price: number;
        fee: number;
    }>;
    /**
     * Split a large order into smaller orders
     * @param request - Order execution request
     * @returns Array of split orders
     */
    splitOrder(request: OrderExecutionRequest): Promise<OrderExecutionRequest[]>;
    /**
     * Estimate market impact of a large order
     * @param symbol - Trading symbol
     * @param quantity - Order quantity
     * @param side - Order side
     * @returns Market impact estimation
     * @private
     */
    private estimateMarketImpact;
    /**
     * Check if an exchange has enough liquidity for an order
     * @param exchangeId - Exchange ID
     * @param symbol - Trading symbol
     * @param side - Order side
     * @param quantity - Order quantity
     * @returns Whether the exchange has enough liquidity
     * @private
     */
    private checkExchangeLiquidity;
    /**
     * Get exchange fee for a trading pair
     * @param exchangeId - Exchange ID
     * @param symbol - Trading symbol
     * @param side - Order side
     * @returns Fee percentage
     * @private
     */
    private getExchangeFee;
    /**
     * Calculate effective price including fees
     * @param price - Raw price
     * @param fee - Fee percentage
     * @param side - Order side
     * @returns Effective price
     * @private
     */
    private calculateEffectivePrice;
    /**
     * Refresh exchange data
     * @private
     */
    private refreshExchangeData;
}
declare const smartOrderRouter: SmartOrderRouter;
export default smartOrderRouter;
