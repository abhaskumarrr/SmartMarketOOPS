/**
 * Order Execution Service
 * Handles order execution across different exchanges
 */
import { OrderExecutionRequest, OrderExecutionResult, OrderExecutionOptions, OrderExecutionStatus, IOrderExecutionService } from '../../types/orderExecution';
/**
 * Order Execution Service class
 * Handles order execution across different exchanges
 */
export declare class OrderExecutionService implements IOrderExecutionService {
    private exchangeConnectors;
    private riskManagementService;
    /**
     * Create a new Order Execution Service
     */
    constructor();
    /**
     * Register an exchange connector
     * @param exchangeId - Exchange identifier
     * @param connector - Exchange connector
     */
    private registerExchangeConnector;
    /**
     * Validate an order request
     * @param request - Order execution request
     * @returns Validation error message or null if valid
     */
    private validateOrderRequest;
    /**
     * Perform risk check for an order
     * @param request - Order execution request
     * @returns Result of risk check
     */
    private performRiskCheck;
    /**
     * Map internal order request to exchange-specific format
     * @param request - Order execution request
     * @returns Exchange-specific request
     */
    private mapToExchangeRequest;
    /**
     * Map exchange response to order execution result
     * @param response - Exchange response
     * @param request - Original order request
     * @returns Order execution result
     */
    private mapExchangeResponseToResult;
    /**
     * Store order in database
     * @param order - Order execution result
     */
    private storeOrder;
    /**
     * Execute an order
     * @param request - Order execution request
     * @param options - Order execution options
     * @returns Order execution result
     */
    executeOrder(request: OrderExecutionRequest, options?: OrderExecutionOptions): Promise<OrderExecutionResult>;
    /**
     * Cancel an order
     * @param orderId - Order ID
     * @param exchangeId - Exchange ID
     * @param userId - User ID
     * @returns Whether cancellation was successful
     */
    cancelOrder(orderId: string, exchangeId: string, userId: string): Promise<boolean>;
    /**
     * Get order details
     * @param orderId - Order ID
     * @param exchangeId - Exchange ID
     * @param userId - User ID
     * @returns Order execution result
     */
    getOrder(orderId: string, exchangeId: string, userId: string): Promise<OrderExecutionResult>;
    /**
     * Get orders by user
     * @param userId - User ID
     * @param status - Optional order status filter
     * @returns Array of order execution results
     */
    getOrdersByUser(userId: string, status?: OrderExecutionStatus[]): Promise<OrderExecutionResult[]>;
}
