/**
 * Order Execution Controller
 * Handles HTTP requests for order execution
 */
import { Request, Response } from 'express';
interface AuthenticatedRequest extends Request {
    user?: {
        id: string;
        [key: string]: any;
    };
}
/**
 * Order Execution Controller
 */
export declare const orderExecutionController: {
    /**
     * Execute an order
     * POST /api/orders/execute
     */
    executeOrder(req: AuthenticatedRequest, res: Response): Promise<void>;
    /**
     * Cancel an order
     * POST /api/orders/:id/cancel
     */
    cancelOrder(req: AuthenticatedRequest, res: Response): Promise<void>;
    /**
     * Get order details
     * GET /api/orders/:id
     */
    getOrder(req: AuthenticatedRequest, res: Response): Promise<void>;
    /**
     * Get user orders
     * GET /api/orders
     */
    getUserOrders(req: AuthenticatedRequest, res: Response): Promise<void>;
    /**
     * Get orders by position
     * GET /api/positions/:id/orders
     */
    getPositionOrders(req: AuthenticatedRequest, res: Response): Promise<void>;
    /**
     * Get orders by strategy
     * GET /api/strategies/:id/orders
     */
    getStrategyOrders(req: AuthenticatedRequest, res: Response): Promise<void>;
    /**
     * Get orders by bot
     * GET /api/bots/:id/orders
     */
    getBotOrders(req: AuthenticatedRequest, res: Response): Promise<void>;
    /**
     * Get orders by signal
     * GET /api/signals/:id/orders
     */
    getSignalOrders(req: AuthenticatedRequest, res: Response): Promise<void>;
};
export default orderExecutionController;
