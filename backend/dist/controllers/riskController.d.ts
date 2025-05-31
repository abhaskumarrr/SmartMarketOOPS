/**
 * Risk Management Controller
 * Handles risk management API endpoints
 */
import { Request, Response } from 'express';
interface AuthenticatedRequest extends Request {
    user?: {
        id: string;
        [key: string]: any;
    };
}
/**
 * Get risk settings for a user or bot
 * @param req - Express request
 * @param res - Express response
 */
export declare const getRiskSettings: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Create or update risk settings
 * @param req - Express request
 * @param res - Express response
 */
export declare const saveRiskSettings: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Calculate position size
 * @param req - Express request
 * @param res - Express response
 */
export declare const calculatePositionSize: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Generate risk report
 * @param req - Express request
 * @param res - Express response
 */
export declare const generateRiskReport: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Analyze trade risk
 * @param req - Express request
 * @param res - Express response
 */
export declare const analyzeTradeRisk: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get risk alerts
 * @param req - Express request
 * @param res - Express response
 */
export declare const getRiskAlerts: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Acknowledge risk alert
 * @param req - Express request
 * @param res - Express response
 */
export declare const acknowledgeRiskAlert: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Check circuit breaker status
 * @param req - Express request
 * @param res - Express response
 */
export declare const getCircuitBreakerStatus: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Reset circuit breaker
 * @param req - Express request
 * @param res - Express response
 */
export declare const resetCircuitBreaker: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Check if trading is allowed
 * @param req - Express request
 * @param res - Express response
 */
export declare const checkTradingAllowed: (req: AuthenticatedRequest, res: Response) => Promise<void>;
export {};
