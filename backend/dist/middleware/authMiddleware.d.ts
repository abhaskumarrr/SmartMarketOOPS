/**
 * Auth Middleware
 * Provides authentication and authorization middleware for routes
 */
import { Request, Response, NextFunction } from 'express';
export interface AuthenticatedRequest extends Request {
    user?: {
        id: string;
        email?: string;
        role?: string;
        [key: string]: any;
    };
}
/**
 * Authenticate JWT token and attach user to request
 * @param req - Express request
 * @param res - Express response
 * @param next - Express next function
 */
export declare const authenticateJWT: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
/**
 * Authorize roles middleware
 * @param roles - Allowed roles
 * @returns Middleware function
 */
export declare const authorizeRoles: (roles: string[]) => (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
