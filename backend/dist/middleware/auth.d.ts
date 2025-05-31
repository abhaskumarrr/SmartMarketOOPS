/**
 * Authentication Middleware
 * Validates JWT tokens and protects routes
 */
import { Request, Response, NextFunction } from 'express';
import { Permission, AuthenticatedRequest } from '../types/auth';
/**
 * Rate limiter for authentication routes
 * Limits login, register, and password reset attempts
 */
export declare const authRateLimiter: import("express-rate-limit").RateLimitRequestHandler;
/**
 * CSRF protection middleware
 */
export declare const csrfProtection: import("express-serve-static-core").RequestHandler<import("express-serve-static-core").ParamsDictionary, any, any, import("qs").ParsedQs, Record<string, any>>;
/**
 * Middleware to protect routes
 * Verifies JWT token and attaches user to request
 */
export declare const protect: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
/**
 * Middleware to verify refresh token
 * Used for token refresh routes
 */
export declare const verifyRefreshToken: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
/**
 * Middleware to require email verification
 * Used for routes that require verified email
 */
export declare const requireVerified: (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
/**
 * Middleware to require specific role(s)
 * Used for routes that require specific user roles
 * @param roles - Array of allowed roles
 */
export declare const requireRole: (roles: string[]) => (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
/**
 * Middleware to require specific permission(s)
 * Used for fine-grained access control based on permissions
 * @param permissions - Array of required permissions (all must be present)
 */
export declare const requirePermission: (permissions: Permission[]) => (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
/**
 * Middleware to require any of the specified permissions
 * At least one of the permissions must be present
 * @param permissions - Array of permissions (at least one must be present)
 */
export declare const requireAnyPermission: (permissions: Permission[]) => (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
/**
 * Middleware to verify user is authenticated
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
export declare const isAuthenticated: (req: Request, res: Response, next: NextFunction) => Promise<void>;
/**
 * Middleware to verify user has required role
 * @param {string[]} roles - Array of allowed roles
 */
export declare const hasRole: (roles: string[]) => (req: Request, res: Response, next: NextFunction) => void;
declare const _default: {
    protect: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
    verifyRefreshToken: (req: AuthenticatedRequest, res: Response, next: NextFunction) => Promise<void>;
    requireVerified: (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
    requireRole: (roles: string[]) => (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
    requirePermission: (permissions: Permission[]) => (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
    requireAnyPermission: (permissions: Permission[]) => (req: AuthenticatedRequest, res: Response, next: NextFunction) => void;
    authRateLimiter: import("express-rate-limit").RateLimitRequestHandler;
    csrfProtection: import("express-serve-static-core").RequestHandler<import("express-serve-static-core").ParamsDictionary, any, any, import("qs").ParsedQs, Record<string, any>>;
    isAuthenticated: (req: Request, res: Response, next: NextFunction) => Promise<void>;
    hasRole: (roles: string[]) => (req: Request, res: Response, next: NextFunction) => void;
};
export default _default;
