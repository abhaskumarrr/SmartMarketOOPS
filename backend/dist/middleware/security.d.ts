/**
 * Security Middleware
 * Enhanced security features for SmartMarketOOPS
 */
import { Request, Response, NextFunction } from 'express';
/**
 * Enhanced rate limiter for authentication routes
 */
export declare const authRateLimiter: import("express-rate-limit").RateLimitRequestHandler;
/**
 * Rate limiter for API routes
 */
export declare const apiRateLimiter: import("express-rate-limit").RateLimitRequestHandler;
/**
 * Rate limiter for password reset attempts
 */
export declare const passwordResetRateLimiter: import("express-rate-limit").RateLimitRequestHandler;
/**
 * Memory-efficient CSRF protection
 */
export declare const csrfProtection: {
    /**
     * Generate CSRF token
     */
    generateToken: (sessionId: string) => string;
    /**
     * Verify CSRF token
     */
    verifyToken: (sessionId: string, token: string) => boolean;
    /**
     * CSRF middleware
     */
    middleware: (req: Request, res: Response, next: NextFunction) => void;
};
/**
 * Security headers middleware using helmet
 */
export declare const securityHeaders: (req: import("http").IncomingMessage, res: import("http").ServerResponse, next: (err?: unknown) => void) => void;
/**
 * Input validation middleware
 */
export declare const validateInput: {
    /**
     * Sanitize string input
     */
    sanitizeString: (input: string) => string;
    /**
     * Validate email format
     */
    isValidEmail: (email: string) => boolean;
    /**
     * Validate password strength
     */
    isValidPassword: (password: string) => {
        valid: boolean;
        message?: string;
    };
    /**
     * Request validation middleware
     */
    middleware: (req: Request, res: Response, next: NextFunction) => void;
};
/**
 * Request logging middleware for security monitoring
 */
export declare const securityLogger: (req: Request, res: Response, next: NextFunction) => void;
/**
 * Memory usage monitoring middleware
 */
export declare const memoryMonitor: (req: Request, res: Response, next: NextFunction) => void;
declare const _default: {
    authRateLimiter: import("express-rate-limit").RateLimitRequestHandler;
    apiRateLimiter: import("express-rate-limit").RateLimitRequestHandler;
    passwordResetRateLimiter: import("express-rate-limit").RateLimitRequestHandler;
    csrfProtection: {
        /**
         * Generate CSRF token
         */
        generateToken: (sessionId: string) => string;
        /**
         * Verify CSRF token
         */
        verifyToken: (sessionId: string, token: string) => boolean;
        /**
         * CSRF middleware
         */
        middleware: (req: Request, res: Response, next: NextFunction) => void;
    };
    securityHeaders: (req: import("http").IncomingMessage, res: import("http").ServerResponse, next: (err?: unknown) => void) => void;
    validateInput: {
        /**
         * Sanitize string input
         */
        sanitizeString: (input: string) => string;
        /**
         * Validate email format
         */
        isValidEmail: (email: string) => boolean;
        /**
         * Validate password strength
         */
        isValidPassword: (password: string) => {
            valid: boolean;
            message?: string;
        };
        /**
         * Request validation middleware
         */
        middleware: (req: Request, res: Response, next: NextFunction) => void;
    };
    securityLogger: (req: Request, res: Response, next: NextFunction) => void;
    memoryMonitor: (req: Request, res: Response, next: NextFunction) => void;
};
export default _default;
