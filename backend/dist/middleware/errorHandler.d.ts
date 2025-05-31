/**
 * Error Handling Middleware
 * Provides consistent error handling across the application
 */
import { Request, Response, NextFunction } from 'express';
/**
 * Custom error interface with statusCode
 */
interface CustomError extends Error {
    statusCode?: number;
}
/**
 * Helper function to create standardized error objects
 * @param {string} message - Error message
 * @param {number} statusCode - HTTP status code
 * @returns {CustomError} Custom error with statusCode property
 */
export declare const createError: (message: string, statusCode?: number) => CustomError;
/**
 * 404 Not Found handler for undefined routes
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
export declare const notFoundHandler: (req: Request, res: Response, next: NextFunction) => void;
/**
 * Global error handling middleware
 * @param {CustomError} err - Error object
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
export declare const errorHandler: (err: CustomError, req: Request, res: Response, next: NextFunction) => void;
/**
 * Async error handler wrapper
 * Automatically catches errors in async route handlers and passes them to next()
 */
export declare const asyncHandler: (fn: Function) => (req: Request, res: Response, next: NextFunction) => Promise<void>;
declare const _default: {
    createError: (message: string, statusCode?: number) => CustomError;
    errorHandler: (err: CustomError, req: Request, res: Response, next: NextFunction) => void;
    notFoundHandler: (req: Request, res: Response, next: NextFunction) => void;
    asyncHandler: (fn: Function) => (req: Request, res: Response, next: NextFunction) => Promise<void>;
};
export default _default;
