"use strict";
/**
 * Error Handling Middleware
 * Provides consistent error handling across the application
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.asyncHandler = exports.errorHandler = exports.notFoundHandler = exports.createError = void 0;
/**
 * Helper function to create standardized error objects
 * @param {string} message - Error message
 * @param {number} statusCode - HTTP status code
 * @returns {CustomError} Custom error with statusCode property
 */
const createError = (message, statusCode = 500) => {
    const error = new Error(message);
    error.statusCode = statusCode;
    return error;
};
exports.createError = createError;
/**
 * 404 Not Found handler for undefined routes
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
const notFoundHandler = (req, res, next) => {
    const error = (0, exports.createError)(`Resource not found: ${req.originalUrl}`, 404);
    next(error);
};
exports.notFoundHandler = notFoundHandler;
/**
 * Global error handling middleware
 * @param {CustomError} err - Error object
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
const errorHandler = (err, req, res, next) => {
    // Log the error
    console.error('Error:', err);
    // Default error status and message
    const statusCode = err.statusCode || 500;
    const message = err.message || 'Internal Server Error';
    // Create response object
    const response = {
        success: false,
        status: statusCode,
        message
    };
    // Add stack trace in development mode
    if (process.env.NODE_ENV === 'development') {
        response.stack = err.stack;
    }
    // Send response with appropriate status code
    res.status(statusCode).json(response);
};
exports.errorHandler = errorHandler;
/**
 * Async error handler wrapper
 * Automatically catches errors in async route handlers and passes them to next()
 */
const asyncHandler = (fn) => {
    return (req, res, next) => {
        return Promise.resolve(fn(req, res, next)).catch(next);
    };
};
exports.asyncHandler = asyncHandler;
exports.default = {
    createError: exports.createError,
    errorHandler: exports.errorHandler,
    notFoundHandler: exports.notFoundHandler,
    asyncHandler: exports.asyncHandler
};
//# sourceMappingURL=errorHandler.js.map