/**
 * Error Handling Middleware
 * Provides consistent error handling across the application
 */

import { Request, Response, NextFunction } from 'express';
import env from '../utils/env';

/**
 * Custom error interface with statusCode
 */
interface CustomError extends Error {
  statusCode?: number;
}

/**
 * Error response interface
 */
interface ErrorResponse {
  success: boolean;
  status: number;
  message: string;
  stack?: string;
}

/**
 * Helper function to create standardized error objects
 * @param {string} message - Error message
 * @param {number} statusCode - HTTP status code
 * @returns {CustomError} Custom error with statusCode property
 */
export const createError = (message: string, statusCode = 500): CustomError => {
  const error: CustomError = new Error(message);
  error.statusCode = statusCode;
  return error;
};

/**
 * 404 Not Found handler for undefined routes
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
export const notFoundHandler = (req: Request, res: Response, next: NextFunction): void => {
  const error = createError(`Resource not found: ${req.originalUrl}`, 404);
  next(error);
};

/**
 * Global error handling middleware
 * @param {CustomError} err - Error object
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
export const errorHandler = (err: CustomError, req: Request, res: Response, next: NextFunction): void => {
  // Log the error
  console.error('Error:', err);
  
  // Default error status and message
  const statusCode = err.statusCode || 500;
  const message = err.message || 'Internal Server Error';
  
  // Create response object
  const response: ErrorResponse = {
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

/**
 * Async error handler wrapper
 * Automatically catches errors in async route handlers and passes them to next()
 */
export const asyncHandler = (fn: Function) => {
  return (req: Request, res: Response, next: NextFunction): Promise<void> => {
    return Promise.resolve(fn(req, res, next)).catch(next);
  };
};

export default {
  createError,
  errorHandler,
  notFoundHandler,
  asyncHandler
}; 