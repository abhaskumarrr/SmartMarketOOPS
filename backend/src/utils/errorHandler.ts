/**
 * Centralized Error Handling Utility
 * Provides consistent error handling across the application
 */

import { Request, Response, NextFunction } from 'express';
import { logger } from './logger';

// Custom error class for API errors
export class ApiError extends Error {
  statusCode: number;
  isOperational: boolean;
  errorCode?: string;
  details?: any;

  constructor(
    message: string,
    statusCode: number = 500,
    isOperational: boolean = true,
    errorCode?: string,
    details?: any
  ) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    this.errorCode = errorCode;
    this.details = details;
    
    // Capture stack trace
    Error.captureStackTrace(this, this.constructor);
  }
}

// Error handler for async functions
export const asyncHandler = (fn: Function) => (req: Request, res: Response, next: NextFunction) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

// Global error handler middleware
export const errorMiddleware = (err: any, req: Request, res: Response, next: NextFunction) => {
  // Default error values
  let statusCode = err.statusCode || 500;
  let message = err.message || 'Internal Server Error';
  let errorCode = err.errorCode || 'INTERNAL_ERROR';
  const isOperational = err.isOperational !== undefined ? err.isOperational : true;
  let details = err.details || null;
  
  // Handle specific error types
  if (err.name === 'ValidationError') {
    statusCode = 400;
    message = 'Validation Error';
    errorCode = 'VALIDATION_ERROR';
    details = err.errors;
  } else if (err.name === 'UnauthorizedError') {
    statusCode = 401;
    message = 'Unauthorized';
    errorCode = 'UNAUTHORIZED';
  } else if (err.name === 'JsonWebTokenError') {
    statusCode = 401;
    message = 'Invalid token';
    errorCode = 'INVALID_TOKEN';
  } else if (err.name === 'TokenExpiredError') {
    statusCode = 401;
    message = 'Token expired';
    errorCode = 'TOKEN_EXPIRED';
  } else if (err.code === 'P2002') {
    // Prisma unique constraint error
    statusCode = 409;
    message = 'Resource already exists';
    errorCode = 'RESOURCE_EXISTS';
    details = { fields: err.meta?.target };
  } else if (err.code === 'P2025') {
    // Prisma record not found
    statusCode = 404;
    message = 'Resource not found';
    errorCode = 'RESOURCE_NOT_FOUND';
  }

  // Log error
  if (statusCode >= 500) {
    logger.error(`[${errorCode}] ${message}`, {
      path: req.path,
      method: req.method,
      statusCode,
      error: err.stack,
      isOperational
    });
  } else {
    logger.warn(`[${errorCode}] ${message}`, {
      path: req.path,
      method: req.method,
      statusCode,
      error: err.message,
      details
    });
  }

  // Send response
  res.status(statusCode).json({
    success: false,
    error: {
      message,
      code: errorCode,
      ...(details && { details })
    }
  });
};

// 404 handler
export const notFoundHandler = (req: Request, res: Response, next: NextFunction) => {
  const error = new ApiError(`Not Found - ${req.originalUrl}`, 404, true, 'NOT_FOUND');
  next(error);
};

// Utility functions for common errors
export const throwNotFound = (message: string = 'Resource not found', errorCode: string = 'RESOURCE_NOT_FOUND') => {
  throw new ApiError(message, 404, true, errorCode);
};

export const throwUnauthorized = (message: string = 'Unauthorized', errorCode: string = 'UNAUTHORIZED') => {
  throw new ApiError(message, 401, true, errorCode);
};

export const throwForbidden = (message: string = 'Forbidden', errorCode: string = 'FORBIDDEN') => {
  throw new ApiError(message, 403, true, errorCode);
};

export const throwBadRequest = (message: string = 'Bad request', errorCode: string = 'BAD_REQUEST', details?: any) => {
  throw new ApiError(message, 400, true, errorCode, details);
};

export const throwConflict = (message: string = 'Resource already exists', errorCode: string = 'RESOURCE_EXISTS') => {
  throw new ApiError(message, 409, true, errorCode);
};

export const throwServerError = (message: string = 'Internal Server Error', errorCode: string = 'INTERNAL_ERROR') => {
  throw new ApiError(message, 500, false, errorCode);
};

// Export default error handler
export default {
  ApiError,
  asyncHandler,
  errorMiddleware,
  notFoundHandler,
  throwNotFound,
  throwUnauthorized,
  throwForbidden,
  throwBadRequest,
  throwConflict,
  throwServerError
};