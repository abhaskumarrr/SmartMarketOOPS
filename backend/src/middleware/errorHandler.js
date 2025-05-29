/**
 * Error Handling Middleware
 * Provides consistent error handling across the application
 */

const { env } = require('../utils/env');

/**
 * Helper function to create standardized error objects
 * @param {string} message - Error message
 * @param {number} statusCode - HTTP status code
 * @returns {Error} Custom error with statusCode property
 */
const createError = (message, statusCode = 500) => {
  const error = new Error(message);
  error.statusCode = statusCode;
  return error;
};

/**
 * 404 Not Found handler for undefined routes
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const notFoundHandler = (req, res, next) => {
  const error = createError(`Resource not found: ${req.originalUrl}`, 404);
  next(error);
};

/**
 * Global error handling middleware
 * @param {Error} err - Error object
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
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
  if (env.NODE_ENV === 'development') {
    response.stack = err.stack;
  }
  
  // Send response with appropriate status code
  res.status(statusCode).json(response);
};

module.exports = {
  createError,
  notFoundHandler,
  errorHandler
}; 