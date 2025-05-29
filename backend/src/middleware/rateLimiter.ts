/**
 * Rate Limiter Middleware
 * Prevents brute force attacks and abuse by limiting request rates
 */

import rateLimit from 'express-rate-limit';
import { Request, Response } from 'express';

/**
 * Standard rate limiter for general API endpoints
 * 100 requests per minute
 */
export const standardLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // 100 requests per minute
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    error: 'Too many requests, please try again later'
  }
});

/**
 * Strict rate limiter for sensitive operations like authentication
 * 5 requests per 15 minutes
 */
export const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 requests per 15 minutes
  standardHeaders: true, 
  legacyHeaders: false,
  message: {
    success: false,
    error: 'Too many authentication attempts, please try again later'
  }
});

/**
 * Rate limiter for API key validation
 * 10 requests per 5 minutes
 */
export const apiKeyValidationLimiter = rateLimit({
  windowMs: 5 * 60 * 1000, // 5 minutes
  max: 10, // 10 requests per 5 minutes
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    error: 'Too many API key validation attempts, please try again later'
  }
});

/**
 * Rate limiter for API key management operations
 * 20 requests per 10 minutes
 */
export const apiKeyManagementLimiter = rateLimit({
  windowMs: 10 * 60 * 1000, // 10 minutes
  max: 20, // 20 requests per 10 minutes
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    error: 'Too many API key management requests, please try again later'
  }
});

/**
 * IP-based rate limiter for high-risk operations
 * Limits based on IP address
 * 3 requests per hour
 */
export const sensitiveOperationsLimiter = rateLimit({
  windowMs: 60 * 60 * 1000, // 1 hour
  max: 3, // 3 requests per hour
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    success: false,
    error: 'Rate limit exceeded for sensitive operations'
  }
});

export default {
  standardLimiter,
  authLimiter,
  apiKeyValidationLimiter,
  apiKeyManagementLimiter,
  sensitiveOperationsLimiter
}; 