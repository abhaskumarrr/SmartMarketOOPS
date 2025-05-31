/**
 * Rate Limiter Middleware
 * Prevents brute force attacks and abuse by limiting request rates
 */
/**
 * Standard rate limiter for general API endpoints
 * 100 requests per minute
 */
export declare const standardLimiter: import("express-rate-limit").RateLimitRequestHandler;
/**
 * Strict rate limiter for sensitive operations like authentication
 * 5 requests per 15 minutes
 */
export declare const authLimiter: import("express-rate-limit").RateLimitRequestHandler;
/**
 * Rate limiter for API key validation
 * 10 requests per 5 minutes
 */
export declare const apiKeyValidationLimiter: import("express-rate-limit").RateLimitRequestHandler;
/**
 * Rate limiter for API key management operations
 * 20 requests per 10 minutes
 */
export declare const apiKeyManagementLimiter: import("express-rate-limit").RateLimitRequestHandler;
/**
 * IP-based rate limiter for high-risk operations
 * Limits based on IP address
 * 3 requests per hour
 */
export declare const sensitiveOperationsLimiter: import("express-rate-limit").RateLimitRequestHandler;
declare const _default: {
    standardLimiter: import("express-rate-limit").RateLimitRequestHandler;
    authLimiter: import("express-rate-limit").RateLimitRequestHandler;
    apiKeyValidationLimiter: import("express-rate-limit").RateLimitRequestHandler;
    apiKeyManagementLimiter: import("express-rate-limit").RateLimitRequestHandler;
    sensitiveOperationsLimiter: import("express-rate-limit").RateLimitRequestHandler;
};
export default _default;
