"use strict";
/**
 * Rate Limiter Middleware
 * Prevents brute force attacks and abuse by limiting request rates
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.sensitiveOperationsLimiter = exports.apiKeyManagementLimiter = exports.apiKeyValidationLimiter = exports.authLimiter = exports.standardLimiter = void 0;
const express_rate_limit_1 = __importDefault(require("express-rate-limit"));
/**
 * Standard rate limiter for general API endpoints
 * 100 requests per minute
 */
exports.standardLimiter = (0, express_rate_limit_1.default)({
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
exports.authLimiter = (0, express_rate_limit_1.default)({
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
exports.apiKeyValidationLimiter = (0, express_rate_limit_1.default)({
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
exports.apiKeyManagementLimiter = (0, express_rate_limit_1.default)({
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
exports.sensitiveOperationsLimiter = (0, express_rate_limit_1.default)({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 3, // 3 requests per hour
    standardHeaders: true,
    legacyHeaders: false,
    message: {
        success: false,
        error: 'Rate limit exceeded for sensitive operations'
    }
});
exports.default = {
    standardLimiter: exports.standardLimiter,
    authLimiter: exports.authLimiter,
    apiKeyValidationLimiter: exports.apiKeyValidationLimiter,
    apiKeyManagementLimiter: exports.apiKeyManagementLimiter,
    sensitiveOperationsLimiter: exports.sensitiveOperationsLimiter
};
//# sourceMappingURL=rateLimiter.js.map