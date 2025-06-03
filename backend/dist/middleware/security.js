"use strict";
/**
 * Security Middleware
 * Enhanced security features for SmartMarketOOPS
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.memoryMonitor = exports.securityLogger = exports.validateInput = exports.securityHeaders = exports.csrfProtection = exports.passwordResetRateLimiter = exports.apiRateLimiter = exports.authRateLimiter = void 0;
const express_rate_limit_1 = __importDefault(require("express-rate-limit"));
const helmet_1 = __importDefault(require("helmet"));
const crypto_1 = __importDefault(require("crypto"));
const env_1 = __importDefault(require("../utils/env"));
// CSRF token storage (in production, use Redis or database)
const csrfTokens = new Map();
/**
 * Enhanced rate limiter for authentication routes
 */
exports.authRateLimiter = (0, express_rate_limit_1.default)({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 requests per IP per window
    standardHeaders: true,
    legacyHeaders: false,
    message: {
        success: false,
        message: 'Too many authentication attempts, please try again later.',
        retryAfter: 15 * 60 // 15 minutes in seconds
    },
    // Skip successful requests
    skipSuccessfulRequests: true,
    // Custom key generator for more granular control
    keyGenerator: (req) => {
        return req.ip + ':' + (req.headers['user-agent'] || 'unknown');
    }
});
/**
 * Rate limiter for API routes
 */
exports.apiRateLimiter = (0, express_rate_limit_1.default)({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // 100 requests per IP per window
    standardHeaders: true,
    legacyHeaders: false,
    message: {
        success: false,
        message: 'Too many API requests, please try again later.'
    }
});
/**
 * Rate limiter for password reset attempts
 */
exports.passwordResetRateLimiter = (0, express_rate_limit_1.default)({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 3, // 3 password reset attempts per hour
    standardHeaders: true,
    legacyHeaders: false,
    message: {
        success: false,
        message: 'Too many password reset attempts, please try again later.'
    }
});
/**
 * Memory-efficient CSRF protection
 */
exports.csrfProtection = {
    /**
     * Generate CSRF token
     */
    generateToken: (sessionId) => {
        const token = crypto_1.default.randomBytes(32).toString('hex');
        const expires = Date.now() + (60 * 60 * 1000); // 1 hour
        // Clean up expired tokens periodically
        if (csrfTokens.size > 1000) {
            const now = Date.now();
            for (const [key, value] of csrfTokens.entries()) {
                if (value.expires < now) {
                    csrfTokens.delete(key);
                }
            }
        }
        csrfTokens.set(sessionId, { token, expires });
        return token;
    },
    /**
     * Verify CSRF token
     */
    verifyToken: (sessionId, token) => {
        const stored = csrfTokens.get(sessionId);
        if (!stored)
            return false;
        if (stored.expires < Date.now()) {
            csrfTokens.delete(sessionId);
            return false;
        }
        return stored.token === token;
    },
    /**
     * CSRF middleware
     */
    middleware: (req, res, next) => {
        // Skip CSRF for GET, HEAD, OPTIONS
        if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
            next();
            return;
        }
        // Skip CSRF for API endpoints with valid JWT (they have their own protection)
        if (req.headers.authorization && req.headers.authorization.startsWith('Bearer ')) {
            next();
            return;
        }
        const sessionId = req.sessionID || req.headers['x-session-id'];
        const csrfToken = req.headers['x-csrf-token'] || req.body._csrf;
        if (!sessionId || !csrfToken) {
            res.status(403).json({
                success: false,
                message: 'CSRF token required'
            });
            return;
        }
        if (!exports.csrfProtection.verifyToken(sessionId, csrfToken)) {
            res.status(403).json({
                success: false,
                message: 'Invalid CSRF token'
            });
            return;
        }
        next();
    }
};
/**
 * Security headers middleware using helmet
 */
exports.securityHeaders = (0, helmet_1.default)({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
            fontSrc: ["'self'", "https://fonts.gstatic.com"],
            imgSrc: ["'self'", "data:", "https:"],
            scriptSrc: ["'self'"],
            connectSrc: ["'self'", "wss:", "ws:"],
            frameSrc: ["'none'"],
            objectSrc: ["'none'"],
            baseUri: ["'self'"],
            formAction: ["'self'"],
            frameAncestors: ["'none'"]
        }
    },
    crossOriginEmbedderPolicy: false, // Disable for development
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    }
});
/**
 * Input validation middleware
 */
exports.validateInput = {
    /**
     * Sanitize string input
     */
    sanitizeString: (input) => {
        if (typeof input !== 'string')
            return '';
        return input.trim().replace(/[<>]/g, '');
    },
    /**
     * Validate email format
     */
    isValidEmail: (email) => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    },
    /**
     * Validate password strength
     */
    isValidPassword: (password) => {
        if (password.length < 8) {
            return { valid: false, message: 'Password must be at least 8 characters long' };
        }
        if (!/(?=.*[a-z])/.test(password)) {
            return { valid: false, message: 'Password must contain at least one lowercase letter' };
        }
        if (!/(?=.*[A-Z])/.test(password)) {
            return { valid: false, message: 'Password must contain at least one uppercase letter' };
        }
        if (!/(?=.*\d)/.test(password)) {
            return { valid: false, message: 'Password must contain at least one number' };
        }
        if (!/(?=.*[@$!%*?&])/.test(password)) {
            return { valid: false, message: 'Password must contain at least one special character' };
        }
        return { valid: true };
    },
    /**
     * Request validation middleware
     */
    middleware: (req, res, next) => {
        // Sanitize string inputs
        if (req.body) {
            for (const key in req.body) {
                if (typeof req.body[key] === 'string') {
                    req.body[key] = exports.validateInput.sanitizeString(req.body[key]);
                }
            }
        }
        next();
    }
};
/**
 * Request logging middleware for security monitoring
 */
const securityLogger = (req, res, next) => {
    const startTime = Date.now();
    // Log security-relevant requests
    const securityPaths = ['/api/auth/', '/api/admin/', '/api/user/'];
    const isSecurityPath = securityPaths.some(path => req.path.startsWith(path));
    if (isSecurityPath) {
        console.log(`[SECURITY] ${req.method} ${req.path} - IP: ${req.ip} - User-Agent: ${req.headers['user-agent']}`);
    }
    // Override res.json to log response status
    const originalJson = res.json;
    res.json = function (body) {
        const duration = Date.now() - startTime;
        if (isSecurityPath) {
            console.log(`[SECURITY] ${req.method} ${req.path} - Status: ${res.statusCode} - Duration: ${duration}ms`);
            // Log failed authentication attempts
            if (res.statusCode === 401 || res.statusCode === 403) {
                console.warn(`[SECURITY ALERT] Failed auth attempt - IP: ${req.ip} - Path: ${req.path}`);
            }
        }
        return originalJson.call(this, body);
    };
    next();
};
exports.securityLogger = securityLogger;
/**
 * Memory usage monitoring middleware
 */
const memoryMonitor = (req, res, next) => {
    const memUsage = process.memoryUsage();
    const memUsageMB = {
        rss: Math.round(memUsage.rss / 1024 / 1024),
        heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024),
        heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024),
        external: Math.round(memUsage.external / 1024 / 1024)
    };
    // Log memory usage if it's high (for M2 MacBook Air 8GB optimization)
    if (memUsageMB.heapUsed > 512) { // 512MB threshold
        console.warn(`[MEMORY] High memory usage: ${JSON.stringify(memUsageMB)}MB`);
    }
    // Add memory info to response headers in development
    if (env_1.default.NODE_ENV === 'development') {
        res.setHeader('X-Memory-Usage', JSON.stringify(memUsageMB));
    }
    next();
};
exports.memoryMonitor = memoryMonitor;
exports.default = {
    authRateLimiter: exports.authRateLimiter,
    apiRateLimiter: exports.apiRateLimiter,
    passwordResetRateLimiter: exports.passwordResetRateLimiter,
    csrfProtection: exports.csrfProtection,
    securityHeaders: exports.securityHeaders,
    validateInput: exports.validateInput,
    securityLogger: exports.securityLogger,
    memoryMonitor: exports.memoryMonitor
};
//# sourceMappingURL=security.js.map