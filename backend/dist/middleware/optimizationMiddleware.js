"use strict";
/**
 * API Optimization Middleware
 * Handles rate limiting, compression, caching, and performance monitoring
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OptimizationMiddleware = void 0;
const express_rate_limit_1 = __importDefault(require("express-rate-limit"));
const perf_hooks_1 = require("perf_hooks");
const cacheService_1 = require("../services/cacheService");
class OptimizationMiddleware {
    constructor() {
        this.requestMetrics = [];
        this.maxMetricsHistory = 1000;
    }
    // Performance monitoring middleware
    performanceMonitor() {
        return (req, res, next) => {
            const start = perf_hooks_1.performance.now();
            // Add request ID for tracing
            req.requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            res.on('finish', () => {
                const duration = perf_hooks_1.performance.now() - start;
                const metric = {
                    path: req.path,
                    method: req.method,
                    duration,
                    statusCode: res.statusCode,
                    timestamp: Date.now(),
                    userAgent: req.get('User-Agent'),
                    ip: req.ip,
                };
                this.addRequestMetric(metric);
                // Log slow requests
                if (duration > 1000) {
                    console.warn(`Slow request: ${req.method} ${req.path} - ${duration.toFixed(2)}ms`);
                }
                // Add performance headers only if response hasn't been sent
                if (!res.headersSent) {
                    res.set('X-Response-Time', `${duration.toFixed(2)}ms`);
                    res.set('X-Request-ID', req.requestId);
                }
            });
            next();
        };
    }
    // Response compression middleware
    compressionMiddleware() {
        return (req, res, next) => {
            const originalSend = res.send;
            res.send = function (data) {
                // Only compress JSON responses larger than 1KB
                if (res.get('Content-Type')?.includes('application/json') &&
                    typeof data === 'string' &&
                    data.length > 1024) {
                    res.set('Content-Encoding', 'gzip');
                    // In a real implementation, you'd use actual compression here
                }
                return originalSend.call(this, data);
            };
            next();
        };
    }
    // API response caching middleware
    cacheMiddleware(options = {}) {
        const { ttl = 300, // 5 minutes default
        keyGenerator = (req) => `api:${req.method}:${req.originalUrl}`, skipCache = (req) => req.method !== 'GET', } = options;
        return async (req, res, next) => {
            // Skip caching for non-GET requests or when skipCache returns true
            if (skipCache(req)) {
                return next();
            }
            const cache = (0, cacheService_1.getCacheService)();
            if (!cache) {
                return next();
            }
            const cacheKey = keyGenerator(req);
            try {
                // Check cache
                const cached = await cache.get(cacheKey);
                if (cached) {
                    if (!res.headersSent) {
                        res.set('X-Cache', 'HIT');
                        res.set('Cache-Control', `public, max-age=${ttl}`);
                        return res.json(cached);
                    }
                }
                // Cache miss - intercept response
                const originalSend = res.send;
                res.send = function (data) {
                    // Only cache successful JSON responses
                    if (res.statusCode >= 200 &&
                        res.statusCode < 300 &&
                        res.get('Content-Type')?.includes('application/json')) {
                        try {
                            const parsedData = typeof data === 'string' ? JSON.parse(data) : data;
                            cache.set(cacheKey, parsedData, { ttl });
                            if (!res.headersSent) {
                                res.set('X-Cache', 'MISS');
                                res.set('Cache-Control', `public, max-age=${ttl}`);
                            }
                        }
                        catch (error) {
                            console.error('Error caching response:', error);
                        }
                    }
                    return originalSend.call(this, data);
                };
                next();
            }
            catch (error) {
                console.error('Cache middleware error:', error);
                next();
            }
        };
    }
    // Advanced rate limiting with different tiers
    createRateLimiter(options = {}) {
        return (0, express_rate_limit_1.default)({
            windowMs: options.windowMs || 15 * 60 * 1000, // 15 minutes
            max: options.max || 100, // limit each IP to 100 requests per windowMs
            message: options.message || {
                error: 'Too many requests from this IP, please try again later.',
                retryAfter: Math.ceil((options.windowMs || 15 * 60 * 1000) / 1000),
            },
            keyGenerator: options.keyGenerator || ((req) => req.ip),
            skipSuccessfulRequests: options.skipSuccessfulRequests || false,
            skipFailedRequests: options.skipFailedRequests || false,
            standardHeaders: true,
            legacyHeaders: false,
            // Custom store could be implemented with Redis for distributed rate limiting
        });
    }
    // Request validation and sanitization
    requestValidation() {
        return (req, res, next) => {
            // Sanitize query parameters
            if (req.query) {
                for (const [key, value] of Object.entries(req.query)) {
                    if (typeof value === 'string') {
                        // Basic XSS protection
                        req.query[key] = value.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
                    }
                }
            }
            // Validate content length
            const contentLength = parseInt(req.get('content-length') || '0');
            if (contentLength > 10 * 1024 * 1024) { // 10MB limit
                return res.status(413).json({
                    error: 'Request entity too large',
                    maxSize: '10MB',
                });
            }
            next();
        };
    }
    // Security headers middleware
    securityHeaders() {
        return (req, res, next) => {
            // Security headers
            res.set({
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block',
                'Referrer-Policy': 'strict-origin-when-cross-origin',
                'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            });
            // CORS headers for API
            if (req.path.startsWith('/api/')) {
                res.set({
                    'Access-Control-Allow-Origin': process.env.FRONTEND_URL || 'http://localhost:3000',
                    'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Requested-With',
                    'Access-Control-Allow-Credentials': 'true',
                });
            }
            next();
        };
    }
    // Request timeout middleware
    requestTimeout(timeoutMs = 30000) {
        return (req, res, next) => {
            const timeout = setTimeout(() => {
                if (!res.headersSent) {
                    res.status(408).json({
                        error: 'Request timeout',
                        timeout: timeoutMs,
                    });
                }
            }, timeoutMs);
            res.on('finish', () => {
                clearTimeout(timeout);
            });
            next();
        };
    }
    // Enhanced error handling middleware
    errorHandler() {
        return (error, req, res, next) => {
            const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            const isDevelopment = process.env.NODE_ENV === 'development';
            // Log comprehensive error details
            console.error('API Error:', {
                errorId,
                requestId: req.requestId,
                path: req.path,
                method: req.method,
                userAgent: req.get('User-Agent'),
                ip: req.ip,
                timestamp: new Date().toISOString(),
                error: {
                    name: error.name,
                    message: error.message,
                    stack: isDevelopment ? error.stack : undefined,
                    code: error.code,
                    status: error.status,
                },
                requestBody: req.method !== 'GET' ? req.body : undefined,
                queryParams: req.query,
            });
            // Determine error type and response
            let statusCode = 500;
            let errorType = 'INTERNAL_SERVER_ERROR';
            let message = 'Something went wrong';
            let details = undefined;
            // Validation errors
            if (error.name === 'ValidationError' || error.status === 400) {
                statusCode = 400;
                errorType = 'VALIDATION_ERROR';
                message = 'Invalid request data';
                details = isDevelopment ? error.details || error.errors : undefined;
            }
            // Authentication errors
            else if (error.name === 'UnauthorizedError' || error.status === 401) {
                statusCode = 401;
                errorType = 'AUTHENTICATION_ERROR';
                message = 'Authentication required';
            }
            // Authorization errors
            else if (error.status === 403) {
                statusCode = 403;
                errorType = 'AUTHORIZATION_ERROR';
                message = 'Access denied';
            }
            // Not found errors
            else if (error.status === 404) {
                statusCode = 404;
                errorType = 'NOT_FOUND_ERROR';
                message = 'Resource not found';
            }
            // Rate limiting errors
            else if (error.status === 429) {
                statusCode = 429;
                errorType = 'RATE_LIMIT_ERROR';
                message = 'Too many requests';
            }
            // Database connection errors
            else if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
                statusCode = 503;
                errorType = 'SERVICE_UNAVAILABLE';
                message = 'Service temporarily unavailable';
            }
            // Database constraint errors
            else if (error.code === 'P2002') { // Prisma unique constraint
                statusCode = 409;
                errorType = 'CONFLICT_ERROR';
                message = 'Resource already exists';
            }
            // Timeout errors
            else if (error.code === 'ETIMEDOUT') {
                statusCode = 408;
                errorType = 'TIMEOUT_ERROR';
                message = 'Request timeout';
            }
            // File size errors
            else if (error.code === 'LIMIT_FILE_SIZE') {
                statusCode = 413;
                errorType = 'FILE_TOO_LARGE';
                message = 'File size exceeds limit';
            }
            // Custom application errors
            else if (error.isOperational) {
                statusCode = error.statusCode || 400;
                errorType = error.type || 'APPLICATION_ERROR';
                message = error.message;
                details = isDevelopment ? error.details : undefined;
            }
            // Development mode - show more details
            else if (isDevelopment) {
                message = error.message || 'Internal server error';
                details = {
                    stack: error.stack,
                    code: error.code,
                    name: error.name,
                };
            }
            // Security headers for error responses
            res.set({
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
            });
            // Send error response
            const errorResponse = {
                success: false,
                error: {
                    type: errorType,
                    message,
                    code: statusCode,
                    errorId,
                    timestamp: new Date().toISOString(),
                    ...(details && { details }),
                    ...(isDevelopment && { requestId: req.requestId }),
                },
            };
            res.status(statusCode).json(errorResponse);
        };
    }
    // Health check endpoint
    healthCheck() {
        return async (req, res) => {
            const start = perf_hooks_1.performance.now();
            try {
                // Check database
                const { getDatabaseOptimizationService } = await Promise.resolve().then(() => __importStar(require('../services/databaseOptimizationService')));
                const dbService = getDatabaseOptimizationService();
                const dbHealth = dbService ? await dbService.healthCheck() : { status: 'unknown' };
                // Check cache
                const cache = (0, cacheService_1.getCacheService)();
                const cacheHealth = cache ? await cache.getInfo() : null;
                const responseTime = perf_hooks_1.performance.now() - start;
                res.json({
                    status: 'healthy',
                    timestamp: new Date().toISOString(),
                    responseTime: `${responseTime.toFixed(2)}ms`,
                    services: {
                        database: dbHealth.status,
                        cache: cacheHealth ? 'healthy' : 'unavailable',
                    },
                    metrics: this.getMetricsSummary(),
                });
            }
            catch (error) {
                res.status(503).json({
                    status: 'unhealthy',
                    error: error instanceof Error ? error.message : 'Unknown error',
                });
            }
        };
    }
    addRequestMetric(metric) {
        this.requestMetrics.push(metric);
        if (this.requestMetrics.length > this.maxMetricsHistory) {
            this.requestMetrics = this.requestMetrics.slice(-this.maxMetricsHistory);
        }
    }
    getMetricsSummary() {
        const recentMetrics = this.requestMetrics.filter(m => m.timestamp > Date.now() - 60 * 60 * 1000 // Last hour
        );
        if (recentMetrics.length === 0) {
            return { requests: 0, avgResponseTime: 0, errorRate: 0 };
        }
        const totalRequests = recentMetrics.length;
        const avgResponseTime = recentMetrics.reduce((sum, m) => sum + m.duration, 0) / totalRequests;
        const errorRequests = recentMetrics.filter(m => m.statusCode >= 400).length;
        const errorRate = (errorRequests / totalRequests) * 100;
        return {
            requests: totalRequests,
            avgResponseTime: parseFloat(avgResponseTime.toFixed(2)),
            errorRate: parseFloat(errorRate.toFixed(2)),
        };
    }
    getMetrics() {
        return {
            recent: this.getMetricsSummary(),
            all: this.requestMetrics,
        };
    }
}
exports.OptimizationMiddleware = OptimizationMiddleware;
// Singleton instance
const optimizationMiddleware = new OptimizationMiddleware();
exports.default = optimizationMiddleware;
//# sourceMappingURL=optimizationMiddleware.js.map