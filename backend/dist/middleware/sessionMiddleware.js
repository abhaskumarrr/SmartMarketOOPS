"use strict";
/**
 * Session Middleware
 * Handles session validation, activity tracking, and timeout management
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.setRememberMeCookie = exports.setDeviceIdCookie = exports.validateUserSession = exports.sessionActivity = exports.secureCookieParser = exports.cookieOptions = void 0;
const sessionManager_1 = require("../utils/sessionManager");
const errorHandler_1 = require("./errorHandler");
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
const cookie_parser_1 = __importDefault(require("cookie-parser"));
const env_1 = __importDefault(require("../utils/env"));
// Secure cookie configuration
exports.cookieOptions = {
    httpOnly: true, // Cannot be accessed by JavaScript
    secure: env_1.default.NODE_ENV === 'production', // HTTPS only in production
    sameSite: 'strict', // Strict same-site policy
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
    path: '/', // Available across the site
    domain: env_1.default.NODE_ENV === 'production' ? env_1.default.COOKIE_DOMAIN : undefined,
};
/**
 * Cookie parser middleware with signing
 */
exports.secureCookieParser = (0, cookie_parser_1.default)(env_1.default.COOKIE_SECRET || 'SmartMarketOOPS-secret-key');
/**
 * Middleware to track user activity and update session
 */
const sessionActivity = async (req, res, next) => {
    try {
        // Only track activity for authenticated users
        if (!req.user || !req.headers.authorization) {
            return next();
        }
        const token = req.headers.authorization.split(' ')[1];
        if (!token) {
            return next();
        }
        // Update session last activity timestamp
        await (0, sessionManager_1.updateSessionActivity)(token);
        // Optionally extend session expiry time
        if (req.path !== '/api/auth/refresh-token') { // Don't extend during refresh
            await (0, sessionManager_1.extendSession)(token);
        }
        next();
    }
    catch (error) {
        // Don't fail the request if session tracking fails
        console.error('Session activity tracking error:', error);
        next();
    }
};
exports.sessionActivity = sessionActivity;
/**
 * Session validation middleware
 * More comprehensive than the basic JWT check
 */
const validateUserSession = async (req, res, next) => {
    try {
        // Get token from header
        const authHeader = req.headers.authorization;
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            return next(new errorHandler_1.AppError('Not authorized, no token', 401));
        }
        // Extract token
        const token = authHeader.split(' ')[1];
        // Validate session
        const session = await (0, sessionManager_1.validateSession)(token);
        if (!session) {
            return next(new errorHandler_1.AppError('Session expired or invalid', 401));
        }
        // Get user from database
        const user = await prismaClient_1.default.user.findUnique({
            where: { id: session.userId }
        });
        if (!user) {
            return next(new errorHandler_1.AppError('User not found', 401));
        }
        // Device fingerprint verification (optional security enhancement)
        const currentFingerprint = (0, sessionManager_1.generateDeviceFingerprint)(req);
        const storedFingerprint = session.deviceId;
        // If we have a stored fingerprint and it doesn't match (potential session hijacking)
        if (storedFingerprint && currentFingerprint !== storedFingerprint && env_1.default.NODE_ENV === 'production') {
            console.warn(`Suspicious session activity detected: ${user.email}, IP: ${req.ip}`);
            // Depending on security requirements, you might:
            // 1. Just log the suspicious activity
            // 2. Require additional verification
            // 3. Invalidate the session completely
            // For demonstration, we'll just attach a warning but allow the request
            req.suspiciousActivity = true;
        }
        // Attach user to request
        req.user = {
            id: user.id,
            name: user.name,
            email: user.email,
            role: user.role,
            isVerified: user.isVerified,
            sessionId: session.id
        };
        next();
    }
    catch (error) {
        next(new errorHandler_1.AppError('Session validation failed', 401));
    }
};
exports.validateUserSession = validateUserSession;
/**
 * Middleware to set session tracking cookie
 */
const setDeviceIdCookie = (req, res, next) => {
    // Check if device ID cookie exists
    if (!req.cookies.deviceId) {
        // Generate device ID
        const deviceId = (0, sessionManager_1.generateDeviceFingerprint)(req);
        // Set cookie
        res.cookie('deviceId', deviceId, {
            ...exports.cookieOptions,
            maxAge: 365 * 24 * 60 * 60 * 1000, // 1 year
        });
    }
    next();
};
exports.setDeviceIdCookie = setDeviceIdCookie;
/**
 * Remember Me cookie handling
 */
const setRememberMeCookie = (req, res, rememberMe) => {
    if (rememberMe) {
        res.cookie('rememberMe', 'true', {
            ...exports.cookieOptions,
            maxAge: 30 * 24 * 60 * 60 * 1000, // 30 days
        });
    }
    else {
        res.clearCookie('rememberMe');
    }
};
exports.setRememberMeCookie = setRememberMeCookie;
exports.default = {
    secureCookieParser: exports.secureCookieParser,
    sessionActivity: exports.sessionActivity,
    validateUserSession: exports.validateUserSession,
    setDeviceIdCookie: exports.setDeviceIdCookie,
    setRememberMeCookie: exports.setRememberMeCookie,
    cookieOptions: exports.cookieOptions
};
//# sourceMappingURL=sessionMiddleware.js.map