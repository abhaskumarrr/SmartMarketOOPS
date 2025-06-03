"use strict";
/**
 * Authentication Middleware
 * Validates JWT tokens and protects routes
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.hasRole = exports.isAuthenticated = exports.requireAnyPermission = exports.requirePermission = exports.requireRole = exports.requireVerified = exports.verifyRefreshToken = exports.protect = exports.csrfProtection = exports.authRateLimiter = void 0;
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
const jwt_1 = require("../utils/jwt");
const env_1 = __importDefault(require("../utils/env"));
const express_rate_limit_1 = __importDefault(require("express-rate-limit"));
const csurf_1 = __importDefault(require("csurf"));
const authorizationService_1 = __importDefault(require("../services/authorizationService"));
const logger_1 = require("../utils/logger");
// Create logger
const logger = (0, logger_1.createLogger)('AuthMiddleware');
/**
 * Rate limiter for authentication routes
 * Limits login, register, and password reset attempts
 */
exports.authRateLimiter = (0, express_rate_limit_1.default)({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 requests per IP
    standardHeaders: true,
    legacyHeaders: false,
    message: {
        success: false,
        message: 'Too many requests, please try again later.'
    }
});
/**
 * CSRF protection middleware
 */
exports.csrfProtection = (0, csurf_1.default)({
    cookie: {
        httpOnly: true,
        secure: env_1.default.NODE_ENV === 'production',
        sameSite: 'strict'
    }
});
/**
 * Middleware to protect routes
 * Verifies JWT token and attaches user to request
 */
const protect = async (req, res, next) => {
    try {
        // Get token from header using utility function
        const authHeader = req.headers.authorization;
        const token = (0, jwt_1.extractTokenFromHeader)(authHeader || '');
        if (!token) {
            res.status(401).json({
                success: false,
                message: 'Not authorized, no token provided'
            });
            return;
        }
        // Verify token
        const decoded = (0, jwt_1.verifyToken)(token);
        if (!decoded) {
            res.status(401).json({
                success: false,
                message: 'Not authorized, invalid token'
            });
            return;
        }
        // Check if token is expired (additional safety check)
        if ((0, jwt_1.isTokenExpired)(decoded)) {
            res.status(401).json({
                success: false,
                message: 'Token expired, please refresh'
            });
            return;
        }
        // Validate token type
        if (decoded.type !== 'access') {
            res.status(401).json({
                success: false,
                message: 'Invalid token type'
            });
            return;
        }
        // Get user from database
        const user = await prismaClient_1.default.user.findUnique({
            where: { id: decoded.id }
        });
        if (!user) {
            res.status(401).json({
                success: false,
                message: 'User not found'
            });
            return;
        }
        // Attach user to request
        req.user = {
            id: user.id,
            name: user.name,
            email: user.email,
            role: user.role,
            isVerified: user.isVerified
        };
        next();
    }
    catch (error) {
        console.error('Auth middleware error:', error);
        res.status(401).json({
            success: false,
            message: 'Not authorized, token failed',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.protect = protect;
/**
 * Middleware to verify refresh token
 * Used for token refresh routes
 */
const verifyRefreshToken = async (req, res, next) => {
    try {
        // Get refresh token from request body
        const { refreshToken } = req.body;
        if (!refreshToken) {
            res.status(401).json({
                success: false,
                message: 'Refresh token is required'
            });
            return;
        }
        // Verify refresh token
        const decoded = (0, jwt_1.verifyRefreshToken)(refreshToken);
        if (!decoded) {
            res.status(401).json({
                success: false,
                message: 'Invalid refresh token'
            });
            return;
        }
        // Get user from database
        const user = await prismaClient_1.default.user.findUnique({
            where: { id: decoded.id }
        });
        if (!user) {
            res.status(401).json({
                success: false,
                message: 'User not found'
            });
            return;
        }
        // Attach user to request
        req.user = {
            id: user.id,
            name: user.name,
            email: user.email,
            role: user.role,
            isVerified: user.isVerified
        };
        next();
    }
    catch (error) {
        console.error('Refresh token middleware error:', error);
        res.status(401).json({
            success: false,
            message: 'Invalid refresh token',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.verifyRefreshToken = verifyRefreshToken;
/**
 * Middleware to require email verification
 * Used for routes that require verified email
 */
const requireVerified = (req, res, next) => {
    if (!req.user) {
        res.status(401).json({
            success: false,
            message: 'Not authenticated'
        });
        return;
    }
    if (!req.user.isVerified) {
        res.status(403).json({
            success: false,
            message: 'Email verification required'
        });
        return;
    }
    next();
};
exports.requireVerified = requireVerified;
/**
 * Middleware to require specific role(s)
 * Used for routes that require specific user roles
 * @param roles - Array of allowed roles
 */
const requireRole = (roles) => {
    return (req, res, next) => {
        if (!req.user) {
            res.status(401).json({
                success: false,
                message: 'Not authenticated'
            });
            return;
        }
        if (!roles.includes(req.user.role)) {
            res.status(403).json({
                success: false,
                message: `Requires ${roles.join(' or ')} role`
            });
            return;
        }
        next();
    };
};
exports.requireRole = requireRole;
/**
 * Middleware to require specific permission(s)
 * Used for fine-grained access control based on permissions
 * @param permissions - Array of required permissions (all must be present)
 */
const requirePermission = (permissions) => {
    return (req, res, next) => {
        if (!req.user) {
            res.status(401).json({
                success: false,
                message: 'Not authenticated'
            });
            return;
        }
        if (!authorizationService_1.default.hasAllPermissions(req.user.role, permissions)) {
            res.status(403).json({
                success: false,
                message: `Insufficient permissions. Required: ${permissions.join(', ')}`
            });
            return;
        }
        next();
    };
};
exports.requirePermission = requirePermission;
/**
 * Middleware to require any of the specified permissions
 * At least one of the permissions must be present
 * @param permissions - Array of permissions (at least one must be present)
 */
const requireAnyPermission = (permissions) => {
    return (req, res, next) => {
        if (!req.user) {
            res.status(401).json({
                success: false,
                message: 'Not authenticated'
            });
            return;
        }
        if (!authorizationService_1.default.hasAnyPermission(req.user.role, permissions)) {
            res.status(403).json({
                success: false,
                message: `Insufficient permissions. Required any of: ${permissions.join(', ')}`
            });
            return;
        }
        next();
    };
};
exports.requireAnyPermission = requireAnyPermission;
/**
 * Middleware to verify user is authenticated
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
const isAuthenticated = async (req, res, next) => {
    try {
        // Get token from header
        const authHeader = req.headers.authorization;
        const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN format
        if (!token) {
            res.status(401).json({ error: 'No authentication token provided' });
            return;
        }
        // Verify token
        const secret = process.env.JWT_SECRET;
        if (!secret) {
            console.error('JWT_SECRET not configured');
            res.status(500).json({ error: 'Internal server error' });
            return;
        }
        try {
            const decoded = jsonwebtoken_1.default.verify(token, secret);
            // Find the session to validate it's still active
            const session = await prismaClient_1.default.session.findFirst({
                where: {
                    token,
                    userId: decoded.id,
                    isValid: true,
                    expiresAt: { gt: new Date() }
                },
                include: {
                    user: {
                        select: {
                            id: true,
                            email: true,
                            role: true
                        }
                    }
                }
            });
            if (!session) {
                res.status(401).json({ error: 'Invalid or expired session' });
                return;
            }
            // Update last active time
            await prismaClient_1.default.session.update({
                where: { id: session.id },
                data: { lastActiveAt: new Date() }
            });
            // Attach user to request
            req.user = {
                id: session.user.id,
                email: session.user.email,
                role: session.user.role
            };
            next();
        }
        catch (error) {
            res.status(401).json({ error: 'Invalid authentication token' });
        }
    }
    catch (error) {
        console.error('Authentication error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
};
exports.isAuthenticated = isAuthenticated;
/**
 * Middleware to verify user has required role
 * @param {string[]} roles - Array of allowed roles
 */
const hasRole = (roles) => {
    return (req, res, next) => {
        if (!req.user) {
            res.status(401).json({ error: 'Authentication required' });
            return;
        }
        if (!roles.includes(req.user.role)) {
            res.status(403).json({ error: 'Insufficient permissions' });
            return;
        }
        next();
    };
};
exports.hasRole = hasRole;
exports.default = {
    protect: exports.protect,
    verifyRefreshToken: exports.verifyRefreshToken,
    requireVerified: exports.requireVerified,
    requireRole: exports.requireRole,
    requirePermission: exports.requirePermission,
    requireAnyPermission: exports.requireAnyPermission,
    authRateLimiter: exports.authRateLimiter,
    csrfProtection: exports.csrfProtection,
    isAuthenticated: exports.isAuthenticated,
    hasRole: exports.hasRole
};
//# sourceMappingURL=auth.js.map