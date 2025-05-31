"use strict";
/**
 * Session Manager
 * Handles session creation, validation, and management
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.generateDeviceFingerprint = exports.cleanupExpiredSessions = exports.extendSession = exports.getUserSessions = exports.revokeAllSessions = exports.invalidateSession = exports.updateSessionActivity = exports.validateSession = exports.createSession = void 0;
const prismaClient_1 = __importDefault(require("./prismaClient"));
const uuid_1 = require("uuid");
// Environment variables
const { JWT_SECRET = 'your-secret-key', JWT_EXPIRY = '1h', SESSION_EXPIRY = '24h' } = process.env;
// Session configuration with defaults
const sessionConfig = {
    maxAge: parseInt(process.env.SESSION_MAX_AGE || '3600000', 10), // 1 hour
    rememberMeMaxAge: parseInt(process.env.REMEMBER_ME_MAX_AGE || '2592000000', 10), // 30 days
    inactivityTimeout: parseInt(process.env.SESSION_INACTIVITY_TIMEOUT || '1800000', 10), // 30 minutes
};
/**
 * Create a new session for a user
 * @param userId - User ID
 * @param req - Express request object for extracting client info
 * @param rememberMe - Whether to extend session duration
 * @returns Session with tokens
 */
const createSession = async (userId, req, rememberMe = false) => {
    // Generate JWT tokens
    const token = generateToken(userId);
    const refreshToken = generateRefreshToken(userId);
    // Calculate expiry
    const expiryMs = rememberMe ? sessionConfig.rememberMeMaxAge : sessionConfig.maxAge;
    const expiresAt = new Date(Date.now() + expiryMs);
    // Get client information
    const ipAddress = req.ip || req.socket.remoteAddress || 'unknown';
    const userAgent = req.headers['user-agent'] || 'unknown';
    // Generate or retrieve device ID (could be from cookies or headers)
    let deviceId = req.cookies.deviceId || req.headers['x-device-id'];
    if (!deviceId) {
        deviceId = (0, uuid_1.v4)(); // Generate a new device ID if none exists
    }
    // Create session record
    const session = await prismaClient_1.default.session.create({
        data: {
            userId,
            token,
            refreshToken,
            ipAddress,
            userAgent,
            deviceId,
            expiresAt,
            rememberMe,
            metadata: {
                // Additional metadata about the session
                createdFrom: req.headers.origin || 'unknown',
                lastRoute: req.path,
            },
        },
    });
    return { token, refreshToken, session };
};
exports.createSession = createSession;
/**
 * Validate a session
 * @param token - JWT token
 * @returns Valid session or null
 */
const validateSession = async (token) => {
    // First verify the JWT token
    const decoded = verifyToken(token);
    if (!decoded) {
        return null;
    }
    // Check if session exists and is valid
    const session = await prismaClient_1.default.session.findFirst({
        where: {
            token,
            isValid: true,
            expiresAt: {
                gt: new Date(), // Not expired
            },
        },
    });
    if (!session) {
        return null;
    }
    // Check for inactivity timeout
    const lastActiveThreshold = new Date(Date.now() - sessionConfig.inactivityTimeout);
    if (session.lastActiveAt < lastActiveThreshold) {
        // Session is inactive for too long, invalidate it
        await (0, exports.invalidateSession)(token);
        return null;
    }
    // Update last active timestamp
    await (0, exports.updateSessionActivity)(token);
    return session;
};
exports.validateSession = validateSession;
/**
 * Update session's last activity timestamp
 * @param token - JWT token
 */
const updateSessionActivity = async (token) => {
    await prismaClient_1.default.session.updateMany({
        where: { token },
        data: { lastActiveAt: new Date() },
    });
};
exports.updateSessionActivity = updateSessionActivity;
/**
 * Invalidate a single session
 * @param token - JWT token
 * @returns Success indicator
 */
const invalidateSession = async (token) => {
    try {
        await prismaClient_1.default.session.updateMany({
            where: { token },
            data: { isValid: false },
        });
        return true;
    }
    catch (error) {
        console.error('Error invalidating session:', error);
        return false;
    }
};
exports.invalidateSession = invalidateSession;
/**
 * Revoke all sessions for a user except current one
 * @param userId - User ID
 * @param currentToken - Current session token to preserve
 * @returns Number of sessions invalidated
 */
const revokeAllSessions = async (userId, currentToken) => {
    const result = await prismaClient_1.default.session.updateMany({
        where: {
            userId,
            ...(currentToken ? { token: { not: currentToken } } : {}),
            isValid: true,
        },
        data: { isValid: false },
    });
    return result.count;
};
exports.revokeAllSessions = revokeAllSessions;
/**
 * Get all active sessions for a user
 * @param userId - User ID
 * @returns List of active sessions
 */
const getUserSessions = async (userId) => {
    return prismaClient_1.default.session.findMany({
        where: {
            userId,
            isValid: true,
            expiresAt: {
                gt: new Date(),
            },
        },
        orderBy: {
            lastActiveAt: 'desc',
        },
    });
};
exports.getUserSessions = getUserSessions;
/**
 * Extend session duration (e.g., after user activity)
 * @param token - JWT token
 * @param extendBy - Milliseconds to extend by (defaults to session max age)
 */
const extendSession = async (token, extendBy) => {
    const session = await prismaClient_1.default.session.findUnique({
        where: { token },
    });
    if (!session)
        return;
    const extension = extendBy || (session.rememberMe
        ? sessionConfig.rememberMeMaxAge
        : sessionConfig.maxAge);
    const newExpiryDate = new Date(Date.now() + extension);
    await prismaClient_1.default.session.update({
        where: { token },
        data: {
            expiresAt: newExpiryDate,
            lastActiveAt: new Date()
        },
    });
};
exports.extendSession = extendSession;
/**
 * Clean up expired sessions (can be run as a scheduled job)
 */
const cleanupExpiredSessions = async () => {
    const result = await prismaClient_1.default.session.deleteMany({
        where: {
            OR: [
                { expiresAt: { lt: new Date() } },
                { isValid: false },
            ],
        },
    });
    return result.count;
};
exports.cleanupExpiredSessions = cleanupExpiredSessions;
/**
 * Generate fingerprint for device identification
 * @param req - Express request
 * @returns Device fingerprint string
 */
const generateDeviceFingerprint = (req) => {
    const components = [
        req.headers['user-agent'] || 'unknown',
        req.ip || req.socket.remoteAddress || 'unknown',
        req.headers['accept-language'] || 'unknown',
    ];
    // Simple fingerprint generation - in production, use a more sophisticated approach
    return Buffer.from(components.join('|')).toString('base64');
};
exports.generateDeviceFingerprint = generateDeviceFingerprint;
exports.default = {
    createSession: exports.createSession,
    validateSession: exports.validateSession,
    updateSessionActivity: exports.updateSessionActivity,
    extendSession: exports.extendSession,
    invalidateSession: exports.invalidateSession,
    refreshSession,
    getUserSessions: exports.getUserSessions,
    cleanupExpiredSessions: exports.cleanupExpiredSessions,
    generateDeviceFingerprint: exports.generateDeviceFingerprint,
    getSessionMetadata,
    updateSessionMetadata,
    sessionConfig
};
//# sourceMappingURL=sessionManager.js.map