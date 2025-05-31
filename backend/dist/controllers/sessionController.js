"use strict";
/**
 * Session Controller
 * Handles session management, device tracking, and session security
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.revokeAllSessions = exports.revokeSession = exports.checkSession = exports.getUserSessions = void 0;
const sessionManager_1 = __importDefault(require("../utils/sessionManager"));
const errorHandler_1 = require("../middleware/errorHandler");
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
const logger_1 = require("../utils/logger");
// Create logger
const logger = (0, logger_1.createLogger)('SessionController');
/**
 * Get all active sessions for the current user
 * @route GET /api/sessions
 * @access Private
 */
exports.getUserSessions = (0, errorHandler_1.asyncHandler)(async (req, res) => {
    if (!req.user) {
        throw new errorHandler_1.AppError('Not authenticated', 401);
    }
    const sessions = await sessionManager_1.default.getUserSessions(req.user.id);
    // Map sessions to a more user-friendly format with device info
    const sessionsData = sessions.map(session => ({
        id: session.id,
        device: parseUserAgent(session.userAgent || 'Unknown'),
        ipAddress: session.ipAddress,
        lastActive: session.lastActiveAt,
        createdAt: session.createdAt,
        isCurrentSession: req.user?.sessionId === session.id,
    }));
    res.status(200).json({
        success: true,
        data: sessionsData
    });
});
/**
 * Check current session status
 * @route GET /api/sessions/check
 * @access Private
 */
exports.checkSession = (0, errorHandler_1.asyncHandler)(async (req, res) => {
    if (!req.user || !req.user.sessionId) {
        throw new errorHandler_1.AppError('Not authenticated', 401);
    }
    // Get session info
    const session = await prismaClient_1.default.session.findUnique({
        where: { id: req.user.sessionId }
    });
    if (!session) {
        throw new errorHandler_1.AppError('Session not found', 404);
    }
    // Check for suspicious activity
    const suspiciousActivity = req.suspiciousActivity || false;
    res.status(200).json({
        success: true,
        data: {
            active: session.isValid && new Date() < session.expiresAt,
            expiresAt: session.expiresAt,
            lastActive: session.lastActiveAt,
            rememberMe: session.rememberMe,
            suspiciousActivity
        }
    });
});
/**
 * Revoke a specific session
 * @route DELETE /api/sessions/:sessionId
 * @access Private
 */
exports.revokeSession = (0, errorHandler_1.asyncHandler)(async (req, res) => {
    if (!req.user) {
        throw new errorHandler_1.AppError('Not authenticated', 401);
    }
    const { sessionId } = req.params;
    // Prevent revoking current session through this endpoint
    if (req.user.sessionId === sessionId) {
        throw new errorHandler_1.AppError('Cannot revoke current session. Use logout instead.', 400);
    }
    // Check if session belongs to the user
    const session = await prismaClient_1.default.session.findFirst({
        where: {
            id: sessionId,
            userId: req.user.id
        }
    });
    if (!session) {
        throw new errorHandler_1.AppError('Session not found or does not belong to user', 404);
    }
    // Invalidate the session
    await prismaClient_1.default.session.update({
        where: { id: sessionId },
        data: { isValid: false }
    });
    res.status(200).json({
        success: true,
        message: 'Session revoked successfully'
    });
});
/**
 * Revoke all other sessions except current one
 * @route DELETE /api/sessions
 * @access Private
 */
exports.revokeAllSessions = (0, errorHandler_1.asyncHandler)(async (req, res) => {
    if (!req.user || !req.user.sessionId) {
        throw new errorHandler_1.AppError('Not authenticated', 401);
    }
    const currentSessionId = req.user.sessionId;
    // Invalidate all other sessions
    const result = await prismaClient_1.default.session.updateMany({
        where: {
            userId: req.user.id,
            id: { not: currentSessionId },
            isValid: true
        },
        data: { isValid: false }
    });
    res.status(200).json({
        success: true,
        message: `${result.count} sessions revoked successfully`
    });
});
/**
 * Helper function to parse user agent into a readable device description
 * @param userAgent - Browser user agent string
 * @returns Readable device description
 */
function parseUserAgent(userAgent) {
    // Simple parsing - in production, use a more sophisticated user agent parser
    if (!userAgent || userAgent === 'unknown')
        return 'Unknown device';
    let device = 'Unknown device';
    // Mobile detection
    if (userAgent.includes('iPhone') || userAgent.includes('iPad')) {
        device = userAgent.includes('iPad') ? 'iPad' : 'iPhone';
    }
    else if (userAgent.includes('Android')) {
        device = 'Android device';
    }
    else if (userAgent.includes('Windows Phone')) {
        device = 'Windows Phone';
    }
    // Desktop detection
    else if (userAgent.includes('Windows')) {
        device = 'Windows computer';
    }
    else if (userAgent.includes('Macintosh') || userAgent.includes('Mac OS')) {
        device = 'Mac computer';
    }
    else if (userAgent.includes('Linux')) {
        device = 'Linux computer';
    }
    // Browser detection
    let browser = 'Unknown browser';
    if (userAgent.includes('Chrome') && !userAgent.includes('Chromium')) {
        browser = 'Chrome';
    }
    else if (userAgent.includes('Firefox')) {
        browser = 'Firefox';
    }
    else if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) {
        browser = 'Safari';
    }
    else if (userAgent.includes('Edge')) {
        browser = 'Edge';
    }
    else if (userAgent.includes('MSIE') || userAgent.includes('Trident/')) {
        browser = 'Internet Explorer';
    }
    return `${device} (${browser})`;
}
exports.default = {
    getUserSessions: exports.getUserSessions,
    checkSession: exports.checkSession,
    revokeSession: exports.revokeSession,
    revokeAllSessions: exports.revokeAllSessions
};
//# sourceMappingURL=sessionController.js.map