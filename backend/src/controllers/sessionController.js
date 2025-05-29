/**
 * Session Controller
 * Handles session management operations
 */

const prisma = require('../utils/prismaClient');
const { getDeviceInfo, detectSuspiciousActivity } = require('../middleware/sessionMiddleware');
const { createError } = require('../middleware/errorHandler');

/**
 * Get all active sessions for the current user
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getUserSessions = async (req, res, next) => {
  try {
    const userId = req.user.id;
    const currentSessionId = req.sessionId;
    
    // Get all active sessions for the user
    const sessions = await prisma.session.findMany({
      where: {
        userId,
        isValid: true
      },
      orderBy: {
        createdAt: 'desc'
      }
    });
    
    // Transform and mark current session
    const formattedSessions = sessions.map(session => ({
      id: session.id,
      device: session.device,
      ipAddress: session.ipAddress,
      createdAt: session.createdAt,
      lastActive: session.lastActive,
      expiresAt: session.expiresAt,
      isCurrentSession: session.id === currentSessionId
    }));
    
    res.json({
      success: true,
      data: formattedSessions
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Get details of a specific session
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getSessionById = async (req, res, next) => {
  try {
    const userId = req.user.id;
    const sessionId = req.params.id;
    const currentSessionId = req.sessionId;
    
    // Find the session
    const session = await prisma.session.findFirst({
      where: {
        id: sessionId,
        userId
      }
    });
    
    if (!session) {
      return next(createError('Session not found', 404));
    }
    
    // Check for suspicious activity
    const isSuspicious = detectSuspiciousActivity(session, req);
    
    // Transform the session data
    const formattedSession = {
      id: session.id,
      device: session.device,
      ipAddress: session.ipAddress,
      createdAt: session.createdAt,
      lastActive: session.lastActive,
      expiresAt: session.expiresAt,
      isValid: session.isValid,
      isCurrentSession: session.id === currentSessionId,
      suspiciousActivity: isSuspicious
    };
    
    res.json({
      success: true,
      data: formattedSession
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Revoke a specific session
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const revokeSession = async (req, res, next) => {
  try {
    const userId = req.user.id;
    const sessionId = req.params.id;
    const currentSessionId = req.sessionId;
    
    // Find the session
    const session = await prisma.session.findFirst({
      where: {
        id: sessionId,
        userId
      }
    });
    
    if (!session) {
      return next(createError('Session not found', 404));
    }
    
    // Check if trying to revoke current session
    if (session.id === currentSessionId) {
      return next(createError('Cannot revoke current session. Use logout instead.', 400));
    }
    
    // Invalidate the session
    await prisma.session.update({
      where: { id: sessionId },
      data: { isValid: false }
    });
    
    res.json({
      success: true,
      message: 'Session revoked successfully'
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Revoke all sessions except the current one
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const revokeAllSessions = async (req, res, next) => {
  try {
    const userId = req.user.id;
    const currentSessionId = req.sessionId;
    
    // Revoke all sessions except the current one
    const result = await prisma.session.updateMany({
      where: {
        userId,
        id: { not: currentSessionId },
        isValid: true
      },
      data: {
        isValid: false
      }
    });
    
    res.json({
      success: true,
      message: `${result.count} sessions revoked successfully`
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  getUserSessions,
  getSessionById,
  revokeSession,
  revokeAllSessions
}; 