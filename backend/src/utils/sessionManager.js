/**
 * Session Manager
 * Handles session creation, validation, and management
 */

const prisma = require('./prismaClient');
const { v4: uuidv4 } = require('uuid');
const { generateAccessToken, generateRefreshToken } = require('./jwt');
const { getDeviceInfo } = require('../middleware/sessionMiddleware');

/**
 * Create a new session for a user
 * @param {object} user - User object
 * @param {object} req - Express request object
 * @param {boolean} rememberMe - Whether to extend session lifetime
 * @returns {Promise<object>} Session data with tokens
 */
const createSession = async (user, req, rememberMe = false) => {
  try {
    const deviceInfo = getDeviceInfo(req.headers['user-agent']);
    const ipAddress = req.ip || req.connection.remoteAddress;
    
    // Default session expiry (24 hours)
    let expiresAt = new Date();
    expiresAt.setHours(expiresAt.getHours() + 24);
    
    // If remember me is enabled, extend to 30 days
    if (rememberMe) {
      expiresAt = new Date();
      expiresAt.setDate(expiresAt.getDate() + 30);
    }
    
    // Create new session record
    const session = await prisma.session.create({
      data: {
        id: uuidv4(),
        userId: user.id,
        ipAddress,
        device: deviceInfo,
        expiresAt,
        lastActive: new Date(),
        isValid: true,
        deviceId: req.deviceId || null
      }
    });
    
    // Generate tokens
    const accessToken = generateAccessToken(user, session.id, {
      expiresIn: rememberMe ? '1h' : '15m'
    });
    
    const refreshToken = generateRefreshToken(user, session.id, {
      expiresIn: rememberMe ? '30d' : '7d'
    });
    
    return {
      session,
      accessToken,
      refreshToken
    };
  } catch (error) {
    console.error('Error creating session:', error);
    throw new Error('Failed to create session');
  }
};

/**
 * Validate a session
 * @param {string} sessionId - Session ID
 * @param {string} userId - User ID
 * @returns {Promise<boolean>} Whether the session is valid
 */
const validateSession = async (sessionId, userId) => {
  try {
    const session = await prisma.session.findFirst({
      where: {
        id: sessionId,
        userId,
        isValid: true,
        expiresAt: {
          gt: new Date()
        }
      }
    });
    
    return !!session;
  } catch (error) {
    console.error('Error validating session:', error);
    return false;
  }
};

/**
 * Check for session inactivity timeout
 * @param {string} sessionId - Session ID
 * @param {number} timeoutMinutes - Timeout in minutes
 * @returns {Promise<boolean>} Whether the session has timed out
 */
const checkSessionTimeout = async (sessionId, timeoutMinutes = 60) => {
  try {
    const session = await prisma.session.findUnique({
      where: { id: sessionId }
    });
    
    if (!session) {
      return true; // Session not found, consider it timed out
    }
    
    const lastActive = new Date(session.lastActive);
    const now = new Date();
    const diffMs = now - lastActive;
    const diffMinutes = Math.floor(diffMs / 1000 / 60);
    
    return diffMinutes > timeoutMinutes;
  } catch (error) {
    console.error('Error checking session timeout:', error);
    return true; // Error, consider it timed out
  }
};

/**
 * Update session last active time
 * @param {string} sessionId - Session ID
 * @returns {Promise<boolean>} Success indicator
 */
const updateSessionActivity = async (sessionId) => {
  try {
    await prisma.session.update({
      where: { id: sessionId },
      data: { lastActive: new Date() }
    });
    
    return true;
  } catch (error) {
    console.error('Error updating session activity:', error);
    return false;
  }
};

/**
 * Extend session timeout
 * @param {string} sessionId - Session ID
 * @param {number} addHours - Hours to add to expiry
 * @returns {Promise<boolean>} Success indicator
 */
const extendSession = async (sessionId, addHours = 24) => {
  try {
    const session = await prisma.session.findUnique({
      where: { id: sessionId }
    });
    
    if (!session) {
      return false;
    }
    
    const newExpiresAt = new Date(session.expiresAt);
    newExpiresAt.setHours(newExpiresAt.getHours() + addHours);
    
    await prisma.session.update({
      where: { id: sessionId },
      data: { expiresAt: newExpiresAt }
    });
    
    return true;
  } catch (error) {
    console.error('Error extending session:', error);
    return false;
  }
};

/**
 * Invalidate a session
 * @param {string} sessionId - Session ID
 * @returns {Promise<boolean>} Success indicator
 */
const invalidateSession = async (sessionId) => {
  try {
    await prisma.session.update({
      where: { id: sessionId },
      data: { isValid: false }
    });
    
    return true;
  } catch (error) {
    console.error('Error invalidating session:', error);
    return false;
  }
};

/**
 * Invalidate all sessions for a user except the current one
 * @param {string} userId - User ID
 * @param {string} currentSessionId - Current session ID to exclude
 * @returns {Promise<number>} Number of sessions invalidated
 */
const invalidateAllUserSessions = async (userId, currentSessionId = null) => {
  try {
    const where = {
      userId,
      isValid: true
    };
    
    if (currentSessionId) {
      where.id = { not: currentSessionId };
    }
    
    const result = await prisma.session.updateMany({
      where,
      data: { isValid: false }
    });
    
    return result.count;
  } catch (error) {
    console.error('Error invalidating user sessions:', error);
    return 0;
  }
};

/**
 * Check for suspicious session activity
 * @param {string} sessionId - Session ID
 * @param {object} req - Express request object
 * @returns {Promise<boolean>} Whether suspicious activity was detected
 */
const checkSuspiciousActivity = async (sessionId, req) => {
  try {
    const session = await prisma.session.findUnique({
      where: { id: sessionId }
    });
    
    if (!session) {
      return true; // Session not found, consider it suspicious
    }
    
    // Check IP address
    const currentIp = req.ip || req.connection.remoteAddress;
    if (session.ipAddress && session.ipAddress !== currentIp) {
      // IP changed - could be suspicious
      return true;
    }
    
    // Check device ID cookie
    if (session.deviceId && req.deviceId && session.deviceId !== req.deviceId) {
      // Device ID changed - could be suspicious
      return true;
    }
    
    // Check user agent / device type
    const currentDevice = getDeviceInfo(req.headers['user-agent']);
    if (session.device && session.device.type !== 'unknown' && 
        currentDevice.type !== 'unknown' && 
        session.device.type !== currentDevice.type) {
      // Device type changed - could be suspicious
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('Error checking for suspicious activity:', error);
    return true; // Error, consider it suspicious
  }
};

/**
 * Refresh a session and generate new tokens
 * @param {string} sessionId - Session ID
 * @param {object} user - User object
 * @param {boolean} rememberMe - Whether to extend token lifetimes
 * @returns {Promise<object>} New tokens
 */
const refreshSession = async (sessionId, user, rememberMe = false) => {
  try {
    // Update session last active time
    await updateSessionActivity(sessionId);
    
    // Generate new tokens
    const accessToken = generateAccessToken(user, sessionId, {
      expiresIn: rememberMe ? '1h' : '15m'
    });
    
    const refreshToken = generateRefreshToken(user, sessionId, {
      expiresIn: rememberMe ? '30d' : '7d'
    });
    
    return {
      accessToken,
      refreshToken
    };
  } catch (error) {
    console.error('Error refreshing session:', error);
    throw new Error('Failed to refresh session');
  }
};

/**
 * Get session metadata for analytics
 * @param {string} sessionId - Session ID
 * @returns {Promise<object>} Session metadata
 */
const getSessionMetadata = async (sessionId) => {
  try {
    const session = await prisma.session.findUnique({
      where: { id: sessionId },
      select: {
        id: true,
        ipAddress: true,
        device: true,
        createdAt: true,
        lastActive: true,
        deviceId: true
      }
    });
    
    return session || null;
  } catch (error) {
    console.error('Error getting session metadata:', error);
    return null;
  }
};

/**
 * Update session metadata
 * @param {string} sessionId - Session ID
 * @param {object} metadata - Metadata to update
 * @returns {Promise<boolean>} Success indicator
 */
const updateSessionMetadata = async (sessionId, metadata) => {
  try {
    await prisma.session.update({
      where: { id: sessionId },
      data: { ...metadata }
    });
    
    return true;
  } catch (error) {
    console.error('Error updating session metadata:', error);
    return false;
  }
};

module.exports = {
  createSession,
  validateSession,
  checkSessionTimeout,
  updateSessionActivity,
  extendSession,
  invalidateSession,
  invalidateAllUserSessions,
  checkSuspiciousActivity,
  refreshSession,
  getSessionMetadata,
  updateSessionMetadata
}; 