/**
 * Session Middleware
 * Handles session management and tracking
 */

const cookieParser = require('cookie-parser');
const { v4: uuidv4 } = require('uuid');
const UAParser = require('ua-parser-js');
const { env } = require('../utils/env');
const prisma = require('../utils/prismaClient');
const { verifyAccessToken } = require('../utils/jwt');

// Cookie options
const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: env.NODE_ENV === 'production',
  sameSite: 'strict',
  signed: true,
  maxAge: 365 * 24 * 60 * 60 * 1000, // 1 year
  domain: env.COOKIE_DOMAIN || undefined
};

// Device ID cookie name
const DEVICE_ID_COOKIE = 'device_id';

/**
 * Secure cookie parser middleware with signing
 */
const secureCookieParser = cookieParser(env.COOKIE_SECRET);

/**
 * Set device ID cookie for session tracking
 * This helps identify the same browser across sessions
 */
const setDeviceIdCookie = (req, res, next) => {
  try {
    // Check if device ID cookie exists
    const deviceId = req.signedCookies[DEVICE_ID_COOKIE];
    
    if (!deviceId) {
      // Generate new device ID if not present
      const newDeviceId = uuidv4();
      res.cookie(DEVICE_ID_COOKIE, newDeviceId, COOKIE_OPTIONS);
      req.deviceId = newDeviceId;
    } else {
      req.deviceId = deviceId;
    }
    
    next();
  } catch (error) {
    console.error('Error setting device ID cookie:', error);
    next();
  }
};

/**
 * Get device information from user agent
 * @param {string} userAgent - User agent string
 * @returns {object} Device information
 */
const getDeviceInfo = (userAgent) => {
  if (!userAgent) {
    return { type: 'unknown', browser: 'unknown', os: 'unknown' };
  }
  
  const parser = new UAParser(userAgent);
  const device = parser.getDevice();
  const browser = parser.getBrowser();
  const os = parser.getOS();
  
  return {
    type: device.type || (os.name === 'iOS' || os.name === 'Android' ? 'mobile' : 'desktop'),
    browser: `${browser.name || 'unknown'} ${browser.version || ''}`.trim(),
    os: `${os.name || 'unknown'} ${os.version || ''}`.trim()
  };
};

/**
 * Middleware to track session activity
 * Updates the last active time of the session
 */
const sessionActivity = async (req, res, next) => {
  try {
    // Get token from Authorization header
    let token;
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    }
    
    if (token) {
      try {
        // Verify token without throwing - we don't want to block the request
        const decoded = verifyAccessToken(token);
        
        if (decoded && decoded.sessionId) {
          // Update session last active time
          await prisma.session.update({
            where: { id: decoded.sessionId },
            data: { lastActive: new Date() }
          });
        }
      } catch (error) {
        // Don't block request for token issues
        console.debug('Session activity update skipped:', error.message);
      }
    }
    
    next();
  } catch (error) {
    // Don't block the request for session tracking issues
    console.error('Error tracking session activity:', error);
    next();
  }
};

/**
 * Detect suspicious activity in a session
 * @param {object} session - Session object
 * @param {object} req - Express request object
 * @returns {boolean} True if suspicious activity detected
 */
const detectSuspiciousActivity = (session, req) => {
  const userAgent = req.headers['user-agent'];
  
  // Different device type
  if (session.device && session.device.type !== 'unknown') {
    const currentDevice = getDeviceInfo(userAgent);
    if (currentDevice.type !== 'unknown' && currentDevice.type !== session.device.type) {
      return true;
    }
  }
  
  // IP address change from a different geographic location
  // This would require IP geolocation which is not implemented yet
  
  return false;
};

module.exports = {
  secureCookieParser,
  setDeviceIdCookie,
  sessionActivity,
  getDeviceInfo,
  detectSuspiciousActivity,
  COOKIE_OPTIONS,
  DEVICE_ID_COOKIE
}; 