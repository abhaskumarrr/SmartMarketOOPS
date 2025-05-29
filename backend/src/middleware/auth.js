/**
 * Authentication Middleware
 * Handles user authentication and authorization
 */

const { verifyAccessToken } = require('../utils/jwt');
const { createError } = require('./errorHandler');
const prisma = require('../utils/prismaClient');
const { checkSessionTimeout } = require('../utils/sessionManager');

/**
 * Authentication middleware
 * Verifies the JWT token and attaches the user to the request
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const auth = async (req, res, next) => {
  try {
    // Get token from Authorization header
    let token;
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
    }
    
    // Check if token exists
    if (!token) {
      return next(createError('Authentication required. Please log in.', 401));
    }
    
    // Verify token
    const decoded = verifyAccessToken(token);
    
    // Check if session is valid
    if (decoded.sessionId) {
      const sessionTimedOut = await checkSessionTimeout(decoded.sessionId);
      if (sessionTimedOut) {
        return next(createError('Session expired. Please log in again.', 401));
      }
      
      // Set session ID on request
      req.sessionId = decoded.sessionId;
    }
    
    // Get user from database
    const user = await prisma.user.findUnique({
      where: { id: decoded.sub || decoded.userId }
    });
    
    // Check if user exists
    if (!user) {
      return next(createError('The user associated with this token no longer exists.', 401));
    }
    
    // Attach user to request
    req.user = user;
    
    next();
  } catch (error) {
    // Handle different error types
    if (error.name === 'TokenExpiredError') {
      return next(createError('Your session has expired. Please log in again.', 401));
    } else if (error.name === 'JsonWebTokenError') {
      return next(createError('Invalid authentication token. Please log in again.', 401));
    }
    
    next(createError('Authentication failed. Please log in again.', 401));
  }
};

/**
 * Role-based authorization middleware
 * @param {...string} roles - Allowed roles
 * @returns {Function} Middleware function
 */
const authorize = (...roles) => {
  return (req, res, next) => {
    // Check if user exists and has a role
    if (!req.user || !req.user.role) {
      return next(createError('User role not found. Access denied.', 403));
    }
    
    // Check if user's role is allowed
    if (!roles.includes(req.user.role)) {
      return next(createError('You do not have permission to access this resource.', 403));
    }
    
    next();
  };
};

module.exports = {
  auth,
  authorize
}; 