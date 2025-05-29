/**
 * Authentication Controller
 * Handles user authentication operations
 */

const bcrypt = require('bcrypt');
const prisma = require('../utils/prismaClient');
const { verifyRefreshToken } = require('../utils/jwt');
const { createSession, invalidateSession } = require('../utils/sessionManager');
const { createError } = require('../middleware/errorHandler');
const { COOKIE_OPTIONS } = require('../middleware/sessionMiddleware');

/**
 * Register a new user
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const register = async (req, res, next) => {
  try {
    const { email, password, name } = req.body;
    
    // Validate required fields
    if (!email || !password) {
      return next(createError('Email and password are required', 400));
    }
    
    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email }
    });
    
    if (existingUser) {
      return next(createError('User with this email already exists', 400));
    }
    
    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    
    // Create user
    const user = await prisma.user.create({
      data: {
        email,
        password: hashedPassword,
        name: name || email.split('@')[0], // Use part of email as name if not provided
        role: 'user' // Default role
      }
    });
    
    // Create session
    const { session, accessToken, refreshToken } = await createSession(user, req, true);
    
    // Send response
    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      data: {
        user: {
          id: user.id,
          email: user.email,
          name: user.name,
          role: user.role
        },
        accessToken,
        refreshToken
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Login a user
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const login = async (req, res, next) => {
  try {
    const { email, password, rememberMe = false } = req.body;
    
    // Validate required fields
    if (!email || !password) {
      return next(createError('Email and password are required', 400));
    }
    
    // Find user
    const user = await prisma.user.findUnique({
      where: { email }
    });
    
    if (!user) {
      return next(createError('Invalid credentials', 401));
    }
    
    // Check password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    
    if (!isPasswordValid) {
      return next(createError('Invalid credentials', 401));
    }
    
    // Create session
    const { session, accessToken, refreshToken } = await createSession(user, req, rememberMe);
    
    // Send response
    res.json({
      success: true,
      message: 'Login successful',
      data: {
        user: {
          id: user.id,
          email: user.email,
          name: user.name,
          role: user.role
        },
        accessToken,
        refreshToken
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Refresh access token
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const refreshToken = async (req, res, next) => {
  try {
    const { refreshToken: token } = req.body;
    
    if (!token) {
      return next(createError('Refresh token is required', 400));
    }
    
    try {
      // Verify refresh token
      const decoded = verifyRefreshToken(token);
      
      // Get user
      const user = await prisma.user.findUnique({
        where: { id: decoded.sub || decoded.userId }
      });
      
      if (!user) {
        return next(createError('User not found', 404));
      }
      
      // Check if session is valid
      const session = await prisma.session.findUnique({
        where: { id: decoded.sessionId }
      });
      
      if (!session || !session.isValid || new Date(session.expiresAt) < new Date()) {
        return next(createError('Session expired', 401));
      }
      
      // Generate new tokens
      const { accessToken, refreshToken } = await require('../utils/sessionManager').refreshSession(
        decoded.sessionId, 
        user, 
        decoded.rememberMe
      );
      
      // Send response
      res.json({
        success: true,
        data: {
          accessToken,
          refreshToken
        }
      });
    } catch (error) {
      if (error.name === 'TokenExpiredError') {
        return next(createError('Refresh token expired', 401));
      }
      
      if (error.name === 'JsonWebTokenError') {
        return next(createError('Invalid refresh token', 401));
      }
      
      throw error;
    }
  } catch (error) {
    next(error);
  }
};

/**
 * Logout a user
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const logout = async (req, res, next) => {
  try {
    const sessionId = req.sessionId;
    
    if (sessionId) {
      await invalidateSession(sessionId);
    }
    
    // Clear cookies (if using cookies)
    res.clearCookie('refreshToken', COOKIE_OPTIONS);
    
    res.json({
      success: true,
      message: 'Logged out successfully'
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Get current user profile
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getCurrentUser = async (req, res, next) => {
  try {
    // User is already attached to request by auth middleware
    const user = req.user;
    
    res.json({
      success: true,
      data: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
        createdAt: user.createdAt,
        updatedAt: user.updatedAt
      }
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  register,
  login,
  refreshToken,
  logout,
  getCurrentUser
}; 