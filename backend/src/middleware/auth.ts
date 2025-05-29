/**
 * Authentication Middleware
 * Validates JWT tokens and protects routes
 */

import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import prisma from '../utils/prismaClient';
import { verifyToken, verifyRefreshToken as verifyRefreshTokenUtil } from '../utils/jwt';
import env from '../utils/env';
import rateLimit from 'express-rate-limit';
import csrf from 'csurf';
import { Permission, AuthenticatedRequest } from '../types/auth';
import authorizationService from '../services/authorizationService';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';
import { createLogger, LogData } from '../utils/logger';

// Create logger
const logger = createLogger('AuthMiddleware');

/**
 * Rate limiter for authentication routes
 * Limits login, register, and password reset attempts
 */
export const authRateLimiter = rateLimit({
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
export const csrfProtection = csrf({ 
  cookie: { 
    httpOnly: true, 
    secure: env.NODE_ENV === 'production',
    sameSite: 'strict'
  } 
});

/**
 * Middleware to protect routes
 * Verifies JWT token and attaches user to request
 */
export const protect = async (req: AuthenticatedRequest, res: Response, next: NextFunction): Promise<void> => {
  try {
    // Get token from header
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({
        success: false,
        message: 'Not authorized, no token'
      });
      return;
    }
    
    // Verify token
    const token = authHeader.split(' ')[1];
    const decoded = verifyToken(token);
    
    if (!decoded) {
      res.status(401).json({
        success: false,
        message: 'Not authorized, token failed'
      });
      return;
    }
    
    // Get user from database
    const user = await prisma.user.findUnique({
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
  } catch (error) {
    console.error('Auth middleware error:', error);
    res.status(401).json({
      success: false,
      message: 'Not authorized, token failed',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Middleware to verify refresh token
 * Used for token refresh routes
 */
export const verifyRefreshToken = async (req: AuthenticatedRequest, res: Response, next: NextFunction): Promise<void> => {
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
    const decoded = verifyRefreshTokenUtil(refreshToken);
    
    if (!decoded) {
      res.status(401).json({
        success: false,
        message: 'Invalid refresh token'
      });
      return;
    }
    
    // Get user from database
    const user = await prisma.user.findUnique({
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
  } catch (error) {
    console.error('Refresh token middleware error:', error);
    res.status(401).json({
      success: false,
      message: 'Invalid refresh token',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Middleware to require email verification
 * Used for routes that require verified email
 */
export const requireVerified = (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
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

/**
 * Middleware to require specific role(s)
 * Used for routes that require specific user roles
 * @param roles - Array of allowed roles
 */
export const requireRole = (roles: string[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
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

/**
 * Middleware to require specific permission(s)
 * Used for fine-grained access control based on permissions
 * @param permissions - Array of required permissions (all must be present)
 */
export const requirePermission = (permissions: Permission[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }
    
    if (!authorizationService.hasAllPermissions(req.user.role, permissions)) {
      res.status(403).json({
        success: false,
        message: `Insufficient permissions. Required: ${permissions.join(', ')}`
      });
      return;
    }
    
    next();
  };
};

/**
 * Middleware to require any of the specified permissions
 * At least one of the permissions must be present
 * @param permissions - Array of permissions (at least one must be present)
 */
export const requireAnyPermission = (permissions: Permission[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }
    
    if (!authorizationService.hasAnyPermission(req.user.role, permissions)) {
      res.status(403).json({
        success: false,
        message: `Insufficient permissions. Required any of: ${permissions.join(', ')}`
      });
      return;
    }
    
    next();
  };
};

/**
 * Middleware to verify user is authenticated
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 * @param {NextFunction} next - Express next function
 */
export const isAuthenticated = async (req: Request, res: Response, next: NextFunction): Promise<void> => {
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
      const decoded: any = jwt.verify(token, secret);
      
      // Find the session to validate it's still active
      const session = await prisma.session.findFirst({
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
      await prisma.session.update({
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
    } catch (error) {
      res.status(401).json({ error: 'Invalid authentication token' });
    }
  } catch (error) {
    console.error('Authentication error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

/**
 * Middleware to verify user has required role
 * @param {string[]} roles - Array of allowed roles
 */
export const hasRole = (roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction): void => {
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

export default {
  protect,
  verifyRefreshToken,
  requireVerified,
  requireRole,
  requirePermission,
  requireAnyPermission,
  authRateLimiter,
  csrfProtection,
  isAuthenticated,
  hasRole
}; 