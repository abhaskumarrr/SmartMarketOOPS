/**
 * Session Middleware
 * Handles session validation, activity tracking, and timeout management
 */

import { Request, Response, NextFunction } from 'express';
import { 
  validateSession, 
  updateSessionActivity, 
  extendSession,
  generateDeviceFingerprint 
} from '../utils/sessionManager';
import { AppError } from './errorHandler';
import { AuthenticatedRequest } from '../types/auth';
import prisma from '../utils/prismaClient';
import cookieParser from 'cookie-parser';
import env from '../utils/env';

// Secure cookie configuration
export const cookieOptions = {
  httpOnly: true,                      // Cannot be accessed by JavaScript
  secure: env.NODE_ENV === 'production', // HTTPS only in production
  sameSite: 'strict' as const,         // Strict same-site policy
  maxAge: 24 * 60 * 60 * 1000,         // 24 hours
  path: '/',                           // Available across the site
  domain: env.NODE_ENV === 'production' ? env.COOKIE_DOMAIN : undefined,
};

/**
 * Cookie parser middleware with signing
 */
export const secureCookieParser = cookieParser(env.COOKIE_SECRET || 'SmartMarketOOPS-secret-key');

/**
 * Middleware to track user activity and update session
 */
export const sessionActivity = async (req: AuthenticatedRequest, res: Response, next: NextFunction): Promise<void> => {
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
    await updateSessionActivity(token);
    
    // Optionally extend session expiry time
    if (req.path !== '/api/auth/refresh-token') { // Don't extend during refresh
      await extendSession(token);
    }

    next();
  } catch (error) {
    // Don't fail the request if session tracking fails
    console.error('Session activity tracking error:', error);
    next();
  }
};

/**
 * Session validation middleware
 * More comprehensive than the basic JWT check
 */
export const validateUserSession = async (req: AuthenticatedRequest, res: Response, next: NextFunction): Promise<void> => {
  try {
    // Get token from header
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return next(new AppError('Not authorized, no token', 401));
    }
    
    // Extract token
    const token = authHeader.split(' ')[1];
    
    // Validate session
    const session = await validateSession(token);
    
    if (!session) {
      return next(new AppError('Session expired or invalid', 401));
    }
    
    // Get user from database
    const user = await prisma.user.findUnique({
      where: { id: session.userId }
    });
    
    if (!user) {
      return next(new AppError('User not found', 401));
    }
    
    // Device fingerprint verification (optional security enhancement)
    const currentFingerprint = generateDeviceFingerprint(req);
    const storedFingerprint = session.deviceId;
    
    // If we have a stored fingerprint and it doesn't match (potential session hijacking)
    if (storedFingerprint && currentFingerprint !== storedFingerprint && env.NODE_ENV === 'production') {
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
  } catch (error) {
    next(new AppError('Session validation failed', 401));
  }
};

/**
 * Middleware to set session tracking cookie
 */
export const setDeviceIdCookie = (req: Request, res: Response, next: NextFunction): void => {
  // Check if device ID cookie exists
  if (!req.cookies.deviceId) {
    // Generate device ID
    const deviceId = generateDeviceFingerprint(req);
    
    // Set cookie
    res.cookie('deviceId', deviceId, {
      ...cookieOptions,
      maxAge: 365 * 24 * 60 * 60 * 1000, // 1 year
    });
  }
  
  next();
};

/**
 * Remember Me cookie handling
 */
export const setRememberMeCookie = (req: Request, res: Response, rememberMe: boolean): void => {
  if (rememberMe) {
    res.cookie('rememberMe', 'true', {
      ...cookieOptions,
      maxAge: 30 * 24 * 60 * 60 * 1000, // 30 days
    });
  } else {
    res.clearCookie('rememberMe');
  }
};

export default {
  secureCookieParser,
  sessionActivity,
  validateUserSession,
  setDeviceIdCookie,
  setRememberMeCookie,
  cookieOptions
}; 