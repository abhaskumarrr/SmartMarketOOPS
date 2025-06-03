/**
 * Session Manager
 * Handles session creation, validation, and management
 */

import jwt from 'jsonwebtoken';
import prisma from './prismaClient';
import { v4 as uuidv4 } from 'uuid';
import { Request } from 'express';
import crypto from 'crypto';

// Environment variables
const {
  JWT_SECRET = 'your-secret-key',
  JWT_EXPIRY = '1h',
  SESSION_EXPIRY = '24h'
} = process.env;

// Session interface matching our Prisma model
interface Session {
  id: string;
  userId: string;
  token: string;
  refreshToken?: string | null;
  ipAddress?: string | null;
  userAgent?: string | null;
  deviceId?: string | null;
  isValid: boolean;
  expiresAt: Date;
  lastActiveAt: Date;
  createdAt: Date;
  updatedAt: Date;
  rememberMe: boolean;
  metadata?: any | null;
}

interface SessionConfig {
  maxAge: number; // Session max age in milliseconds
  rememberMeMaxAge: number; // Extended session duration with "remember me"
  inactivityTimeout: number; // Session timeout after inactivity
}

// Session configuration with defaults
const sessionConfig: SessionConfig = {
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
export const createSession = async (
  userId: string,
  req: Request,
  rememberMe: boolean = false
): Promise<{
  token: string;
  refreshToken: string;
  session: Session;
}> => {
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
  let deviceId = req.cookies.deviceId || req.headers['x-device-id'] as string;
  if (!deviceId) {
    deviceId = uuidv4(); // Generate a new device ID if none exists
  }

  // Create session record
  const session = await prisma.session.create({
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

/**
 * Validate a session
 * @param token - JWT token
 * @returns Valid session or null
 */
export const validateSession = async (token: string): Promise<Session | null> => {
  // First verify the JWT token
  const decoded = verifyToken(token);
  if (!decoded) {
    return null;
  }

  // Check if session exists and is valid
  const session = await prisma.session.findFirst({
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
    await invalidateSession(token);
    return null;
  }

  // Update last active timestamp
  await updateSessionActivity(token);

  return session;
};

/**
 * Update session's last activity timestamp
 * @param token - JWT token
 */
export const updateSessionActivity = async (token: string): Promise<void> => {
  await prisma.session.updateMany({
    where: { token },
    data: { lastActiveAt: new Date() },
  });
};

/**
 * Invalidate a single session
 * @param token - JWT token
 * @returns Success indicator
 */
export const invalidateSession = async (token: string): Promise<boolean> => {
  try {
    await prisma.session.updateMany({
      where: { token },
      data: { isValid: false },
    });
    return true;
  } catch (error) {
    console.error('Error invalidating session:', error);
    return false;
  }
};

/**
 * Revoke all sessions for a user except current one
 * @param userId - User ID
 * @param currentToken - Current session token to preserve
 * @returns Number of sessions invalidated
 */
export const revokeAllSessions = async (
  userId: string,
  currentToken?: string
): Promise<number> => {
  const result = await prisma.session.updateMany({
    where: {
      userId,
      ...(currentToken ? { token: { not: currentToken } } : {}),
      isValid: true,
    },
    data: { isValid: false },
  });

  return result.count;
};

/**
 * Get all active sessions for a user
 * @param userId - User ID
 * @returns List of active sessions
 */
export const getUserSessions = async (userId: string): Promise<Session[]> => {
  return prisma.session.findMany({
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

/**
 * Extend session duration (e.g., after user activity)
 * @param token - JWT token
 * @param extendBy - Milliseconds to extend by (defaults to session max age)
 */
export const extendSession = async (
  token: string,
  extendBy?: number
): Promise<void> => {
  const session = await prisma.session.findUnique({
    where: { token },
  });

  if (!session) return;

  const extension = extendBy || (session.rememberMe 
    ? sessionConfig.rememberMeMaxAge 
    : sessionConfig.maxAge);
  
  const newExpiryDate = new Date(Date.now() + extension);

  await prisma.session.update({
    where: { token },
    data: { 
      expiresAt: newExpiryDate,
      lastActiveAt: new Date()
    },
  });
};

/**
 * Clean up expired sessions (can be run as a scheduled job)
 */
export const cleanupExpiredSessions = async (): Promise<number> => {
  const result = await prisma.session.deleteMany({
    where: {
      OR: [
        { expiresAt: { lt: new Date() } },
        { isValid: false },
      ],
    },
  });

  return result.count;
};

/**
 * Generate fingerprint for device identification
 * @param req - Express request
 * @returns Device fingerprint string
 */
export const generateDeviceFingerprint = (req: Request): string => {
  const components = [
    req.headers['user-agent'] || 'unknown',
    req.ip || req.socket.remoteAddress || 'unknown',
    req.headers['accept-language'] || 'unknown',
  ];
  
  // Simple fingerprint generation - in production, use a more sophisticated approach
  return Buffer.from(components.join('|')).toString('base64');
};

export default {
  createSession,
  validateSession,
  updateSessionActivity,
  extendSession,
  invalidateSession,
  getUserSessions,
  cleanupExpiredSessions,
  generateDeviceFingerprint,
  getSessionMetadata,
  updateSessionMetadata,
  sessionConfig
};