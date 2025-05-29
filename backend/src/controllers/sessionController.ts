/**
 * Session Controller
 * Handles session management, device tracking, and session security
 */

import { Response } from 'express';
import { AuthenticatedRequest } from '../types/auth';
import sessionManager from '../utils/sessionManager';
import { AppError, asyncHandler } from '../middleware/errorHandler';
import prisma from '../utils/prismaClient';
import { createLogger } from '../utils/logger';
import { createAuditLog } from '../utils/auditLog';

// Create logger
const logger = createLogger('SessionController');

/**
 * Get all active sessions for the current user
 * @route GET /api/sessions
 * @access Private
 */
export const getUserSessions = asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  if (!req.user) {
    throw new AppError('Not authenticated', 401);
  }

  const sessions = await sessionManager.getUserSessions(req.user.id);

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
export const checkSession = asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  if (!req.user || !req.user.sessionId) {
    throw new AppError('Not authenticated', 401);
  }

  // Get session info
  const session = await prisma.session.findUnique({
    where: { id: req.user.sessionId }
  });

  if (!session) {
    throw new AppError('Session not found', 404);
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
export const revokeSession = asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  if (!req.user) {
    throw new AppError('Not authenticated', 401);
  }

  const { sessionId } = req.params;

  // Prevent revoking current session through this endpoint
  if (req.user.sessionId === sessionId) {
    throw new AppError('Cannot revoke current session. Use logout instead.', 400);
  }

  // Check if session belongs to the user
  const session = await prisma.session.findFirst({
    where: {
      id: sessionId,
      userId: req.user.id
    }
  });

  if (!session) {
    throw new AppError('Session not found or does not belong to user', 404);
  }

  // Invalidate the session
  await prisma.session.update({
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
export const revokeAllSessions = asyncHandler(async (req: AuthenticatedRequest, res: Response) => {
  if (!req.user || !req.user.sessionId) {
    throw new AppError('Not authenticated', 401);
  }

  const currentSessionId = req.user.sessionId;

  // Invalidate all other sessions
  const result = await prisma.session.updateMany({
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
function parseUserAgent(userAgent: string): string {
  // Simple parsing - in production, use a more sophisticated user agent parser
  if (!userAgent || userAgent === 'unknown') return 'Unknown device';

  let device = 'Unknown device';

  // Mobile detection
  if (userAgent.includes('iPhone') || userAgent.includes('iPad')) {
    device = userAgent.includes('iPad') ? 'iPad' : 'iPhone';
  } else if (userAgent.includes('Android')) {
    device = 'Android device';
  } else if (userAgent.includes('Windows Phone')) {
    device = 'Windows Phone';
  }
  // Desktop detection
  else if (userAgent.includes('Windows')) {
    device = 'Windows computer';
  } else if (userAgent.includes('Macintosh') || userAgent.includes('Mac OS')) {
    device = 'Mac computer';
  } else if (userAgent.includes('Linux')) {
    device = 'Linux computer';
  }

  // Browser detection
  let browser = 'Unknown browser';
  if (userAgent.includes('Chrome') && !userAgent.includes('Chromium')) {
    browser = 'Chrome';
  } else if (userAgent.includes('Firefox')) {
    browser = 'Firefox';
  } else if (userAgent.includes('Safari') && !userAgent.includes('Chrome')) {
    browser = 'Safari';
  } else if (userAgent.includes('Edge')) {
    browser = 'Edge';
  } else if (userAgent.includes('MSIE') || userAgent.includes('Trident/')) {
    browser = 'Internet Explorer';
  }

  return `${device} (${browser})`;
}

export default {
  getUserSessions,
  checkSession,
  revokeSession,
  revokeAllSessions
}; 