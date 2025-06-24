/**
 * Session Service
 * Handles user session management, creation, validation, and refresh
 */

import { Request } from 'express';
import crypto from 'crypto';
import jwt from 'jsonwebtoken';
import prisma from '../utils/prismaClient';
import { logger } from '../utils/logger';

export interface SessionConfig {
  sessionTimeout: number;
  refreshTokenTimeout: number;
  maxConcurrentSessions: number;
  enableSessionRotation: boolean;
}

export interface Session {
  id: string;
  userId: string;
  accessToken: string;
  refreshToken: string;
  expiresAt: Date;
  isActive: boolean;
  metadata?: Record<string, any>;
}

export class SessionService {
  private sessionConfig: SessionConfig = {
    sessionTimeout: 24 * 60 * 60 * 1000, // 24 hours
    refreshTokenTimeout: 7 * 24 * 60 * 60 * 1000, // 7 days
    maxConcurrentSessions: 5,
    enableSessionRotation: true
  };

  constructor(config?: Partial<SessionConfig>) {
    if (config) {
      this.sessionConfig = { ...this.sessionConfig, ...config };
    }
  }

  /**
   * Create a new session for a user
   */
  async createSession(userId: string, req: Request, rememberMe: boolean = false): Promise<Session> {
    try {
      // Generate unique session ID
      const sessionId = crypto.randomUUID();
      
      // Generate tokens
      const accessToken = this.generateAccessToken(userId, sessionId);
      const refreshToken = this.generateRefreshToken(userId, sessionId);
      
      // Calculate expiration
      const timeout = rememberMe ? this.sessionConfig.refreshTokenTimeout : this.sessionConfig.sessionTimeout;
      const expiresAt = new Date(Date.now() + timeout);
      
      // Create session record
      const session: Session = {
        id: sessionId,
        userId,
        accessToken,
        refreshToken,
        expiresAt,
        isActive: true,
        metadata: {
          userAgent: req.headers['user-agent'],
          ip: req.ip,
          rememberMe,
          createdAt: new Date()
        }
      };

      // Store in database (if you have a sessions table)
      // await prisma.session.create({ data: session });

      logger.info(`Session created for user ${userId}`, { sessionId });
      
      return session;
    } catch (error) {
      logger.error('Failed to create session', error);
      throw new Error('Session creation failed');
    }
  }

  /**
   * Refresh an existing session
   */
  async refreshSession(sessionId: string, refreshToken: string): Promise<Session | null> {
    try {
      // Verify refresh token
      const decoded = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET || 'refresh-secret') as any;
      
      if (decoded.sessionId !== sessionId) {
        throw new Error('Invalid session ID in refresh token');
      }

      // Generate new tokens
      const newAccessToken = this.generateAccessToken(decoded.userId, sessionId);
      const newRefreshToken = this.sessionConfig.enableSessionRotation 
        ? this.generateRefreshToken(decoded.userId, sessionId)
        : refreshToken;

      // Update session
      const updatedSession: Session = {
        id: sessionId,
        userId: decoded.userId,
        accessToken: newAccessToken,
        refreshToken: newRefreshToken,
        expiresAt: new Date(Date.now() + this.sessionConfig.sessionTimeout),
        isActive: true
      };

      logger.info(`Session refreshed for user ${decoded.userId}`, { sessionId });
      
      return updatedSession;
    } catch (error) {
      logger.error('Failed to refresh session', error);
      return null;
    }
  }

  /**
   * Validate a session
   */
  async validateSession(sessionId: string, accessToken: string): Promise<boolean> {
    try {
      const decoded = jwt.verify(accessToken, process.env.JWT_SECRET || 'secret') as any;
      return decoded.sessionId === sessionId;
    } catch (error) {
      return false;
    }
  }

  /**
   * Destroy a session
   */
  async destroySession(sessionId: string): Promise<void> {
    try {
      // Remove from database
      // await prisma.session.delete({ where: { id: sessionId } });
      logger.info(`Session destroyed`, { sessionId });
    } catch (error) {
      logger.error('Failed to destroy session', error);
    }
  }

  /**
   * Cleanup expired sessions
   */
  async cleanupExpiredSessions(): Promise<void> {
    try {
      // Remove expired sessions from database
      // await prisma.session.deleteMany({
      //   where: {
      //     expiresAt: { lt: new Date() }
      //   }
      // });
      logger.info('Expired sessions cleaned up');
    } catch (error) {
      logger.error('Failed to cleanup expired sessions', error);
    }
  }

  /**
   * Get active sessions for a user
   */
  async getUserSessions(userId: string): Promise<Session[]> {
    try {
      // Fetch from database
      // const sessions = await prisma.session.findMany({
      //   where: { userId, isActive: true }
      // });
      // return sessions;
      return [];
    } catch (error) {
      logger.error('Failed to get user sessions', error);
      return [];
    }
  }

  /**
   * Generate access token
   */
  private generateAccessToken(userId: string, sessionId: string): string {
    return jwt.sign(
      { 
        userId, 
        sessionId,
        type: 'access'
      },
      process.env.JWT_SECRET || 'secret',
      { expiresIn: '1h' }
    );
  }

  /**
   * Generate refresh token
   */
  private generateRefreshToken(userId: string, sessionId: string): string {
    return jwt.sign(
      { 
        userId, 
        sessionId,
        type: 'refresh'
      },
      process.env.JWT_REFRESH_SECRET || 'refresh-secret',
      { expiresIn: '7d' }
    );
  }
}

// Utility functions for backward compatibility
export const generateToken = (userId: string, sessionId: string) => {
  return jwt.sign(
    { userId, sessionId },
    process.env.JWT_SECRET || 'secret',
    { expiresIn: '1h' }
  );
};

export const generateRefreshToken = (userId: string, sessionId: string) => {
  return jwt.sign(
    { userId, sessionId, type: 'refresh' },
    process.env.JWT_REFRESH_SECRET || 'refresh-secret',
    { expiresIn: '7d' }
  );
}; 