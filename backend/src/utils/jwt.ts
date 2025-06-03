/**
 * JWT Utilities
 * Functions for generating and validating JWT tokens
 */

import jwt from 'jsonwebtoken';
import env from './env';

interface JwtPayload {
  id: string;
  [key: string]: any;
}

/**
 * Generate JWT access token (15 minutes for enhanced security)
 * @param {string} userId - User ID
 * @param {object} additionalPayload - Additional payload data
 * @returns {string} JWT token
 */
export const generateToken = (userId: string, additionalPayload: object = {}): string => {
  return jwt.sign(
    {
      id: userId,
      type: 'access',
      iat: Math.floor(Date.now() / 1000),
      ...additionalPayload
    },
    env.JWT_SECRET,
    { expiresIn: '15m' } // 15 minutes for enhanced security
  );
};

/**
 * Generate JWT refresh token (7 days)
 * @param {string} userId - User ID
 * @param {string} sessionId - Session ID for token rotation
 * @returns {string} JWT refresh token
 */
export const generateRefreshToken = (userId: string, sessionId?: string): string => {
  // Use JWT_REFRESH_SECRET if available, otherwise use JWT_SECRET
  const secret = process.env.JWT_REFRESH_SECRET || env.JWT_SECRET;

  return jwt.sign(
    {
      id: userId,
      type: 'refresh',
      sessionId,
      iat: Math.floor(Date.now() / 1000)
    },
    secret,
    { expiresIn: '7d' } // 7 days for refresh token
  );
};

/**
 * Verify JWT token
 * @param {string} token - JWT token to verify
 * @returns {JwtPayload|null} Decoded token payload or null if invalid
 */
export const verifyToken = (token: string): JwtPayload | null => {
  try {
    return jwt.verify(token, env.JWT_SECRET) as JwtPayload;
  } catch (error) {
    console.error('Token verification failed:', error instanceof Error ? error.message : error);
    return null;
  }
};

/**
 * Verify JWT refresh token
 * @param {string} token - JWT refresh token to verify
 * @returns {JwtPayload|null} Decoded token payload or null if invalid
 */
export const verifyRefreshToken = (token: string): JwtPayload | null => {
  try {
    // Use JWT_REFRESH_SECRET if available, otherwise use JWT_SECRET
    const secret = process.env.JWT_REFRESH_SECRET || env.JWT_SECRET;
    return jwt.verify(token, secret) as JwtPayload;
  } catch (error) {
    console.error('Refresh token verification failed:', error instanceof Error ? error.message : error);
    return null;
  }
};

/**
 * Generate token pair (access + refresh)
 * @param {string} userId - User ID
 * @param {string} sessionId - Session ID
 * @param {object} additionalPayload - Additional payload for access token
 * @returns {object} Token pair
 */
export const generateTokenPair = (userId: string, sessionId?: string, additionalPayload: object = {}) => {
  return {
    accessToken: generateToken(userId, additionalPayload),
    refreshToken: generateRefreshToken(userId, sessionId),
    expiresIn: 15 * 60, // 15 minutes in seconds
    tokenType: 'Bearer'
  };
};

/**
 * Extract token from Authorization header
 * @param {string} authHeader - Authorization header value
 * @returns {string|null} Token or null if invalid
 */
export const extractTokenFromHeader = (authHeader: string): string | null => {
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return null;
  }
  return authHeader.substring(7); // Remove 'Bearer ' prefix
};

/**
 * Check if token is expired
 * @param {JwtPayload} payload - Decoded JWT payload
 * @returns {boolean} True if token is expired
 */
export const isTokenExpired = (payload: JwtPayload): boolean => {
  if (!payload.exp) return true;
  return Date.now() >= payload.exp * 1000;
};

/**
 * Get token expiration time
 * @param {JwtPayload} payload - Decoded JWT payload
 * @returns {Date|null} Expiration date or null
 */
export const getTokenExpiration = (payload: JwtPayload): Date | null => {
  if (!payload.exp) return null;
  return new Date(payload.exp * 1000);
};

export default {
  generateToken,
  generateRefreshToken,
  generateTokenPair,
  verifyToken,
  verifyRefreshToken,
  extractTokenFromHeader,
  isTokenExpired,
  getTokenExpiration
};