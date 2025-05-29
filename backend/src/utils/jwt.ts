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
 * Generate JWT access token
 * @param {string} userId - User ID
 * @returns {string} JWT token
 */
export const generateToken = (userId: string): string => {
  return jwt.sign(
    { id: userId },
    env.JWT_SECRET,
    { expiresIn: '1h' }
  );
};

/**
 * Generate JWT refresh token
 * @param {string} userId - User ID
 * @returns {string} JWT refresh token
 */
export const generateRefreshToken = (userId: string): string => {
  // Use JWT_REFRESH_SECRET if available, otherwise use JWT_SECRET
  const secret = process.env.JWT_REFRESH_SECRET || env.JWT_SECRET;
  
  return jwt.sign(
    { id: userId },
    secret,
    { expiresIn: '7d' }
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

export default {
  generateToken,
  generateRefreshToken,
  verifyToken,
  verifyRefreshToken
}; 