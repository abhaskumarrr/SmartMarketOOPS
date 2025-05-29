/**
 * JWT Authentication Utilities
 * Provides functions for generating and verifying JWT tokens
 */

const jwt = require('jsonwebtoken');
const { env } = require('./env');

/**
 * Generate a JWT access token
 * @param {object} user - User object
 * @param {string} sessionId - Session ID
 * @param {object} options - JWT sign options
 * @returns {string} JWT token
 */
const generateAccessToken = (user, sessionId, options = {}) => {
  const payload = {
    sub: user.id,
    email: user.email,
    role: user.role,
    sessionId
  };
  
  const defaultOptions = {
    expiresIn: env.JWT_EXPIRES_IN
  };
  
  const mergedOptions = { ...defaultOptions, ...options };
  
  return jwt.sign(payload, env.JWT_SECRET, mergedOptions);
};

/**
 * Generate a JWT refresh token
 * @param {object} user - User object
 * @param {string} sessionId - Session ID
 * @param {object} options - JWT sign options
 * @returns {string} JWT refresh token
 */
const generateRefreshToken = (user, sessionId, options = {}) => {
  const payload = {
    sub: user.id,
    sessionId,
    type: 'refresh'
  };
  
  const defaultOptions = {
    expiresIn: '7d'
  };
  
  const mergedOptions = { ...defaultOptions, ...options };
  
  return jwt.sign(payload, env.JWT_REFRESH_SECRET, mergedOptions);
};

/**
 * Verify a JWT access token
 * @param {string} token - JWT token to verify
 * @returns {object} Decoded token payload
 */
const verifyAccessToken = (token) => {
  return jwt.verify(token, env.JWT_SECRET);
};

/**
 * Verify a JWT refresh token
 * @param {string} token - JWT refresh token to verify
 * @returns {object} Decoded token payload
 */
const verifyRefreshToken = (token) => {
  return jwt.verify(token, env.JWT_REFRESH_SECRET);
};

module.exports = {
  generateAccessToken,
  generateRefreshToken,
  verifyAccessToken,
  verifyRefreshToken
}; 