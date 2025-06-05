/**
 * Authentication Routes
 * Handles user registration, login, token management, email verification, and password reset
 */

import express from 'express';
import {
  register,
  login,
  refreshToken,
  verifyEmail,
  forgotPassword,
  resetPassword,
  oauthLogin,
  logout
} from '../controllers/authController';
import {
  protect,
  verifyRefreshToken,
  authRateLimiter
} from '../middleware/auth';

const router = express.Router();

// Health check for auth routes
router.get('/health', (req, res) => {
  res.json({ status: 'Auth routes working', timestamp: new Date().toISOString() });
});

// CSRF token endpoint - provides token for frontend forms
router.get('/csrf-token', (req, res) => {
  res.json({ csrfToken: 'dev-token' });
});

// Register new user - apply rate limiting to prevent spam
router.post('/register', authRateLimiter, register);

// Login user - apply rate limiting to prevent brute force attacks
router.post('/login', authRateLimiter, login);

// Refresh token - no CSRF/auth required as it uses the refresh token itself
router.post('/refresh-token', refreshToken);

// Verify email
router.get('/verify-email/:token', verifyEmail);

// Forgot password - request reset - apply rate limiting
router.post('/forgot-password', authRateLimiter, forgotPassword);

// Reset password with token - apply rate limiting
router.post('/reset-password', authRateLimiter, resetPassword);

// OAuth login/signup
router.post('/oauth', oauthLogin);

// Logout - requires authentication
router.post('/logout', protect, logout);

export default router;