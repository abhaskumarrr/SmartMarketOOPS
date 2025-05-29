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
  authRateLimiter, 
  csrfProtection 
} from '../middleware/auth';

const router = express.Router();

// CSRF token endpoint - provides token for frontend forms
router.get('/csrf-token', csrfProtection, (req, res) => {
  res.json({ csrfToken: req.csrfToken() });
});

// Register new user - apply rate limiting to prevent spam
router.post('/register', authRateLimiter, csrfProtection, register);

// Login user - apply rate limiting to prevent brute force attacks
router.post('/login', authRateLimiter, csrfProtection, login);

// Refresh token - no CSRF/auth required as it uses the refresh token itself
router.post('/refresh-token', refreshToken);

// Verify email
router.get('/verify-email/:token', verifyEmail);

// Forgot password - request reset - apply rate limiting
router.post('/forgot-password', authRateLimiter, csrfProtection, forgotPassword);

// Reset password with token - apply rate limiting
router.post('/reset-password', authRateLimiter, csrfProtection, resetPassword);

// OAuth login/signup
router.post('/oauth', csrfProtection, oauthLogin);

// Logout - requires authentication
router.post('/logout', protect, csrfProtection, logout);

export default router; 