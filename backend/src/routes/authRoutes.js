/**
 * Authentication Routes
 * Handles user registration, login, and token management
 */

const express = require('express');
const router = express.Router();
const authController = require('../controllers/authController');
const { auth } = require('../middleware/auth');

// Register new user
router.post('/register', authController.register);

// Login user
router.post('/login', authController.login);

// Refresh token
router.post('/refresh', authController.refreshToken);

// Logout user
router.post('/logout', auth, authController.logout);

// Get current user profile
router.get('/me', auth, authController.getCurrentUser);

module.exports = router; 