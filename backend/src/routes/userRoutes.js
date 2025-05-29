/**
 * User Routes
 * Endpoints for user profile management
 */

const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');
const { auth } = require('../middleware/auth');

// Get current user profile
router.get('/profile', auth, userController.getProfile);

// Update current user profile
router.put('/profile', auth, userController.updateProfile);

// Admin only: Get all users
router.get('/', auth, userController.getUsers);

module.exports = router; 