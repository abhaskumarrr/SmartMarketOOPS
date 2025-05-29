/**
 * Role Management Routes
 * Provides endpoints for managing user roles
 */

const express = require('express');
const router = express.Router();
const { auth } = require('../middleware/auth');
const roleController = require('../controllers/roleController');

// Get all roles (admin only)
router.get('/', auth, roleController.getAllRoles);

// Get permissions for current user's role
router.get('/permissions', auth, roleController.getUserPermissions);

// Assign role to user (admin only)
router.post('/assign', auth, roleController.assignRole);

module.exports = router; 