/**
 * User Routes
 * Endpoints for user profile management
 */

import express from 'express';
import { getProfile, updateProfile, getUsers, updateUserRole } from '../controllers/userController';
import { protect, requirePermission } from '../middleware/auth';

const router = express.Router();

// Get current user profile
router.get('/profile', protect, requirePermission(['profile:read']), getProfile);

// Update current user profile
router.put('/profile', protect, requirePermission(['profile:update']), updateProfile);

// Admin only: Get all users
router.get('/', protect, requirePermission(['users:read']), getUsers);

// Admin only: Update a user's role
router.put('/:id/role', protect, requirePermission(['admin:manageRoles']), updateUserRole);

export default router; 