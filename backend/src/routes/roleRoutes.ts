/**
 * Role Routes
 * Endpoints for role management and permission operations
 */

import express from 'express';
import {
  getAllRoles,
  getAllPermissions,
  getRole,
  getUsersByRole,
  checkPermission
} from '../controllers/roleController';
import { protect, requirePermission } from '../middleware/auth';

const router = express.Router();

// All routes require authentication
router.use(protect);

// Get all roles with permissions
router.get('/', requirePermission(['admin:manageRoles']), getAllRoles);

// Get all available permissions
router.get('/permissions', requirePermission(['admin:manageRoles']), getAllPermissions);

// Check if current user has a specific permission
router.get('/check-permission/:permission', checkPermission);

// Get specific role
router.get('/:roleName', requirePermission(['admin:manageRoles']), getRole);

// Get users by role
router.get('/:roleName/users', requirePermission(['admin:manageRoles', 'users:read']), getUsersByRole);

export default router; 