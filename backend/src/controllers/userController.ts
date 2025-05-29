/**
 * User Controller
 * Handles user profile management and user operations
 */

import { Response } from 'express';
import prisma from '../utils/prismaClient';
import bcrypt from 'bcryptjs';
import env from '../utils/env';
import { AuthenticatedRequest } from '../types/auth';
import authorizationService from '../services/authorizationService';
import { v4 as uuidv4 } from 'uuid';
import { createLogger } from '../utils/logger';
import { createAuditLog } from '../utils/auditLog';

// Create logger
const logger = createLogger('UserController');

interface UpdateProfileRequest {
  name?: string;
  email?: string;
  password?: string;
}

/**
 * Get current user profile
 * @route GET /api/users/profile
 * @access Private
 */
export const getProfile = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }

    const user = await prisma.user.findUnique({
      where: { id: req.user.id },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        isVerified: true,
        createdAt: true,
        updatedAt: true
      }
    });

    if (!user) {
      res.status(404).json({
        success: false,
        message: 'User not found'
      });
      return;
    }

    res.status(200).json({
      success: true,
      data: user
    });
  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching profile',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Update current user profile
 * @route PUT /api/users/profile
 * @access Private
 */
export const updateProfile = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }

    const { name, email, password } = req.body as UpdateProfileRequest;
    const userId = req.user.id;

    // Build update data object
    const updateData: any = {};
    if (name) updateData.name = name;
    if (email) updateData.email = email;
    
    // If updating password, hash it
    if (password) {
      const salt = await bcrypt.genSalt(10);
      updateData.password = await bcrypt.hash(password, salt);
    }

    // If no updates provided
    if (Object.keys(updateData).length === 0) {
      res.status(400).json({
        success: false,
        message: 'Please provide at least one field to update'
      });
      return;
    }

    // If changing email, check if it already exists for another user
    if (email && email !== req.user.email) {
      const existingUser = await prisma.user.findUnique({
        where: { email }
      });

      if (existingUser && existingUser.id !== userId) {
        res.status(400).json({
          success: false,
          message: 'Email already in use'
        });
        return;
      }

      // If changing email, reset verification status
      updateData.isVerified = false;
    }

    // Update user
    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: updateData,
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        isVerified: true,
        createdAt: true,
        updatedAt: true
      }
    });

    res.status(200).json({
      success: true,
      data: updatedUser
    });
  } catch (error) {
    console.error('Update profile error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while updating profile',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Get all users (admin only)
 * @route GET /api/users
 * @access Private/Admin
 */
export const getUsers = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }
    
    // Check if current user is admin (handled by requireRole middleware, but double-check)
    if (req.user.role !== 'admin') {
      res.status(403).json({
        success: false,
        message: 'Not authorized, admin access required'
      });
      return;
    }

    // Get all users with pagination
    const page = Number(req.query.page) || 1;
    const limit = Number(req.query.limit) || 10;
    const skip = (page - 1) * limit;

    const users = await prisma.user.findMany({
      skip,
      take: limit,
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        isVerified: true,
        createdAt: true,
        updatedAt: true,
        lastLoginAt: true
      },
      orderBy: {
        createdAt: 'desc'
      }
    });

    // Get total count for pagination
    const total = await prisma.user.count();

    res.status(200).json({
      success: true,
      data: users,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    console.error('Get users error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching users',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Update user role (admin only)
 * @route PUT /api/users/:id/role
 * @access Private/Admin
 */
export const updateUserRole = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }
    
    // Check if current user has permission to manage roles
    if (!authorizationService.hasPermission(req.user.role, 'admin:manageRoles')) {
      res.status(403).json({
        success: false,
        message: 'Not authorized to manage user roles'
      });
      return;
    }

    const { id } = req.params;
    const { role } = req.body;

    // Validate role exists in our system
    if (!role || !authorizationService.roleExists(role)) {
      res.status(400).json({
        success: false,
        message: `Invalid role. Valid roles are: ${Object.keys(authorizationService.getAllRoles()).join(', ')}`
      });
      return;
    }

    // Don't allow changing own role (to prevent admin lockout)
    if (id === req.user.id) {
      res.status(400).json({
        success: false,
        message: 'Cannot change your own role'
      });
      return;
    }

    // Find user
    const userExists = await prisma.user.findUnique({
      where: { id }
    });

    if (!userExists) {
      res.status(404).json({
        success: false,
        message: 'User not found'
      });
      return;
    }

    // Update user role
    const updatedUser = await prisma.user.update({
      where: { id },
      data: { role },
      select: {
        id: true,
        name: true,
        email: true,
        role: true
      }
    });

    res.status(200).json({
      success: true,
      data: updatedUser
    });
  } catch (error) {
    console.error('Update user role error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while updating user role',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
}; 