/**
 * Role Controller
 * Handles role management and permission operations
 */

import { Response } from 'express';
import prisma from '../utils/prismaClient';
import env from '../utils/env';
import authorizationService from '../services/authorizationService';
import { AuthenticatedRequest, Permission } from '../types/auth';
import { createLogger } from '../utils/logger';
import { createAuditLog } from '../utils/auditLog';

// Create logger
const logger = createLogger('RoleController');

/**
 * Get all roles with permissions
 * @route GET /api/roles
 * @access Private/Admin
 */
export const getAllRoles = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }

    const roles = authorizationService.getAllRoles();
    
    res.status(200).json({
      success: true,
      data: roles
    });
  } catch (error) {
    console.error('Get roles error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching roles',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Get all permissions
 * @route GET /api/roles/permissions
 * @access Private/Admin
 */
export const getAllPermissions = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }

    const permissions = authorizationService.getAllPermissions();
    
    res.status(200).json({
      success: true,
      data: permissions
    });
  } catch (error) {
    console.error('Get permissions error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching permissions',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Get role details
 * @route GET /api/roles/:roleName
 * @access Private/Admin
 */
export const getRole = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }
    
    const { roleName } = req.params;
    const role = authorizationService.getRole(roleName);
    
    if (!role) {
      res.status(404).json({
        success: false,
        message: `Role '${roleName}' not found`
      });
      return;
    }
    
    res.status(200).json({
      success: true,
      data: role
    });
  } catch (error) {
    console.error('Get role error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching role',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Get users by role
 * @route GET /api/roles/:roleName/users
 * @access Private/Admin
 */
export const getUsersByRole = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }
    
    const { roleName } = req.params;
    
    // Check if role exists
    if (!authorizationService.roleExists(roleName)) {
      res.status(404).json({
        success: false,
        message: `Role '${roleName}' not found`
      });
      return;
    }
    
    // Get users with pagination
    const page = Number(req.query.page) || 1;
    const limit = Number(req.query.limit) || 10;
    const skip = (page - 1) * limit;
    
    const users = await prisma.user.findMany({
      where: { role: roleName },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        isVerified: true,
        createdAt: true,
        updatedAt: true
      },
      skip,
      take: limit,
      orderBy: {
        createdAt: 'desc'
      }
    });
    
    const total = await prisma.user.count({
      where: { role: roleName }
    });
    
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
    console.error('Get users by role error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching users by role',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

/**
 * Check if current user has permission
 * @route GET /api/roles/check-permission/:permission
 * @access Private
 */
export const checkPermission = async (req: AuthenticatedRequest, res: Response): Promise<void> => {
  try {
    if (!req.user) {
      res.status(401).json({
        success: false,
        message: 'Not authenticated'
      });
      return;
    }
    
    const { permission } = req.params;
    const hasPermission = authorizationService.hasPermission(
      req.user.role, 
      permission as Permission
    );
    
    res.status(200).json({
      success: true,
      data: {
        hasPermission
      }
    });
  } catch (error) {
    console.error('Check permission error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while checking permission',
      error: env.NODE_ENV === 'development' ? (error as Error).message : undefined
    });
  }
};

export default {
  getAllRoles,
  getAllPermissions,
  getRole,
  getUsersByRole,
  checkPermission
}; 