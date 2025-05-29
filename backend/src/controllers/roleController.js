/**
 * Role Controller
 * Handles role management operations
 */

const prisma = require('../utils/prismaClient');
const { createError } = require('../middleware/errorHandler');

// Define available roles and their permissions
const ROLES = {
  USER: 'user',
  ADMIN: 'admin',
  TRADER: 'trader',
  ANALYST: 'analyst'
};

// Define permissions for each role
const PERMISSIONS = {
  [ROLES.USER]: [
    'view_account',
    'edit_profile',
    'view_market_data'
  ],
  [ROLES.TRADER]: [
    'view_account',
    'edit_profile',
    'view_market_data',
    'create_bot',
    'execute_trade',
    'view_trading_history'
  ],
  [ROLES.ANALYST]: [
    'view_account',
    'edit_profile',
    'view_market_data',
    'view_analytics',
    'export_data',
    'view_predictions'
  ],
  [ROLES.ADMIN]: [
    'view_account',
    'edit_profile',
    'view_market_data',
    'create_bot',
    'execute_trade',
    'view_trading_history',
    'view_analytics',
    'export_data',
    'view_predictions',
    'manage_users',
    'assign_roles',
    'view_all_bots',
    'system_settings'
  ]
};

/**
 * Get all roles and their permissions
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getAllRoles = async (req, res, next) => {
  try {
    // Check if user is admin
    if (req.user.role !== ROLES.ADMIN) {
      return next(createError('Unauthorized. Admin access required.', 403));
    }
    
    // Format roles with their permissions
    const roles = Object.keys(ROLES).map(key => ({
      name: ROLES[key],
      permissions: PERMISSIONS[ROLES[key]]
    }));
    
    res.json({
      success: true,
      data: roles
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Get permissions for the current user's role
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const getUserPermissions = async (req, res, next) => {
  try {
    const userRole = req.user.role || ROLES.USER;
    
    // Get permissions for the user's role
    const permissions = PERMISSIONS[userRole] || PERMISSIONS[ROLES.USER];
    
    res.json({
      success: true,
      data: {
        role: userRole,
        permissions
      }
    });
  } catch (error) {
    next(error);
  }
};

/**
 * Assign a role to a user
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
const assignRole = async (req, res, next) => {
  try {
    // Check if user is admin
    if (req.user.role !== ROLES.ADMIN) {
      return next(createError('Unauthorized. Admin access required.', 403));
    }
    
    const { userId, role } = req.body;
    
    if (!userId) {
      return next(createError('User ID is required', 400));
    }
    
    if (!role || !Object.values(ROLES).includes(role)) {
      return next(createError(`Invalid role. Must be one of: ${Object.values(ROLES).join(', ')}`, 400));
    }
    
    // Check if user exists
    const userExists = await prisma.user.findUnique({
      where: { id: userId }
    });
    
    if (!userExists) {
      return next(createError('User not found', 404));
    }
    
    // Update user's role
    await prisma.user.update({
      where: { id: userId },
      data: { role }
    });
    
    res.json({
      success: true,
      message: `Role '${role}' assigned to user successfully`
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  getAllRoles,
  getUserPermissions,
  assignRole,
  ROLES,
  PERMISSIONS
}; 