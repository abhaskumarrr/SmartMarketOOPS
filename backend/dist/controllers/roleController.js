"use strict";
/**
 * Role Controller
 * Handles role management and permission operations
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.checkPermission = exports.getUsersByRole = exports.getRole = exports.getAllPermissions = exports.getAllRoles = void 0;
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
const env_1 = __importDefault(require("../utils/env"));
const authorizationService_1 = __importDefault(require("../services/authorizationService"));
const logger_1 = require("../utils/logger");
// Create logger
const logger = (0, logger_1.createLogger)('RoleController');
/**
 * Get all roles with permissions
 * @route GET /api/roles
 * @access Private/Admin
 */
const getAllRoles = async (req, res) => {
    try {
        if (!req.user) {
            res.status(401).json({
                success: false,
                message: 'Not authenticated'
            });
            return;
        }
        const roles = authorizationService_1.default.getAllRoles();
        res.status(200).json({
            success: true,
            data: roles
        });
    }
    catch (error) {
        console.error('Get roles error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error while fetching roles',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.getAllRoles = getAllRoles;
/**
 * Get all permissions
 * @route GET /api/roles/permissions
 * @access Private/Admin
 */
const getAllPermissions = async (req, res) => {
    try {
        if (!req.user) {
            res.status(401).json({
                success: false,
                message: 'Not authenticated'
            });
            return;
        }
        const permissions = authorizationService_1.default.getAllPermissions();
        res.status(200).json({
            success: true,
            data: permissions
        });
    }
    catch (error) {
        console.error('Get permissions error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error while fetching permissions',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.getAllPermissions = getAllPermissions;
/**
 * Get role details
 * @route GET /api/roles/:roleName
 * @access Private/Admin
 */
const getRole = async (req, res) => {
    try {
        if (!req.user) {
            res.status(401).json({
                success: false,
                message: 'Not authenticated'
            });
            return;
        }
        const { roleName } = req.params;
        const role = authorizationService_1.default.getRole(roleName);
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
    }
    catch (error) {
        console.error('Get role error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error while fetching role',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.getRole = getRole;
/**
 * Get users by role
 * @route GET /api/roles/:roleName/users
 * @access Private/Admin
 */
const getUsersByRole = async (req, res) => {
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
        if (!authorizationService_1.default.roleExists(roleName)) {
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
        const users = await prismaClient_1.default.user.findMany({
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
        const total = await prismaClient_1.default.user.count({
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
    }
    catch (error) {
        console.error('Get users by role error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error while fetching users by role',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.getUsersByRole = getUsersByRole;
/**
 * Check if current user has permission
 * @route GET /api/roles/check-permission/:permission
 * @access Private
 */
const checkPermission = async (req, res) => {
    try {
        if (!req.user) {
            res.status(401).json({
                success: false,
                message: 'Not authenticated'
            });
            return;
        }
        const { permission } = req.params;
        const hasPermission = authorizationService_1.default.hasPermission(req.user.role, permission);
        res.status(200).json({
            success: true,
            data: {
                hasPermission
            }
        });
    }
    catch (error) {
        console.error('Check permission error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error while checking permission',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.checkPermission = checkPermission;
exports.default = {
    getAllRoles: exports.getAllRoles,
    getAllPermissions: exports.getAllPermissions,
    getRole: exports.getRole,
    getUsersByRole: exports.getUsersByRole,
    checkPermission: exports.checkPermission
};
//# sourceMappingURL=roleController.js.map