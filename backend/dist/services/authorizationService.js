"use strict";
/**
 * Authorization Service
 * Manages roles, permissions, and access control
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.roleExists = exports.getAllPermissions = exports.getRolePermissions = exports.hasAnyPermission = exports.hasAllPermissions = exports.hasPermission = exports.getRole = exports.getAllRoles = void 0;
/**
 * Role Definitions
 * Defines the roles and their associated permissions
 */
const roleDefinitions = {
    // User role - basic access for regular users
    user: {
        name: 'user',
        description: 'Regular user with basic permissions',
        permissions: [
            'profile:read',
            'profile:update',
            'apiKeys:read',
            'apiKeys:create',
            'apiKeys:delete',
            'bots:read',
            'bots:create',
            'bots:update',
            'bots:delete',
            'bots:execute',
            'trading:read',
            'trading:execute'
        ]
    },
    // Analyst role - can view but has limited execution abilities
    analyst: {
        name: 'analyst',
        description: 'Analyst with read permissions and limited execution abilities',
        permissions: [
            'profile:read',
            'profile:update',
            'apiKeys:read',
            'bots:read',
            'trading:read'
        ]
    },
    // Manager role - extended permissions for managing teams
    manager: {
        name: 'manager',
        description: 'Manager with extended permissions for team management',
        permissions: [
            'profile:read',
            'profile:update',
            'apiKeys:read',
            'apiKeys:create',
            'apiKeys:delete',
            'bots:read',
            'bots:create',
            'bots:update',
            'bots:delete',
            'bots:execute',
            'trading:read',
            'trading:execute',
            'users:read'
        ]
    },
    // Admin role - full system access
    admin: {
        name: 'admin',
        description: 'Administrator with full system access',
        permissions: [
            'profile:read',
            'profile:update',
            'apiKeys:read',
            'apiKeys:create',
            'apiKeys:delete',
            'bots:read',
            'bots:create',
            'bots:update',
            'bots:delete',
            'bots:execute',
            'trading:read',
            'trading:execute',
            'users:read',
            'users:create',
            'users:update',
            'users:delete',
            'admin:access',
            'admin:manageRoles',
            'admin:system'
        ]
    }
};
/**
 * Get all available roles
 * @returns Object containing all role definitions
 */
const getAllRoles = () => {
    return { ...roleDefinitions };
};
exports.getAllRoles = getAllRoles;
/**
 * Get a specific role definition
 * @param roleName - Name of the role to retrieve
 * @returns Role definition or undefined if not found
 */
const getRole = (roleName) => {
    return roleDefinitions[roleName];
};
exports.getRole = getRole;
/**
 * Check if a role has a specific permission
 * @param roleName - Name of the role to check
 * @param permission - Permission to check for
 * @returns Boolean indicating if the role has the permission
 */
const hasPermission = (roleName, permission) => {
    const role = roleDefinitions[roleName];
    if (!role)
        return false;
    return role.permissions.includes(permission);
};
exports.hasPermission = hasPermission;
/**
 * Check if a role has all of the specified permissions
 * @param roleName - Name of the role to check
 * @param permissions - Array of permissions to check for
 * @returns Boolean indicating if the role has all the permissions
 */
const hasAllPermissions = (roleName, permissions) => {
    const role = roleDefinitions[roleName];
    if (!role)
        return false;
    return permissions.every(permission => role.permissions.includes(permission));
};
exports.hasAllPermissions = hasAllPermissions;
/**
 * Check if a role has any of the specified permissions
 * @param roleName - Name of the role to check
 * @param permissions - Array of permissions to check for
 * @returns Boolean indicating if the role has any of the permissions
 */
const hasAnyPermission = (roleName, permissions) => {
    const role = roleDefinitions[roleName];
    if (!role)
        return false;
    return permissions.some(permission => role.permissions.includes(permission));
};
exports.hasAnyPermission = hasAnyPermission;
/**
 * Get all permissions for a role
 * @param roleName - Name of the role
 * @returns Array of permissions or empty array if role not found
 */
const getRolePermissions = (roleName) => {
    const role = roleDefinitions[roleName];
    if (!role)
        return [];
    return [...role.permissions];
};
exports.getRolePermissions = getRolePermissions;
/**
 * Get all available permissions across all roles
 * @returns Array of unique permissions
 */
const getAllPermissions = () => {
    const allPermissions = new Set();
    Object.values(roleDefinitions).forEach(role => {
        role.permissions.forEach(permission => {
            allPermissions.add(permission);
        });
    });
    return Array.from(allPermissions);
};
exports.getAllPermissions = getAllPermissions;
/**
 * Check if role exists
 * @param roleName - Name of the role to check
 * @returns Boolean indicating if the role exists
 */
const roleExists = (roleName) => {
    return roleName in roleDefinitions;
};
exports.roleExists = roleExists;
exports.default = {
    getAllRoles: exports.getAllRoles,
    getRole: exports.getRole,
    hasPermission: exports.hasPermission,
    hasAllPermissions: exports.hasAllPermissions,
    hasAnyPermission: exports.hasAnyPermission,
    getRolePermissions: exports.getRolePermissions,
    getAllPermissions: exports.getAllPermissions,
    roleExists: exports.roleExists
};
//# sourceMappingURL=authorizationService.js.map