/**
 * Authorization Service
 * Manages roles, permissions, and access control
 */

import { Permission, Role, RolePermissionMap } from '../types/auth';

/**
 * Role Definitions
 * Defines the roles and their associated permissions
 */
const roleDefinitions: Record<string, Role> = {
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
export const getAllRoles = (): Record<string, Role> => {
  return { ...roleDefinitions };
};

/**
 * Get a specific role definition
 * @param roleName - Name of the role to retrieve
 * @returns Role definition or undefined if not found
 */
export const getRole = (roleName: string): Role | undefined => {
  return roleDefinitions[roleName];
};

/**
 * Check if a role has a specific permission
 * @param roleName - Name of the role to check
 * @param permission - Permission to check for
 * @returns Boolean indicating if the role has the permission
 */
export const hasPermission = (roleName: string, permission: Permission): boolean => {
  const role = roleDefinitions[roleName];
  if (!role) return false;
  
  return role.permissions.includes(permission);
};

/**
 * Check if a role has all of the specified permissions
 * @param roleName - Name of the role to check
 * @param permissions - Array of permissions to check for
 * @returns Boolean indicating if the role has all the permissions
 */
export const hasAllPermissions = (roleName: string, permissions: Permission[]): boolean => {
  const role = roleDefinitions[roleName];
  if (!role) return false;
  
  return permissions.every(permission => role.permissions.includes(permission));
};

/**
 * Check if a role has any of the specified permissions
 * @param roleName - Name of the role to check
 * @param permissions - Array of permissions to check for
 * @returns Boolean indicating if the role has any of the permissions
 */
export const hasAnyPermission = (roleName: string, permissions: Permission[]): boolean => {
  const role = roleDefinitions[roleName];
  if (!role) return false;
  
  return permissions.some(permission => role.permissions.includes(permission));
};

/**
 * Get all permissions for a role
 * @param roleName - Name of the role
 * @returns Array of permissions or empty array if role not found
 */
export const getRolePermissions = (roleName: string): Permission[] => {
  const role = roleDefinitions[roleName];
  if (!role) return [];
  
  return [...role.permissions];
};

/**
 * Get all available permissions across all roles
 * @returns Array of unique permissions
 */
export const getAllPermissions = (): Permission[] => {
  const allPermissions = new Set<Permission>();
  
  Object.values(roleDefinitions).forEach(role => {
    role.permissions.forEach(permission => {
      allPermissions.add(permission);
    });
  });
  
  return Array.from(allPermissions);
};

/**
 * Check if role exists
 * @param roleName - Name of the role to check
 * @returns Boolean indicating if the role exists
 */
export const roleExists = (roleName: string): boolean => {
  return roleName in roleDefinitions;
};

export default {
  getAllRoles,
  getRole,
  hasPermission,
  hasAllPermissions,
  hasAnyPermission,
  getRolePermissions,
  getAllPermissions,
  roleExists
}; 