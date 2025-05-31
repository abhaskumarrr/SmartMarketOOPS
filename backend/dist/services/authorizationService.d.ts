/**
 * Authorization Service
 * Manages roles, permissions, and access control
 */
import { Permission, Role } from '../types/auth';
/**
 * Get all available roles
 * @returns Object containing all role definitions
 */
export declare const getAllRoles: () => Record<string, Role>;
/**
 * Get a specific role definition
 * @param roleName - Name of the role to retrieve
 * @returns Role definition or undefined if not found
 */
export declare const getRole: (roleName: string) => Role | undefined;
/**
 * Check if a role has a specific permission
 * @param roleName - Name of the role to check
 * @param permission - Permission to check for
 * @returns Boolean indicating if the role has the permission
 */
export declare const hasPermission: (roleName: string, permission: Permission) => boolean;
/**
 * Check if a role has all of the specified permissions
 * @param roleName - Name of the role to check
 * @param permissions - Array of permissions to check for
 * @returns Boolean indicating if the role has all the permissions
 */
export declare const hasAllPermissions: (roleName: string, permissions: Permission[]) => boolean;
/**
 * Check if a role has any of the specified permissions
 * @param roleName - Name of the role to check
 * @param permissions - Array of permissions to check for
 * @returns Boolean indicating if the role has any of the permissions
 */
export declare const hasAnyPermission: (roleName: string, permissions: Permission[]) => boolean;
/**
 * Get all permissions for a role
 * @param roleName - Name of the role
 * @returns Array of permissions or empty array if role not found
 */
export declare const getRolePermissions: (roleName: string) => Permission[];
/**
 * Get all available permissions across all roles
 * @returns Array of unique permissions
 */
export declare const getAllPermissions: () => Permission[];
/**
 * Check if role exists
 * @param roleName - Name of the role to check
 * @returns Boolean indicating if the role exists
 */
export declare const roleExists: (roleName: string) => boolean;
declare const _default: {
    getAllRoles: () => Record<string, Role>;
    getRole: (roleName: string) => Role | undefined;
    hasPermission: (roleName: string, permission: Permission) => boolean;
    hasAllPermissions: (roleName: string, permissions: Permission[]) => boolean;
    hasAnyPermission: (roleName: string, permissions: Permission[]) => boolean;
    getRolePermissions: (roleName: string) => Permission[];
    getAllPermissions: () => Permission[];
    roleExists: (roleName: string) => boolean;
};
export default _default;
