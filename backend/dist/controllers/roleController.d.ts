/**
 * Role Controller
 * Handles role management and permission operations
 */
import { Response } from 'express';
import { AuthenticatedRequest } from '../types/auth';
/**
 * Get all roles with permissions
 * @route GET /api/roles
 * @access Private/Admin
 */
export declare const getAllRoles: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get all permissions
 * @route GET /api/roles/permissions
 * @access Private/Admin
 */
export declare const getAllPermissions: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get role details
 * @route GET /api/roles/:roleName
 * @access Private/Admin
 */
export declare const getRole: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get users by role
 * @route GET /api/roles/:roleName/users
 * @access Private/Admin
 */
export declare const getUsersByRole: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Check if current user has permission
 * @route GET /api/roles/check-permission/:permission
 * @access Private
 */
export declare const checkPermission: (req: AuthenticatedRequest, res: Response) => Promise<void>;
declare const _default: {
    getAllRoles: (req: AuthenticatedRequest, res: Response) => Promise<void>;
    getAllPermissions: (req: AuthenticatedRequest, res: Response) => Promise<void>;
    getRole: (req: AuthenticatedRequest, res: Response) => Promise<void>;
    getUsersByRole: (req: AuthenticatedRequest, res: Response) => Promise<void>;
    checkPermission: (req: AuthenticatedRequest, res: Response) => Promise<void>;
};
export default _default;
