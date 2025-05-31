/**
 * User Controller
 * Handles user profile management and user operations
 */
import { Response } from 'express';
import { AuthenticatedRequest } from '../types/auth';
/**
 * Get current user profile
 * @route GET /api/users/profile
 * @access Private
 */
export declare const getProfile: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Update current user profile
 * @route PUT /api/users/profile
 * @access Private
 */
export declare const updateProfile: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Get all users (admin only)
 * @route GET /api/users
 * @access Private/Admin
 */
export declare const getUsers: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Update user role (admin only)
 * @route PUT /api/users/:id/role
 * @access Private/Admin
 */
export declare const updateUserRole: (req: AuthenticatedRequest, res: Response) => Promise<void>;
