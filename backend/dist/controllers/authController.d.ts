/**
 * Authentication Controller
 * Handles user registration, login, token management, password reset, and OAuth
 */
import { Request, Response } from 'express';
interface AuthenticatedRequest extends Request {
    user?: any;
}
/**
 * Register a new user
 * @route POST /api/auth/register
 * @access Public
 */
export declare const register: (req: Request, res: Response) => Promise<void>;
/**
 * Verify user email
 * @route GET /api/auth/verify-email/:token
 * @access Public
 */
export declare const verifyEmail: (req: Request, res: Response) => Promise<void>;
/**
 * Authenticate user and get token
 * @route POST /api/auth/login
 * @access Public
 */
export declare const login: (req: Request, res: Response) => Promise<void>;
/**
 * Refresh access token using refresh token with rotation
 * @route POST /api/auth/refresh-token
 * @access Public
 */
export declare const refreshToken: (req: Request, res: Response) => Promise<void>;
/**
 * Forgot password - send reset email
 * @route POST /api/auth/forgot-password
 * @access Public
 */
export declare const forgotPassword: (req: Request, res: Response) => Promise<void>;
/**
 * Reset password with token
 * @route POST /api/auth/reset-password
 * @access Public
 */
export declare const resetPassword: (req: Request, res: Response) => Promise<void>;
/**
 * OAuth login/signup
 * @route POST /api/auth/oauth
 * @access Public
 */
export declare const oauthLogin: (req: Request, res: Response) => Promise<void>;
/**
 * Logout user
 * @route POST /api/auth/logout
 * @access Private
 */
export declare const logout: (req: AuthenticatedRequest, res: Response) => Promise<void>;
declare const _default: {
    register: (req: Request, res: Response) => Promise<void>;
    verifyEmail: (req: Request, res: Response) => Promise<void>;
    login: (req: Request, res: Response) => Promise<void>;
    refreshToken: (req: Request, res: Response) => Promise<void>;
    forgotPassword: (req: Request, res: Response) => Promise<void>;
    resetPassword: (req: Request, res: Response) => Promise<void>;
    oauthLogin: (req: Request, res: Response) => Promise<void>;
    logout: (req: AuthenticatedRequest, res: Response) => Promise<void>;
};
export default _default;
