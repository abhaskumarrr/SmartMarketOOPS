/**
 * Session Controller
 * Handles session management, device tracking, and session security
 */
import { Response } from 'express';
import { AuthenticatedRequest } from '../types/auth';
/**
 * Get all active sessions for the current user
 * @route GET /api/sessions
 * @access Private
 */
export declare const getUserSessions: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Check current session status
 * @route GET /api/sessions/check
 * @access Private
 */
export declare const checkSession: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Revoke a specific session
 * @route DELETE /api/sessions/:sessionId
 * @access Private
 */
export declare const revokeSession: (req: AuthenticatedRequest, res: Response) => Promise<void>;
/**
 * Revoke all other sessions except current one
 * @route DELETE /api/sessions
 * @access Private
 */
export declare const revokeAllSessions: (req: AuthenticatedRequest, res: Response) => Promise<void>;
declare const _default: {
    getUserSessions: (req: AuthenticatedRequest, res: Response) => Promise<void>;
    checkSession: (req: AuthenticatedRequest, res: Response) => Promise<void>;
    revokeSession: (req: AuthenticatedRequest, res: Response) => Promise<void>;
    revokeAllSessions: (req: AuthenticatedRequest, res: Response) => Promise<void>;
};
export default _default;
