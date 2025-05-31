/**
 * Session Controller
 * Handles session management, device tracking, and session security
 */
import { Response } from 'express';
/**
 * Get all active sessions for the current user
 * @route GET /api/sessions
 * @access Private
 */
export declare const getUserSessions: (req: import("express").Request, res: Response, next: import("express").NextFunction) => Promise<void>;
/**
 * Check current session status
 * @route GET /api/sessions/check
 * @access Private
 */
export declare const checkSession: (req: import("express").Request, res: Response, next: import("express").NextFunction) => Promise<void>;
/**
 * Revoke a specific session
 * @route DELETE /api/sessions/:sessionId
 * @access Private
 */
export declare const revokeSession: (req: import("express").Request, res: Response, next: import("express").NextFunction) => Promise<void>;
/**
 * Revoke all other sessions except current one
 * @route DELETE /api/sessions
 * @access Private
 */
export declare const revokeAllSessions: (req: import("express").Request, res: Response, next: import("express").NextFunction) => Promise<void>;
declare const _default: {
    getUserSessions: (req: import("express").Request, res: Response, next: import("express").NextFunction) => Promise<void>;
    checkSession: (req: import("express").Request, res: Response, next: import("express").NextFunction) => Promise<void>;
    revokeSession: (req: import("express").Request, res: Response, next: import("express").NextFunction) => Promise<void>;
    revokeAllSessions: (req: import("express").Request, res: Response, next: import("express").NextFunction) => Promise<void>;
};
export default _default;
