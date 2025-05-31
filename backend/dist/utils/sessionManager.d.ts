/**
 * Session Manager
 * Handles session creation, validation, and management
 */
import { Request } from 'express';
interface Session {
    id: string;
    userId: string;
    token: string;
    refreshToken?: string | null;
    ipAddress?: string | null;
    userAgent?: string | null;
    deviceId?: string | null;
    isValid: boolean;
    expiresAt: Date;
    lastActiveAt: Date;
    createdAt: Date;
    updatedAt: Date;
    rememberMe: boolean;
    metadata?: any | null;
}
interface SessionConfig {
    maxAge: number;
    rememberMeMaxAge: number;
    inactivityTimeout: number;
}
/**
 * Create a new session for a user
 * @param userId - User ID
 * @param req - Express request object for extracting client info
 * @param rememberMe - Whether to extend session duration
 * @returns Session with tokens
 */
export declare const createSession: (userId: string, req: Request, rememberMe?: boolean) => Promise<{
    token: string;
    refreshToken: string;
    session: Session;
}>;
/**
 * Validate a session
 * @param token - JWT token
 * @returns Valid session or null
 */
export declare const validateSession: (token: string) => Promise<Session | null>;
/**
 * Update session's last activity timestamp
 * @param token - JWT token
 */
export declare const updateSessionActivity: (token: string) => Promise<void>;
/**
 * Invalidate a single session
 * @param token - JWT token
 * @returns Success indicator
 */
export declare const invalidateSession: (token: string) => Promise<boolean>;
/**
 * Revoke all sessions for a user except current one
 * @param userId - User ID
 * @param currentToken - Current session token to preserve
 * @returns Number of sessions invalidated
 */
export declare const revokeAllSessions: (userId: string, currentToken?: string) => Promise<number>;
/**
 * Get all active sessions for a user
 * @param userId - User ID
 * @returns List of active sessions
 */
export declare const getUserSessions: (userId: string) => Promise<Session[]>;
/**
 * Extend session duration (e.g., after user activity)
 * @param token - JWT token
 * @param extendBy - Milliseconds to extend by (defaults to session max age)
 */
export declare const extendSession: (token: string, extendBy?: number) => Promise<void>;
/**
 * Clean up expired sessions (can be run as a scheduled job)
 */
export declare const cleanupExpiredSessions: () => Promise<number>;
/**
 * Generate fingerprint for device identification
 * @param req - Express request
 * @returns Device fingerprint string
 */
export declare const generateDeviceFingerprint: (req: Request) => string;
declare const _default: {
    createSession: (userId: string, req: Request, rememberMe?: boolean) => Promise<{
        token: string;
        refreshToken: string;
        session: Session;
    }>;
    validateSession: (token: string) => Promise<Session | null>;
    updateSessionActivity: (token: string) => Promise<void>;
    extendSession: (token: string, extendBy?: number) => Promise<void>;
    invalidateSession: (token: string) => Promise<boolean>;
    refreshSession: any;
    getUserSessions: (userId: string) => Promise<Session[]>;
    cleanupExpiredSessions: () => Promise<number>;
    generateDeviceFingerprint: (req: Request) => string;
    getSessionMetadata: any;
    updateSessionMetadata: any;
    sessionConfig: SessionConfig;
};
export default _default;
