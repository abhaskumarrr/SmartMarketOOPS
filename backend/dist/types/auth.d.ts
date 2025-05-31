/**
 * Authentication and Authorization Types
 */
import { Request } from 'express';
export interface AuthUser {
    id: string;
    name: string;
    email: string;
    role: string;
    isVerified: boolean;
    sessionId?: string;
}
export interface AuthenticatedRequest extends Request {
    user?: AuthUser;
    suspiciousActivity?: boolean;
}
export interface JwtPayload {
    id: string;
    role: string;
}
export type Permission = 'users:read' | 'users:create' | 'users:update' | 'users:delete' | 'profile:read' | 'profile:update' | 'apiKeys:read' | 'apiKeys:create' | 'apiKeys:delete' | 'bots:read' | 'bots:create' | 'bots:update' | 'bots:delete' | 'bots:execute' | 'trading:read' | 'trading:execute' | 'admin:access' | 'admin:manageRoles' | 'admin:system';
export interface Role {
    name: string;
    description: string;
    permissions: Permission[];
}
export interface RolePermissionMap {
    [key: string]: Permission[];
}
