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
  sessionId?: string; // Optional session ID for session management
}

export interface AuthenticatedRequest extends Request {
  user?: AuthUser;
  suspiciousActivity?: boolean; // Flag for potential security issues
}

export interface JwtPayload {
  id: string;
  role: string;
}

export type Permission = 
  // User Management
  | 'users:read'
  | 'users:create'
  | 'users:update'
  | 'users:delete'
  
  // Profile Management
  | 'profile:read'
  | 'profile:update'
  
  // API Key Management
  | 'apiKeys:read'
  | 'apiKeys:create'
  | 'apiKeys:delete'
  
  // Bot Management
  | 'bots:read'
  | 'bots:create'
  | 'bots:update'
  | 'bots:delete'
  | 'bots:execute'
  
  // Trading Operations
  | 'trading:read'
  | 'trading:execute'
  
  // Admin Operations
  | 'admin:access'
  | 'admin:manageRoles'
  | 'admin:system';

export interface Role {
  name: string;
  description: string;
  permissions: Permission[];
}

export interface RolePermissionMap {
  [key: string]: Permission[];
} 