/**
 * Authentication Roles and Permissions Tests
 * Tests for role-based access control and permission checks
 */

import { Request, Response, NextFunction } from 'express';
import { requireRole, requirePermission, requireAnyPermission } from '../../../src/middleware/auth';
import { AuthenticatedRequest, Permission, AuthUser } from '../../../src/types/auth';
import authorizationService from '../../../src/services/authorizationService';

// Mock the authorization service
jest.mock('../../../src/services/authorizationService', () => ({
  hasAllPermissions: jest.fn(),
  hasAnyPermission: jest.fn(),
  getRolePermissions: jest.fn(),
  isAdmin: jest.fn()
}));

describe('Role and Permission Based Authorization', () => {
  let mockRequest: Partial<AuthenticatedRequest>;
  let mockResponse: Partial<Response>;
  let nextFunction: jest.Mock;

  beforeEach(() => {
    mockRequest = {
      user: {
        id: 'user-123',
        email: 'user@example.com',
        name: 'Test User',
        role: 'user',
        isVerified: true
      }
    };
    
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn()
    };
    
    nextFunction = jest.fn();
    
    jest.clearAllMocks();
  });

  describe('Role-Based Access Control', () => {
    it('should allow access for users with the required role', () => {
      // Setup - user with 'admin' role trying to access admin route
      const adminUser: AuthUser = {
        id: 'admin-123',
        email: 'admin@example.com',
        name: 'Admin User',
        role: 'admin',
        isVerified: true
      };
      
      mockRequest.user = adminUser;
      
      const middleware = requireRole(['admin']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should allow access for users with any of the allowed roles', () => {
      // Setup - user with 'manager' role trying to access route that allows managers or admins
      const managerUser: AuthUser = {
        id: 'manager-123',
        email: 'manager@example.com',
        name: 'Manager User',
        role: 'manager',
        isVerified: true
      };
      
      mockRequest.user = managerUser;
      
      const middleware = requireRole(['admin', 'manager']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should deny access for users without the required role', () => {
      // Setup - regular user trying to access admin route
      // User is already set up in beforeEach with 'user' role
      
      const middleware = requireRole(['admin']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(nextFunction).not.toHaveBeenCalled();
      expect(mockResponse.status).toHaveBeenCalledWith(403);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'Requires admin role'
      }));
    });

    it('should return 401 for unauthenticated requests', () => {
      // Setup - no user object (unauthenticated)
      mockRequest = {};
      
      const middleware = requireRole(['admin']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(nextFunction).not.toHaveBeenCalled();
      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'Not authenticated'
      }));
    });
  });

  describe('Permission-Based Access Control', () => {
    it('should allow access with all required permissions', () => {
      // Setup - user has all required permissions
      (authorizationService.hasAllPermissions as jest.Mock).mockReturnValue(true);
      
      const permissions: Permission[] = ['users:read', 'users:update'];
      const middleware = requirePermission(permissions);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(authorizationService.hasAllPermissions).toHaveBeenCalledWith('user', permissions);
      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should deny access without all required permissions', () => {
      // Setup - user doesn't have all required permissions
      (authorizationService.hasAllPermissions as jest.Mock).mockReturnValue(false);
      
      const permissions: Permission[] = ['users:read', 'users:update'];
      const middleware = requirePermission(permissions);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(authorizationService.hasAllPermissions).toHaveBeenCalledWith('user', permissions);
      expect(nextFunction).not.toHaveBeenCalled();
      expect(mockResponse.status).toHaveBeenCalledWith(403);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'Insufficient permissions. Required: users:read, users:update'
      }));
    });

    it('should allow access with any of the required permissions', () => {
      // Setup - user has at least one of the required permissions
      (authorizationService.hasAnyPermission as jest.Mock).mockReturnValue(true);
      
      const permissions: Permission[] = ['users:read', 'users:update'];
      const middleware = requireAnyPermission(permissions);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(authorizationService.hasAnyPermission).toHaveBeenCalledWith('user', permissions);
      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should deny access without any of the required permissions', () => {
      // Setup - user doesn't have any of the required permissions
      (authorizationService.hasAnyPermission as jest.Mock).mockReturnValue(false);
      
      const permissions: Permission[] = ['users:read', 'users:update'];
      const middleware = requireAnyPermission(permissions);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(authorizationService.hasAnyPermission).toHaveBeenCalledWith('user', permissions);
      expect(nextFunction).not.toHaveBeenCalled();
      expect(mockResponse.status).toHaveBeenCalledWith(403);
      expect(mockResponse.json).toHaveBeenCalledWith(expect.objectContaining({
        success: false,
        message: 'Insufficient permissions. Required any of: users:read, users:update'
      }));
    });
  });

  describe('Role/Permission Combinations', () => {
    it('should combine role and permission checks effectively', () => {
      // Middleware composition pattern - combining role and permission checks
      const checkAdminRole = requireRole(['admin']);
      const checkUserReadPermission = requirePermission(['users:read'] as Permission[]);
      
      // First middleware in the chain - role check
      const adminUser: AuthUser = {
        id: 'admin-123',
        email: 'admin@example.com',
        name: 'Admin User',
        role: 'admin',
        isVerified: true
      };
      
      mockRequest.user = adminUser;
      
      checkAdminRole(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      // Second middleware in the chain - permission check (if role check passes)
      if (nextFunction.mock.calls.length > 0) {
        // Reset the next function mock to check if it's called again
        nextFunction.mockReset();
        
        // Setup permission check to pass
        (authorizationService.hasAllPermissions as jest.Mock).mockReturnValue(true);
        
        checkUserReadPermission(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
        
        expect(nextFunction).toHaveBeenCalled();
      }
      
      // Complete chain should have passed
      expect(mockResponse.status).not.toHaveBeenCalled();
    });
  });
}); 