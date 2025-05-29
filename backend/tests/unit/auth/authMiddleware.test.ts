/**
 * Authentication Middleware Tests
 * Tests for the authentication middleware functionality
 */

import { Request, Response } from 'express';
import { requirePermission, requireAnyPermission } from '../../../src/middleware/auth';
import authorizationService from '../../../src/services/authorizationService';
import { AuthenticatedRequest } from '../../../src/types/auth';

// Mock the authorization service
jest.mock('../../../src/services/authorizationService');

describe('Auth Middleware', () => {
  let mockRequest: Partial<AuthenticatedRequest>;
  let mockResponse: Partial<Response>;
  let nextFunction: jest.Mock;

  beforeEach(() => {
    mockRequest = {
      user: {
        id: '1',
        name: 'Test User',
        email: 'test@example.com',
        role: 'user',
        isVerified: true
      }
    };
    
    mockResponse = {
      status: jest.fn().mockReturnThis(),
      json: jest.fn()
    };
    
    nextFunction = jest.fn();
    
    // Reset all mocks
    jest.clearAllMocks();
  });

  describe('requirePermission', () => {
    it('should call next() if user has all required permissions', () => {
      // Mock the hasAllPermissions method to return true
      (authorizationService.hasAllPermissions as jest.Mock).mockReturnValue(true);
      
      const middleware = requirePermission(['profile:read']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(authorizationService.hasAllPermissions).toHaveBeenCalledWith('user', ['profile:read']);
      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should return 401 if user is not authenticated', () => {
      mockRequest = {}; // No user
      
      const middleware = requirePermission(['profile:read']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        message: 'Not authenticated'
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 403 if user does not have required permissions', () => {
      // Mock the hasAllPermissions method to return false
      (authorizationService.hasAllPermissions as jest.Mock).mockReturnValue(false);
      
      const middleware = requirePermission(['admin:access']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(authorizationService.hasAllPermissions).toHaveBeenCalledWith('user', ['admin:access']);
      expect(mockResponse.status).toHaveBeenCalledWith(403);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        message: 'Insufficient permissions. Required: admin:access'
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });
  });

  describe('requireAnyPermission', () => {
    it('should call next() if user has any of the required permissions', () => {
      // Mock the hasAnyPermission method to return true
      (authorizationService.hasAnyPermission as jest.Mock).mockReturnValue(true);
      
      const middleware = requireAnyPermission(['admin:access', 'profile:read']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(authorizationService.hasAnyPermission).toHaveBeenCalledWith(
        'user', 
        ['admin:access', 'profile:read']
      );
      expect(nextFunction).toHaveBeenCalled();
      expect(mockResponse.status).not.toHaveBeenCalled();
    });

    it('should return 401 if user is not authenticated', () => {
      mockRequest = {}; // No user
      
      const middleware = requireAnyPermission(['admin:access', 'profile:read']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(mockResponse.status).toHaveBeenCalledWith(401);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        message: 'Not authenticated'
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });

    it('should return 403 if user does not have any of the required permissions', () => {
      // Mock the hasAnyPermission method to return false
      (authorizationService.hasAnyPermission as jest.Mock).mockReturnValue(false);
      
      const middleware = requireAnyPermission(['admin:access', 'admin:manageRoles']);
      middleware(mockRequest as AuthenticatedRequest, mockResponse as Response, nextFunction);
      
      expect(authorizationService.hasAnyPermission).toHaveBeenCalledWith(
        'user', 
        ['admin:access', 'admin:manageRoles']
      );
      expect(mockResponse.status).toHaveBeenCalledWith(403);
      expect(mockResponse.json).toHaveBeenCalledWith({
        success: false,
        message: 'Insufficient permissions. Required any of: admin:access, admin:manageRoles'
      });
      expect(nextFunction).not.toHaveBeenCalled();
    });
  });
}); 