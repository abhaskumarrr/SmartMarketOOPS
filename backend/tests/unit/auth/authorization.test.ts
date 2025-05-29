/**
 * Authorization Service Tests
 * Tests for the role-based authorization system
 */

import authorizationService from '../../../src/services/authorizationService';
import { Permission } from '../../../src/types/auth';

describe('Authorization Service', () => {
  describe('Role Definitions', () => {
    it('should define user role with correct permissions', () => {
      const role = authorizationService.getRole('user');
      expect(role).toBeDefined();
      expect(role?.name).toBe('user');
      expect(role?.permissions).toContain('profile:read');
      expect(role?.permissions).toContain('profile:update');
      expect(role?.permissions).toContain('apiKeys:read');
      expect(role?.permissions).not.toContain('admin:access');
    });

    it('should define admin role with all permissions', () => {
      const role = authorizationService.getRole('admin');
      expect(role).toBeDefined();
      expect(role?.name).toBe('admin');
      expect(role?.permissions).toContain('admin:access');
      expect(role?.permissions).toContain('admin:manageRoles');
      expect(role?.permissions).toContain('users:read');
      expect(role?.permissions).toContain('users:update');
    });
    
    it('should define analyst role with limited permissions', () => {
      const role = authorizationService.getRole('analyst');
      expect(role).toBeDefined();
      expect(role?.name).toBe('analyst');
      expect(role?.permissions).toContain('profile:read');
      expect(role?.permissions).toContain('bots:read');
      expect(role?.permissions).not.toContain('bots:execute');
      expect(role?.permissions).not.toContain('admin:access');
    });
  });

  describe('Permission Checks', () => {
    it('should correctly check if a role has a specific permission', () => {
      expect(authorizationService.hasPermission('admin', 'admin:access')).toBe(true);
      expect(authorizationService.hasPermission('user', 'admin:access')).toBe(false);
      expect(authorizationService.hasPermission('user', 'profile:read')).toBe(true);
      expect(authorizationService.hasPermission('analyst', 'bots:execute')).toBe(false);
    });

    it('should handle non-existent roles gracefully', () => {
      expect(authorizationService.hasPermission('nonexistent', 'admin:access')).toBe(false);
    });

    it('should check if a role has all specified permissions', () => {
      expect(authorizationService.hasAllPermissions('admin', ['admin:access', 'users:read'])).toBe(true);
      expect(authorizationService.hasAllPermissions('user', ['profile:read', 'admin:access'])).toBe(false);
    });

    it('should check if a role has any of the specified permissions', () => {
      expect(authorizationService.hasAnyPermission('user', ['admin:access', 'profile:read'])).toBe(true);
      expect(authorizationService.hasAnyPermission('analyst', ['bots:execute', 'admin:access'])).toBe(false);
    });
  });

  describe('Role Management', () => {
    it('should get all permissions for a role', () => {
      const permissions = authorizationService.getRolePermissions('user');
      expect(permissions).toBeInstanceOf(Array);
      expect(permissions.length).toBeGreaterThan(0);
      expect(permissions).toContain('profile:read');
    });

    it('should return empty array for non-existent role', () => {
      const permissions = authorizationService.getRolePermissions('nonexistent');
      expect(permissions).toBeInstanceOf(Array);
      expect(permissions.length).toBe(0);
    });

    it('should get all available roles', () => {
      const roles = authorizationService.getAllRoles();
      expect(Object.keys(roles)).toContain('user');
      expect(Object.keys(roles)).toContain('admin');
      expect(Object.keys(roles)).toContain('manager');
      expect(Object.keys(roles)).toContain('analyst');
    });

    it('should get all available permissions', () => {
      const permissions = authorizationService.getAllPermissions();
      expect(permissions).toBeInstanceOf(Array);
      expect(permissions.length).toBeGreaterThan(0);
      expect(permissions).toContain('admin:access');
      expect(permissions).toContain('profile:read');
    });

    it('should check if a role exists', () => {
      expect(authorizationService.roleExists('admin')).toBe(true);
      expect(authorizationService.roleExists('nonexistent')).toBe(false);
    });
  });
}); 