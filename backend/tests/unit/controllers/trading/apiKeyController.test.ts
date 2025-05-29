/**
 * API Key Controller Tests
 */

import { Request, Response } from 'express';
import * as apiKeyController from '../../../../src/controllers/trading/apiKeyController';
import * as apiKeyManagement from '../../../../src/services/apiKeyManagementService';

// Mock the apiKeyManagement service
jest.mock('../../../../src/services/apiKeyManagementService');

// Prepare mocks for the request and response objects
const mockRequest = (data: any = {}) => {
  return {
    user: { id: 'user-123' },
    params: {},
    query: {},
    body: {},
    ...data
  } as unknown as Request;
};

const mockResponse = () => {
  const res: any = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  return res as Response;
};

describe('API Key Controller', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getAllApiKeys', () => {
    it('should return all API keys for a user', async () => {
      // Mock data
      const mockApiKeys = [
        { id: 'key-1', name: 'Test Key 1', maskedKey: 'abcd****1234' },
        { id: 'key-2', name: 'Test Key 2', maskedKey: 'efgh****5678' }
      ];

      // Mock service
      (apiKeyManagement.listApiKeys as jest.Mock).mockResolvedValue(mockApiKeys);

      // Create mock request and response
      const req = mockRequest();
      const res = mockResponse();

      // Call controller
      await apiKeyController.getAllApiKeys(req, res);

      // Verify service was called
      expect(apiKeyManagement.listApiKeys).toHaveBeenCalledWith('user-123', {
        environment: undefined,
        includeRevoked: false
      });

      // Verify response
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        data: mockApiKeys
      });
    });

    it('should return unauthorized if no user', async () => {
      // Create mock request with no user
      const req = mockRequest({ user: null });
      const res = mockResponse();

      // Call controller
      await apiKeyController.getAllApiKeys(req, res);

      // Verify service was not called
      expect(apiKeyManagement.listApiKeys).not.toHaveBeenCalled();

      // Verify response
      expect(res.status).toHaveBeenCalledWith(401);
      expect(res.json).toHaveBeenCalledWith({ error: 'Authentication required' });
    });
  });

  describe('getApiKeyById', () => {
    it('should return API key details', async () => {
      // Mock data
      const mockApiKeys = [
        { id: 'key-1', name: 'Test Key 1', maskedKey: 'abcd****1234' },
        { id: 'key-2', name: 'Test Key 2', maskedKey: 'efgh****5678' }
      ];
      
      const mockUsageStats = {
        basicStats: { usageCount: 5 },
        recentActivity: [],
        apiCalls: { totalCalls: 5 }
      };

      // Mock services
      (apiKeyManagement.listApiKeys as jest.Mock).mockResolvedValue(mockApiKeys);
      (apiKeyManagement.getApiKeyUsageStats as jest.Mock).mockResolvedValue(mockUsageStats);

      // Create mock request and response
      const req = mockRequest({ params: { id: 'key-1' } });
      const res = mockResponse();

      // Call controller
      await apiKeyController.getApiKeyById(req, res);

      // Verify service was called
      expect(apiKeyManagement.listApiKeys).toHaveBeenCalledWith('user-123');
      expect(apiKeyManagement.getApiKeyUsageStats).toHaveBeenCalledWith('user-123', 'key-1');

      // Verify response
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        data: {
          ...mockApiKeys[0],
          usageStats: mockUsageStats
        }
      });
    });

    it('should return 404 if API key not found', async () => {
      // Mock data
      const mockApiKeys = [
        { id: 'key-1', name: 'Test Key 1', maskedKey: 'abcd****1234' }
      ];

      // Mock service
      (apiKeyManagement.listApiKeys as jest.Mock).mockResolvedValue(mockApiKeys);

      // Create mock request and response
      const req = mockRequest({ params: { id: 'non-existent-key' } });
      const res = mockResponse();

      // Call controller
      await apiKeyController.getApiKeyById(req, res);

      // Verify response
      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'API key not found'
      });
    });
  });

  describe('createApiKey', () => {
    it('should create a new API key', async () => {
      // Mock data
      const newApiKey = {
        id: 'key-123',
        userId: 'user-123',
        name: 'New Key',
        maskedKey: 'abcd****1234',
        scopes: ['read', 'trade'],
        environment: 'testnet',
        expiry: new Date('2023-12-31'),
        isDefault: false,
        isRevoked: false
      };

      // Mock service
      (apiKeyManagement.addApiKey as jest.Mock).mockResolvedValue(newApiKey);

      // Create mock request and response
      const req = mockRequest({
        body: {
          name: 'New Key',
          key: 'abcd1234efgh5678',
          secret: 'secret123456',
          environment: 'testnet',
          scopes: ['read', 'trade']
        }
      });
      const res = mockResponse();

      // Call controller
      await apiKeyController.createApiKey(req, res);

      // Verify service was called
      expect(apiKeyManagement.addApiKey).toHaveBeenCalledWith({
        userId: 'user-123',
        name: 'New Key',
        key: 'abcd1234efgh5678',
        secret: 'secret123456',
        environment: 'testnet',
        scopes: ['read', 'trade'],
        expiry: expect.any(Date),
        isDefault: false,
        ipRestrictions: undefined
      }, true);

      // Verify response
      expect(res.status).toHaveBeenCalledWith(201);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        message: 'API key created successfully',
        data: newApiKey
      });
    });
  });

  describe('revokeApiKey', () => {
    it('should revoke an API key', async () => {
      // Mock service
      (apiKeyManagement.revokeApiKey as jest.Mock).mockResolvedValue(true);

      // Create mock request and response
      const req = mockRequest({
        params: { id: 'key-123' },
        body: { reason: 'No longer needed' }
      });
      const res = mockResponse();

      // Call controller
      await apiKeyController.revokeApiKey(req, res);

      // Verify service was called
      expect(apiKeyManagement.revokeApiKey).toHaveBeenCalledWith(
        'user-123',
        'key-123',
        'user-123', // revokedBy
        'No longer needed'
      );

      // Verify response
      expect(res.status).toHaveBeenCalledWith(200);
      expect(res.json).toHaveBeenCalledWith({
        success: true,
        message: 'API key revoked successfully'
      });
    });

    it('should return 404 if API key not found', async () => {
      // Mock service
      (apiKeyManagement.revokeApiKey as jest.Mock).mockResolvedValue(false);

      // Create mock request and response
      const req = mockRequest({
        params: { id: 'non-existent-key' },
        body: { reason: 'No longer needed' }
      });
      const res = mockResponse();

      // Call controller
      await apiKeyController.revokeApiKey(req, res);

      // Verify response
      expect(res.status).toHaveBeenCalledWith(404);
      expect(res.json).toHaveBeenCalledWith({
        success: false,
        error: 'API key not found or already revoked'
      });
    });
  });
}); 