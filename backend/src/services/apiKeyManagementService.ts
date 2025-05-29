/**
 * API Key Management Service
 * Comprehensive service for managing API keys with security, validation, and analytics
 */

import prisma from '../utils/prismaClient';
import * as apiKeyService from './apiKeyService';
import * as apiKeyEncryption from '../utils/apiKeyEncryption';
import * as apiKeyValidation from './trading/apiKeyValidationService';
import { createAuditLog } from '../utils/auditLog';

/**
 * API Key data structure
 */
interface ApiKeyData {
  id?: string;
  userId: string;
  name: string;
  key: string;
  secret: string;
  scopes: string[];
  environment: string;
  expiry: Date;
  isDefault?: boolean;
  ipRestrictions?: string[];
  rateLimits?: any;
  metadata?: any;
}

/**
 * API Key summary (without sensitive data)
 */
interface ApiKeySummary {
  id: string;
  userId: string;
  name: string;
  maskedKey: string;
  scopes: string[];
  environment: string;
  expiry: Date;
  isDefault: boolean;
  isRevoked: boolean;
  createdAt: Date;
  lastUsedAt?: Date;
  usageCount: number;
  isExpired: boolean;
}

/**
 * Add a new API key with validation
 * @param {ApiKeyData} apiKeyData - API key data
 * @param {boolean} validateKey - Whether to validate the key against Delta Exchange
 * @returns {Promise<ApiKeySummary>} The created API key summary
 */
async function addApiKey(apiKeyData: ApiKeyData, validateKey: boolean = true): Promise<ApiKeySummary> {
  // Validate API key format
  if (!apiKeyEncryption.validateApiKeyFormat(apiKeyData.key)) {
    throw new Error('Invalid API key format');
  }

  // Validate API secret format
  if (!apiKeyEncryption.validateApiSecretFormat(apiKeyData.secret)) {
    throw new Error('Invalid API secret format');
  }

  // Validate API key against Delta Exchange if requested
  if (validateKey) {
    const validationResult = await apiKeyValidation.validateDeltaApiKey(
      apiKeyData.key,
      apiKeyData.secret,
      apiKeyData.environment === 'testnet'
    );

    if (!validationResult.isValid) {
      throw new Error(`API key validation failed: ${validationResult.message}`);
    }
    
    // Also validate trading permissions
    const tradingPermissions = await apiKeyValidation.validateTradingPermissions(
      apiKeyData.key,
      apiKeyData.secret,
      apiKeyData.environment === 'testnet'
    );
    
    if (!tradingPermissions.isValid) {
      throw new Error(`Trading permission validation failed: ${tradingPermissions.message}`);
    }
  }

  // Create the API key in database
  const apiKey = await apiKeyService.addApiKey(
    apiKeyData.userId,
    apiKeyData.key,
    apiKeyData.secret,
    {
      name: apiKeyData.name,
      scopes: apiKeyData.scopes,
      expiry: apiKeyData.expiry,
      environment: apiKeyData.environment,
      ipRestrictions: apiKeyData.ipRestrictions,
      rateLimits: apiKeyData.rateLimits,
      metadata: apiKeyData.metadata
    }
  );

  // If this is set as default, unset any other default keys
  if (apiKeyData.isDefault) {
    await setDefaultApiKey(apiKeyData.userId, apiKey.id);
  }

  // Log the API key creation
  await createAuditLog({
    userId: apiKeyData.userId,
    action: 'api_key.create',
    details: {
      keyId: apiKey.id,
      name: apiKeyData.name,
      environment: apiKeyData.environment
    }
  });

  return {
    id: apiKey.id,
    userId: apiKey.userId,
    name: apiKey.name,
    maskedKey: apiKeyEncryption.maskApiKey(apiKeyData.key),
    scopes: apiKey.scopes,
    environment: apiKey.environment || 'testnet',
    expiry: apiKey.expiry,
    isDefault: apiKeyData.isDefault || false,
    isRevoked: false,
    createdAt: apiKey.createdAt,
    usageCount: 0,
    isExpired: false
  };
}

/**
 * Get API key and secret for a user (sensitive data)
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID
 * @param {string} environment - Environment (testnet/mainnet)
 * @returns {Promise<ApiKeyData|null>} API key data or null if not found
 */
async function getApiKey(
  userId: string,
  keyId?: string,
  environment?: string
): Promise<ApiKeyData | null> {
  const apiKey = await apiKeyService.getApiKey(userId, keyId, environment);

  if (!apiKey) {
    return null;
  }

  return {
    id: apiKey.id,
    userId: apiKey.userId,
    name: apiKey.name,
    key: apiKey.key as string,
    secret: apiKey.secret as string,
    scopes: apiKey.scopes,
    environment: apiKey.environment,
    expiry: apiKey.expiry
  };
}

/**
 * Get user's default API key (for automatic trading)
 * @param {string} userId - User ID
 * @param {string} environment - Environment (testnet/mainnet)
 * @returns {Promise<ApiKeyData|null>} Default API key or null if not found
 */
async function getDefaultApiKey(
  userId: string,
  environment: string = 'testnet'
): Promise<ApiKeyData | null> {
  // Find the default key in the database
  const defaultApiKey = await prisma.apiKey.findFirst({
    where: {
      userId,
      environment,
      isDefault: true,
      isRevoked: false,
      expiry: { gt: new Date() }
    }
  });

  if (!defaultApiKey) {
    return null;
  }

  // Get the full API key with sensitive data
  return getApiKey(userId, defaultApiKey.id, environment);
}

/**
 * Set an API key as the default for a user
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID to set as default
 * @returns {Promise<boolean>} Success indicator
 */
async function setDefaultApiKey(userId: string, keyId: string): Promise<boolean> {
  // Get the key to verify it exists and belongs to the user
  const apiKey = await prisma.apiKey.findFirst({
    where: {
      id: keyId,
      userId,
      isRevoked: false,
      expiry: { gt: new Date() }
    }
  });

  if (!apiKey) {
    throw new Error('API key not found or not valid');
  }

  // Start a transaction to update default flags
  await prisma.$transaction([
    // Unset default flag on all other keys for this user in the same environment
    prisma.apiKey.updateMany({
      where: {
        userId,
        environment: apiKey.environment,
        isDefault: true,
        id: { not: keyId }
      },
      data: {
        isDefault: false
      }
    }),
    // Set default flag on the specified key
    prisma.apiKey.update({
      where: {
        id: keyId
      },
      data: {
        isDefault: true
      }
    })
  ]);

  // Log the default key change
  await createAuditLog({
    userId,
    action: 'api_key.set_default',
    details: {
      keyId,
      environment: apiKey.environment
    }
  });

  return true;
}

/**
 * List API keys for a user (without sensitive data)
 * @param {string} userId - User ID
 * @param {object} filters - Filters for the keys
 * @returns {Promise<ApiKeySummary[]>} Array of API key summaries
 */
async function listApiKeys(
  userId: string,
  filters: { environment?: string; includeRevoked?: boolean } = {}
): Promise<ApiKeySummary[]> {
  const apiKeys = await apiKeyService.listApiKeys(
    userId,
    {
      environment: filters.environment,
      isRevoked: filters.includeRevoked ? undefined : false
    }
  );

  return apiKeys.map(key => ({
    id: key.id,
    userId: key.userId,
    name: key.name,
    maskedKey: key.maskedKey || '',
    scopes: key.scopes,
    environment: key.environment,
    expiry: key.expiry,
    isDefault: key.isDefault || false,
    isRevoked: key.isRevoked,
    createdAt: key.createdAt,
    lastUsedAt: key.lastUsedAt,
    usageCount: key.usageCount || 0,
    isExpired: key.isExpired || key.expiry < new Date()
  }));
}

/**
 * Revoke an API key
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID
 * @param {string} revokedBy - ID of user who revoked the key
 * @param {string} reason - Reason for revocation
 * @returns {Promise<boolean>} Success indicator
 */
async function revokeApiKey(
  userId: string,
  keyId: string,
  revokedBy: string,
  reason: string = 'User requested'
): Promise<boolean> {
  const result = await apiKeyService.revokeApiKey(keyId, userId, revokedBy, reason);

  if (result) {
    // Log the API key revocation
    await createAuditLog({
      userId,
      action: 'api_key.revoke',
      details: {
        keyId,
        revokedBy,
        reason
      }
    });
  }

  return result;
}

/**
 * Rotate an API key (replace with new key and revoke old one)
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID to rotate
 * @param {string} newApiKey - New API key
 * @param {string} newApiSecret - New API secret
 * @param {boolean} validateKey - Whether to validate new key
 * @returns {Promise<ApiKeySummary>} The new API key summary
 */
async function rotateApiKey(
  userId: string,
  keyId: string,
  newApiKey: string,
  newApiSecret: string,
  validateKey: boolean = true
): Promise<ApiKeySummary> {
  // First, get the original key to get its settings
  const oldKey = await prisma.apiKey.findFirst({
    where: {
      id: keyId,
      userId,
      isRevoked: false
    }
  });

  if (!oldKey) {
    throw new Error('API key not found or already revoked');
  }

  // Validate the new key
  if (validateKey) {
    const validationResult = await apiKeyValidation.validateDeltaApiKey(
      newApiKey,
      newApiSecret,
      oldKey.environment === 'testnet'
    );

    if (!validationResult.isValid) {
      throw new Error(`New API key validation failed: ${validationResult.message}`);
    }
    
    // Also validate trading permissions
    const tradingPermissions = await apiKeyValidation.validateTradingPermissions(
      newApiKey,
      newApiSecret,
      oldKey.environment === 'testnet'
    );
    
    if (!tradingPermissions.isValid) {
      throw new Error(`New API key trading permission validation failed: ${tradingPermissions.message}`);
    }
  }

  // Create a new key with the same settings
  const newKeyRecord = await apiKeyService.rotateApiKey(
    keyId,
    userId,
    newApiKey,
    newApiSecret
  );

  // If the old key was default, make the new one default too
  if (oldKey.isDefault) {
    await setDefaultApiKey(userId, newKeyRecord.id);
  }

  // Log the key rotation
  await createAuditLog({
    userId,
    action: 'api_key.rotate',
    details: {
      oldKeyId: keyId,
      newKeyId: newKeyRecord.id,
      environment: oldKey.environment
    }
  });

  return {
    id: newKeyRecord.id,
    userId: newKeyRecord.userId,
    name: newKeyRecord.name,
    maskedKey: apiKeyEncryption.maskApiKey(newApiKey),
    scopes: newKeyRecord.scopes,
    environment: newKeyRecord.environment,
    expiry: newKeyRecord.expiry,
    isDefault: oldKey.isDefault,
    isRevoked: false,
    createdAt: newKeyRecord.createdAt,
    usageCount: 0,
    isExpired: false
  };
}

/**
 * Update API key settings
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID
 * @param {object} updates - Settings to update
 * @returns {Promise<ApiKeySummary>} Updated API key summary
 */
async function updateApiKey(
  userId: string,
  keyId: string,
  updates: {
    name?: string;
    scopes?: string[];
    expiry?: Date;
    ipRestrictions?: string[];
    isDefault?: boolean;
    rateLimits?: any;
    metadata?: any;
  }
): Promise<ApiKeySummary> {
  // Update the key settings
  const updatedKey = await apiKeyService.updateApiKey(
    keyId,
    userId,
    updates
  );

  // Handle default flag
  if (updates.isDefault) {
    await setDefaultApiKey(userId, keyId);
  }

  // Log the key update
  await createAuditLog({
    userId,
    action: 'api_key.update',
    details: {
      keyId,
      updates: {
        ...updates,
        // Don't log potentially sensitive metadata
        metadata: updates.metadata ? '[redacted]' : undefined
      }
    }
  });

  return {
    id: updatedKey.id,
    userId: updatedKey.userId,
    name: updatedKey.name,
    maskedKey: updatedKey.maskedKey || '',
    scopes: updatedKey.scopes,
    environment: updatedKey.environment,
    expiry: updatedKey.expiry,
    isDefault: updates.isDefault || false,
    isRevoked: updatedKey.isRevoked,
    createdAt: updatedKey.createdAt,
    lastUsedAt: updatedKey.lastUsedAt,
    usageCount: updatedKey.usageCount || 0,
    isExpired: updatedKey.isExpired || updatedKey.expiry < new Date()
  };
}

/**
 * Validate an API key against Delta Exchange
 * @param {string} apiKey - API key to validate
 * @param {string} apiSecret - API secret to validate
 * @param {string} environment - Environment (testnet/mainnet)
 * @returns {Promise<apiKeyValidation.ValidationResult>} Validation result
 */
async function validateApiKey(
  apiKey: string,
  apiSecret: string,
  environment: string = 'testnet'
): Promise<apiKeyValidation.ValidationResult> {
  return apiKeyValidation.validateDeltaApiKey(
    apiKey,
    apiSecret,
    environment === 'testnet'
  );
}

/**
 * Get API key usage statistics
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID
 * @returns {Promise<any>} Usage statistics
 */
async function getApiKeyUsageStats(
  userId: string,
  keyId: string
): Promise<any> {
  // Get basic key usage info
  const keyData = await prisma.apiKey.findFirst({
    where: {
      id: keyId,
      userId
    },
    select: {
      id: true,
      usageCount: true,
      lastUsedAt: true,
      createdAt: true,
      environment: true
    }
  });

  if (!keyData) {
    throw new Error('API key not found');
  }

  // Get audit logs for this key
  const auditLogs = await prisma.auditLog.findMany({
    where: {
      userId,
      action: {
        startsWith: 'api_key.'
      },
      details: {
        path: ['keyId'],
        equals: keyId
      }
    },
    orderBy: {
      timestamp: 'desc'
    },
    take: 50
  });

  // Get API call stats from rate limiting/analytics system
  // This is just a placeholder - you would integrate with your actual
  // analytics system here

  return {
    basicStats: {
      usageCount: keyData.usageCount || 0,
      lastUsed: keyData.lastUsedAt,
      createdAt: keyData.createdAt,
      ageInDays: Math.floor((Date.now() - keyData.createdAt.getTime()) / (1000 * 60 * 60 * 24))
    },
    recentActivity: auditLogs.map(log => ({
      action: log.action,
      timestamp: log.timestamp,
      details: log.details
    })),
    // This would be populated from your analytics system
    apiCalls: {
      totalCalls: keyData.usageCount || 0,
      successRate: 100,
      averageResponseTime: 250,
      errorRate: 0
    }
  };
}

export {
  ApiKeyData,
  ApiKeySummary,
  addApiKey,
  getApiKey,
  getDefaultApiKey,
  setDefaultApiKey,
  listApiKeys,
  revokeApiKey,
  rotateApiKey,
  updateApiKey,
  validateApiKey,
  getApiKeyUsageStats
}; 