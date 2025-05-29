/**
 * Enhanced API Key Management Service
 * Handles encryption, storage, and management of Delta Exchange API keys
 */

import prisma from '../utils/prismaClient';
import * as encryption from '../utils/encryption';
import * as secureKey from '../utils/secureKey';
import crypto from 'crypto';

/**
 * API Key record interface
 */
interface ApiKeyRecord {
  id: string;
  userId: string;
  name: string;
  maskedKey?: string;
  key?: string;
  secret?: string;
  scopes: string[];
  expiry: Date;
  environment: string;
  createdAt: Date;
  lastUsedAt?: Date;
  isRevoked: boolean;
  revokedAt?: Date;
  revokedBy?: string;
  revokedReason?: string;
  usageCount: number;
  ipRestrictions?: string[];
  rateLimits?: any;
  metadata?: any;
  isExpired?: boolean;
}

/**
 * Encrypted data structure
 */
interface EncryptedData {
  key: encryption.EncryptedData;
  secret: encryption.EncryptedData;
}

/**
 * API Key creation options
 */
interface ApiKeyOptions {
  name?: string;
  scopes?: string[];
  expiry?: Date;
  environment?: string;
  ipRestrictions?: string[];
  rateLimits?: any;
  metadata?: any;
}

/**
 * Adds a new API key for a user
 * @param {string} userId - The ID of the user
 * @param {string} apiKey - The API key to store
 * @param {string} apiSecret - The API secret to store
 * @param {ApiKeyOptions} options - Options for the API key
 * @returns {Promise<ApiKeyRecord>} The created API key record (without sensitive data)
 */
async function addApiKey(
  userId: string, 
  apiKey: string, 
  apiSecret: string, 
  options: ApiKeyOptions = {}
): Promise<ApiKeyRecord> {
  if (!userId || !apiKey || !apiSecret) {
    throw new Error('User ID, API key, and secret are required');
  }

  // Validate inputs
  if (!validateApiKeyFormat(apiKey) || !validateApiSecretFormat(apiSecret)) {
    throw new Error('Invalid API key or secret format');
  }

  // Validate if user exists
  const user = await prisma.user.findUnique({
    where: { id: userId }
  });

  if (!user) {
    throw new Error(`User with ID ${userId} not found`);
  }

  // Get the master encryption key
  const masterKey = secureKey.getMasterKey();
  
  // Derive a key specific for this user's API keys
  const derivedKey = secureKey.deriveKey(masterKey, 'api-keys', userId);

  // Encrypt the API key and secret
  const encryptedKey = encryption.encrypt(apiKey, derivedKey);
  const encryptedSecret = encryption.encrypt(apiSecret, derivedKey);

  // Generate a hash of the secret for validation purposes
  const hashedSecret = crypto
    .createHash('sha256')
    .update(`${apiSecret}_${userId}`)
    .digest('hex');

  // Format scopes as a comma-separated string
  const scopesStr = Array.isArray(options.scopes) 
    ? options.scopes.join(',') 
    : (options.scopes || 'read,trade');

  // Format IP restrictions
  const ipRestrictionsStr = Array.isArray(options.ipRestrictions)
    ? options.ipRestrictions.join(',')
    : options.ipRestrictions;

  // Store the encrypted data in the database
  const apiKeyRecord = await prisma.apiKey.create({
    data: {
      userId,
      name: options.name || 'Default',
      // Store a masked version of the key as a unique identifier
      key: `${maskApiKey(apiKey)}_${encryptedKey.iv}`,
      // Store the encrypted data in JSON format
      encryptedData: JSON.stringify({
        key: encryptedKey,
        secret: encryptedSecret
      }),
      scopes: scopesStr,
      expiry: options.expiry || new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // Default 30 days
      environment: options.environment || 'testnet',
      hashedSecret,
      ipRestrictions: ipRestrictionsStr,
      rateLimits: options.rateLimits ? JSON.stringify(options.rateLimits) : undefined,
      metadata: options.metadata ? JSON.stringify(options.metadata) : undefined
    }
  });

  return {
    id: apiKeyRecord.id,
    userId: apiKeyRecord.userId,
    name: apiKeyRecord.name,
    maskedKey: maskApiKey(apiKey),
    scopes: apiKeyRecord.scopes.split(','),
    expiry: apiKeyRecord.expiry,
    environment: apiKeyRecord.environment,
    createdAt: apiKeyRecord.createdAt,
    isRevoked: apiKeyRecord.isRevoked,
    usageCount: apiKeyRecord.usageCount,
    ipRestrictions: apiKeyRecord.ipRestrictions?.split(','),
    rateLimits: apiKeyRecord.rateLimits ? JSON.parse(apiKeyRecord.rateLimits as string) : undefined,
    metadata: apiKeyRecord.metadata ? JSON.parse(apiKeyRecord.metadata as string) : undefined
  };
}

/**
 * Retrieves a user's API key and secret
 * @param {string} userId - The ID of the user
 * @param {string} keyId - Optional specific key ID to retrieve
 * @param {string} environment - Optional environment filter (testnet/mainnet)
 * @returns {Promise<ApiKeyRecord|null>} The decrypted API key and secret or null if not found
 */
async function getApiKey(
  userId: string, 
  keyId?: string,
  environment?: string
): Promise<ApiKeyRecord | null> {
  if (!userId) {
    throw new Error('User ID is required');
  }

  // Build the query conditions
  const whereCondition: any = { 
    userId,
    isRevoked: false,
    expiry: { gt: new Date() } // Only retrieve non-expired keys
  };

  // Add key ID filter if provided
  if (keyId) {
    whereCondition.id = keyId;
  }

  // Add environment filter if provided
  if (environment) {
    whereCondition.environment = environment;
  }

  // Find the user's API key in the database
  const apiKeyRecord = await prisma.apiKey.findFirst({
    where: whereCondition,
    orderBy: { createdAt: 'desc' } // Get the most recent key
  });

  if (!apiKeyRecord) {
    return null;
  }

  try {
    // Get the master encryption key
    const masterKey = secureKey.getMasterKey();
    
    // Derive the key specific to this user's API keys
    const derivedKey = secureKey.deriveKey(masterKey, 'api-keys', userId);

    // Parse the encrypted data
    const encryptedData = JSON.parse(apiKeyRecord.encryptedData);
    
    // Decrypt the API key and secret
    const apiKey = encryption.decrypt(encryptedData.key, derivedKey);
    const apiSecret = encryption.decrypt(encryptedData.secret, derivedKey);

    // Update last used timestamp
    await prisma.apiKey.update({
      where: { id: apiKeyRecord.id },
      data: { 
        lastUsedAt: new Date(),
        usageCount: { increment: 1 }
      }
    });

    return {
      id: apiKeyRecord.id,
      userId: apiKeyRecord.userId,
      name: apiKeyRecord.name,
      key: typeof apiKey === 'string' ? apiKey : JSON.stringify(apiKey),
      secret: typeof apiSecret === 'string' ? apiSecret : JSON.stringify(apiSecret),
      scopes: apiKeyRecord.scopes.split(','),
      expiry: apiKeyRecord.expiry,
      environment: apiKeyRecord.environment,
      createdAt: apiKeyRecord.createdAt,
      lastUsedAt: apiKeyRecord.lastUsedAt || undefined,
      isRevoked: apiKeyRecord.isRevoked,
      usageCount: apiKeyRecord.usageCount,
      ipRestrictions: apiKeyRecord.ipRestrictions?.split(','),
      rateLimits: apiKeyRecord.rateLimits ? JSON.parse(apiKeyRecord.rateLimits as string) : undefined,
      metadata: apiKeyRecord.metadata ? JSON.parse(apiKeyRecord.metadata as string) : undefined
    };
  } catch (error) {
    console.error(`Error decrypting API key for user ${userId}:`, error);
    throw new Error('Failed to decrypt API key data');
  }
}

/**
 * Returns a list of user's API keys with masked values
 * @param {string} userId - The ID of the user
 * @param {object} filters - Optional filters like environment, isRevoked, etc.
 * @returns {Promise<ApiKeyRecord[]>} Array of API key records with masked sensitive data
 */
async function listApiKeys(
  userId: string,
  filters: { environment?: string; isRevoked?: boolean } = {}
): Promise<ApiKeyRecord[]> {
  if (!userId) {
    throw new Error('User ID is required');
  }

  // Build the query conditions
  const whereCondition: any = { userId };

  // Add environment filter if provided
  if (filters.environment) {
    whereCondition.environment = filters.environment;
  }

  // Add revoked status filter if provided
  if (filters.isRevoked !== undefined) {
    whereCondition.isRevoked = filters.isRevoked;
  }

  const apiKeys = await prisma.apiKey.findMany({
    where: whereCondition,
    orderBy: { createdAt: 'desc' }
  });

  return apiKeys.map(key => ({
    id: key.id,
    userId: key.userId,
    name: key.name,
    maskedKey: key.key.split('_')[0], // The masked version is stored before the underscore
    scopes: key.scopes.split(','),
    expiry: key.expiry,
    environment: key.environment,
    createdAt: key.createdAt,
    lastUsedAt: key.lastUsedAt || undefined,
    isRevoked: key.isRevoked,
    revokedAt: key.revokedAt || undefined,
    revokedBy: key.revokedBy || undefined,
    revokedReason: key.revokedReason || undefined,
    usageCount: key.usageCount,
    ipRestrictions: key.ipRestrictions?.split(','),
    rateLimits: key.rateLimits ? JSON.parse(key.rateLimits as string) : undefined,
    metadata: key.metadata ? JSON.parse(key.metadata as string) : undefined,
    isExpired: key.expiry < new Date()
  }));
}

/**
 * Revokes an API key
 * @param {string} keyId - The ID of the API key to revoke
 * @param {string} userId - The ID of the user (for verification)
 * @param {string} revokedBy - ID of the user who performed the revocation
 * @param {string} reason - Reason for revocation
 * @returns {Promise<boolean>} True if successful, false otherwise
 */
async function revokeApiKey(
  keyId: string, 
  userId: string,
  revokedBy: string,
  reason: string = 'User requested'
): Promise<boolean> {
  if (!keyId || !userId) {
    throw new Error('API key ID and user ID are required');
  }

  try {
    // Mark key as revoked instead of deleting it
    const result = await prisma.apiKey.updateMany({
      where: {
        id: keyId,
        userId
      },
      data: {
        isRevoked: true,
        revokedAt: new Date(),
        revokedBy,
        revokedReason: reason
      }
    });
    
    return result.count > 0;
  } catch (error) {
    console.error(`Error revoking API key ${keyId}:`, error);
    return false;
  }
}

/**
 * Rotates an API key (creates a new one and revokes the old one)
 * @param {string} keyId - The ID of the API key to rotate
 * @param {string} userId - The ID of the user
 * @param {string} newApiKey - The new API key
 * @param {string} newApiSecret - The new API secret
 * @returns {Promise<ApiKeyRecord>} The new API key record
 */
async function rotateApiKey(
  keyId: string,
  userId: string,
  newApiKey: string,
  newApiSecret: string
): Promise<ApiKeyRecord> {
  // Get the old key to copy its settings
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

  // Create the new key with the same settings
  const newKey = await addApiKey(userId, newApiKey, newApiSecret, {
    name: `${oldKey.name} (rotated)`,
    scopes: oldKey.scopes.split(','),
    expiry: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // Default 30 days
    environment: oldKey.environment,
    ipRestrictions: oldKey.ipRestrictions?.split(','),
    rateLimits: oldKey.rateLimits ? JSON.parse(oldKey.rateLimits as string) : undefined,
    metadata: oldKey.metadata ? JSON.parse(oldKey.metadata as string) : undefined
  });

  // Revoke the old key
  await revokeApiKey(keyId, userId, userId, 'Key rotation');

  return newKey;
}

/**
 * Update API key settings
 * @param {string} keyId - The ID of the API key to update
 * @param {string} userId - The ID of the user
 * @param {object} updates - The fields to update
 * @returns {Promise<ApiKeyRecord>} The updated API key record
 */
async function updateApiKey(
  keyId: string,
  userId: string,
  updates: {
    name?: string;
    scopes?: string[];
    expiry?: Date;
    environment?: string;
    ipRestrictions?: string[];
    rateLimits?: any;
    metadata?: any;
  }
): Promise<ApiKeyRecord> {
  // Validate the key exists and belongs to the user
  const existingKey = await prisma.apiKey.findFirst({
    where: {
      id: keyId,
      userId,
      isRevoked: false
    }
  });

  if (!existingKey) {
    throw new Error('API key not found or already revoked');
  }

  // Prepare update data
  const updateData: any = {};

  if (updates.name) updateData.name = updates.name;
  if (updates.scopes) updateData.scopes = updates.scopes.join(',');
  if (updates.expiry) updateData.expiry = updates.expiry;
  if (updates.environment) updateData.environment = updates.environment;
  if (updates.ipRestrictions) updateData.ipRestrictions = updates.ipRestrictions.join(',');
  if (updates.rateLimits) updateData.rateLimits = JSON.stringify(updates.rateLimits);
  if (updates.metadata) updateData.metadata = JSON.stringify(updates.metadata);

  // Update the key
  const updatedKey = await prisma.apiKey.update({
    where: { id: keyId },
    data: updateData
  });

  // Return formatted record
  return {
    id: updatedKey.id,
    userId: updatedKey.userId,
    name: updatedKey.name,
    maskedKey: updatedKey.key.split('_')[0],
    scopes: updatedKey.scopes.split(','),
    expiry: updatedKey.expiry,
    environment: updatedKey.environment,
    createdAt: updatedKey.createdAt,
    lastUsedAt: updatedKey.lastUsedAt || undefined,
    isRevoked: updatedKey.isRevoked,
    usageCount: updatedKey.usageCount,
    ipRestrictions: updatedKey.ipRestrictions?.split(','),
    rateLimits: updatedKey.rateLimits ? JSON.parse(updatedKey.rateLimits as string) : undefined,
    metadata: updatedKey.metadata ? JSON.parse(updatedKey.metadata as string) : undefined
  };
}

/**
 * Validate if an API key is allowed from a specific IP address
 * @param {string} keyId - The ID of the API key
 * @param {string} ip - The IP address to check
 * @returns {Promise<boolean>} True if allowed, false otherwise
 */
async function validateIpAccess(keyId: string, ip: string): Promise<boolean> {
  const apiKey = await prisma.apiKey.findUnique({
    where: { id: keyId }
  });

  if (!apiKey || apiKey.isRevoked || apiKey.expiry < new Date()) {
    return false;
  }

  // If no IP restrictions, allow access
  if (!apiKey.ipRestrictions) {
    return true;
  }

  // Check if the IP is in the allowed list
  const allowedIps = apiKey.ipRestrictions.split(',');
  return allowedIps.includes(ip);
}

/**
 * Masks an API key for display purposes
 * @param {string} apiKey - The API key to mask
 * @returns {string} Masked version of the key
 */
function maskApiKey(apiKey: string): string {
  if (!apiKey || apiKey.length < 8) {
    return '****';
  }
  
  // Show first 4 and last 4 characters
  return `${apiKey.substring(0, 4)}****${apiKey.substring(apiKey.length - 4)}`;
}

/**
 * Validates the format of a Delta Exchange API key
 * @param {string} apiKey - The API key to validate
 * @returns {boolean} True if valid format, false otherwise
 */
function validateApiKeyFormat(apiKey: string): boolean {
  // Basic validation: Check length and character set
  // Adjust this based on Delta Exchange API key format
  if (!apiKey || typeof apiKey !== 'string') {
    return false;
  }
  
  // Typical format: alphanumeric, possibly with hyphens, 32+ chars
  return /^[a-zA-Z0-9-]{32,}$/.test(apiKey);
}

/**
 * Validates the format of a Delta Exchange API secret
 * @param {string} apiSecret - The API secret to validate
 * @returns {boolean} True if valid format, false otherwise
 */
function validateApiSecretFormat(apiSecret: string): boolean {
  // Basic validation: Check length and character set
  // Adjust this based on Delta Exchange API secret format
  if (!apiSecret || typeof apiSecret !== 'string') {
    return false;
  }
  
  // Typical format: alphanumeric, possibly with special chars, 64+ chars
  return /^[a-zA-Z0-9_-]{64,}$/.test(apiSecret);
}

export {
  addApiKey,
  getApiKey,
  listApiKeys,
  revokeApiKey,
  rotateApiKey,
  updateApiKey,
  validateIpAccess,
  maskApiKey,
  validateApiKeyFormat,
  validateApiSecretFormat,
  ApiKeyRecord,
  ApiKeyOptions
}; 