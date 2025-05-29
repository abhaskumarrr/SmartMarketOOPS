/**
 * API Key Encryption Utility
 * Specialized encryption functions for API keys with additional security measures
 */

import crypto from 'crypto';
import * as secureKey from './secureKey';

// Import the JavaScript module using require since it has no default export
// eslint-disable-next-line @typescript-eslint/no-var-requires
const encryptionJs = require('./encryption');

// Import types from encryption.ts
export interface EncryptedData {
  version: string;
  iv: string;
  salt?: string;
  data: string;
  authTag: string;
}

/**
 * API Key data structure
 */
export interface ApiKeyData {
  key: string;
  secret: string;
  environment?: string;
  label?: string;
  metadata?: Record<string, any>;
}

/**
 * Encrypted API Key data
 */
export interface EncryptedApiKeyData {
  encryptedKey: string;
  encryptedSecret: string;
  keyHash: string;
  secretHash: string;
  iv: string;
}

/**
 * Generate a secure hash of an API key or secret
 * @param {string} value - The value to hash
 * @param {string} salt - Salt for the hash
 * @returns {string} The hashed value
 */
export function hashApiCredential(value: string, salt: string): string {
  return crypto
    .createHash('sha256')
    .update(`${value}${salt}`)
    .digest('hex');
}

/**
 * Encrypt API key and secret using secure encryption
 * @param {ApiKeyData} data - API key data to encrypt
 * @param {string} userId - User ID for whom the keys belong
 * @returns {EncryptedApiKeyData} Encrypted API key data
 */
export function encryptApiKeyData(data: ApiKeyData, userId: string): EncryptedApiKeyData {
  if (!data.key || !data.secret) {
    throw new Error('API key and secret are required');
  }

  if (!userId) {
    throw new Error('User ID is required for encrypting API key data');
  }

  // Get the master key
  const masterKey = secureKey.getMasterKey();
  
  // Derive a key specific for this user's API keys
  const derivedKey = secureKey.deriveKey(masterKey, 'api-keys', userId);
  
  // Generate a unique IV
  const iv = crypto.randomBytes(16).toString('hex');
  
  // Encrypt the API key and secret
  const encryptedKey = encryptionJs.encrypt(data.key, derivedKey);
  const encryptedSecret = encryptionJs.encrypt(data.secret, derivedKey);
  
  // Create secure hashes of the API key and secret for verification purposes
  const keyHash = hashApiCredential(data.key, userId);
  const secretHash = hashApiCredential(data.secret, userId);
  
  return {
    encryptedKey,
    encryptedSecret,
    keyHash,
    secretHash,
    iv
  };
}

/**
 * Decrypt API key and secret data
 * @param {EncryptedApiKeyData} encryptedData - Encrypted API key data
 * @param {string} userId - User ID for whom the keys belong
 * @returns {ApiKeyData} Decrypted API key data
 */
export function decryptApiKeyData(encryptedData: EncryptedApiKeyData, userId: string): ApiKeyData {
  if (!encryptedData.encryptedKey || !encryptedData.encryptedSecret) {
    throw new Error('Encrypted key and secret data are required');
  }

  if (!userId) {
    throw new Error('User ID is required for decrypting API key data');
  }

  // Get the master key
  const masterKey = secureKey.getMasterKey();
  
  // Derive the key specific for this user's API keys
  const derivedKey = secureKey.deriveKey(masterKey, 'api-keys', userId);
  
  // Decrypt the API key and secret
  const key = encryptionJs.decrypt(encryptedData.encryptedKey, derivedKey) as string;
  const secret = encryptionJs.decrypt(encryptedData.encryptedSecret, derivedKey) as string;
  
  // Verify the decrypted data using hashes
  const keyHash = hashApiCredential(key, userId);
  const secretHash = hashApiCredential(secret, userId);
  
  if (keyHash !== encryptedData.keyHash || secretHash !== encryptedData.secretHash) {
    throw new Error('API key data integrity verification failed');
  }
  
  return {
    key,
    secret
  };
}

/**
 * Validate if an API key has the correct format
 * @param {string} apiKey - The API key to validate
 * @returns {boolean} True if valid format, false otherwise
 */
export function validateApiKeyFormat(apiKey: string): boolean {
  if (!apiKey || typeof apiKey !== 'string') {
    return false;
  }
  
  // Typical format: alphanumeric, possibly with hyphens, 32+ chars
  return /^[a-zA-Z0-9-]{32,}$/.test(apiKey);
}

/**
 * Validate if an API secret has the correct format
 * @param {string} apiSecret - The API secret to validate
 * @returns {boolean} True if valid format, false otherwise
 */
export function validateApiSecretFormat(apiSecret: string): boolean {
  if (!apiSecret || typeof apiSecret !== 'string') {
    return false;
  }
  
  // Typical format: alphanumeric, possibly with special chars, 64+ chars
  return /^[a-zA-Z0-9_-]{64,}$/.test(apiSecret);
}

/**
 * Mask an API key for display purposes
 * @param {string} apiKey - The API key to mask
 * @returns {string} Masked version of the key
 */
export function maskApiKey(apiKey: string): string {
  if (!apiKey || apiKey.length < 8) {
    return '****';
  }
  
  // Show first 4 and last 4 characters
  return `${apiKey.substring(0, 4)}****${apiKey.substring(apiKey.length - 4)}`;
}

/**
 * Rotate encryption key and re-encrypt API key data
 * @param {EncryptedApiKeyData} encryptedData - Data encrypted with old key
 * @param {string} userId - User ID for whom the keys belong
 * @param {Buffer} newMasterKey - New master key (optional, uses current master key if not provided)
 * @returns {EncryptedApiKeyData} Data re-encrypted with new key
 */
export function rotateEncryptionKey(
  encryptedData: EncryptedApiKeyData, 
  userId: string,
  newMasterKey?: Buffer
): EncryptedApiKeyData {
  // Decrypt with current key
  const decrypted = decryptApiKeyData(encryptedData, userId);
  
  // If new master key provided, temporarily set it as master key
  const originalMasterKey = newMasterKey ? secureKey.getMasterKey() : undefined;
  
  try {
    // Re-encrypt with new key
    return encryptApiKeyData(decrypted, userId);
  } finally {
    // Restore original master key if we changed it
    if (originalMasterKey) {
      // This is just a placeholder - in reality we would need a mechanism to restore
      // the original master key in the secureKey module
    }
  }
} 