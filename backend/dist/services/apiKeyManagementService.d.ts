/**
 * API Key Management Service
 * Comprehensive service for managing API keys with security, validation, and analytics
 */
import * as apiKeyValidation from './trading/apiKeyValidationService';
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
declare function addApiKey(apiKeyData: ApiKeyData, validateKey?: boolean): Promise<ApiKeySummary>;
/**
 * Get API key and secret for a user (sensitive data)
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID
 * @param {string} environment - Environment (testnet/mainnet)
 * @returns {Promise<ApiKeyData|null>} API key data or null if not found
 */
declare function getApiKey(userId: string, keyId?: string, environment?: string): Promise<ApiKeyData | null>;
/**
 * Get user's default API key (for automatic trading)
 * @param {string} userId - User ID
 * @param {string} environment - Environment (testnet/mainnet)
 * @returns {Promise<ApiKeyData|null>} Default API key or null if not found
 */
declare function getDefaultApiKey(userId: string, environment?: string): Promise<ApiKeyData | null>;
/**
 * Set an API key as the default for a user
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID to set as default
 * @returns {Promise<boolean>} Success indicator
 */
declare function setDefaultApiKey(userId: string, keyId: string): Promise<boolean>;
/**
 * List API keys for a user (without sensitive data)
 * @param {string} userId - User ID
 * @param {object} filters - Filters for the keys
 * @returns {Promise<ApiKeySummary[]>} Array of API key summaries
 */
declare function listApiKeys(userId: string, filters?: {
    environment?: string;
    includeRevoked?: boolean;
}): Promise<ApiKeySummary[]>;
/**
 * Revoke an API key
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID
 * @param {string} revokedBy - ID of user who revoked the key
 * @param {string} reason - Reason for revocation
 * @returns {Promise<boolean>} Success indicator
 */
declare function revokeApiKey(userId: string, keyId: string, revokedBy: string, reason?: string): Promise<boolean>;
/**
 * Rotate an API key (replace with new key and revoke old one)
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID to rotate
 * @param {string} newApiKey - New API key
 * @param {string} newApiSecret - New API secret
 * @param {boolean} validateKey - Whether to validate new key
 * @returns {Promise<ApiKeySummary>} The new API key summary
 */
declare function rotateApiKey(userId: string, keyId: string, newApiKey: string, newApiSecret: string, validateKey?: boolean): Promise<ApiKeySummary>;
/**
 * Update API key settings
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID
 * @param {object} updates - Settings to update
 * @returns {Promise<ApiKeySummary>} Updated API key summary
 */
declare function updateApiKey(userId: string, keyId: string, updates: {
    name?: string;
    scopes?: string[];
    expiry?: Date;
    ipRestrictions?: string[];
    isDefault?: boolean;
    rateLimits?: any;
    metadata?: any;
}): Promise<ApiKeySummary>;
/**
 * Validate an API key against Delta Exchange
 * @param {string} apiKey - API key to validate
 * @param {string} apiSecret - API secret to validate
 * @param {string} environment - Environment (testnet/mainnet)
 * @returns {Promise<apiKeyValidation.ValidationResult>} Validation result
 */
declare function validateApiKey(apiKey: string, apiSecret: string, environment?: string): Promise<apiKeyValidation.ValidationResult>;
/**
 * Get API key usage statistics
 * @param {string} userId - User ID
 * @param {string} keyId - API key ID
 * @returns {Promise<any>} Usage statistics
 */
declare function getApiKeyUsageStats(userId: string, keyId: string): Promise<any>;
export { ApiKeyData, ApiKeySummary, addApiKey, getApiKey, getDefaultApiKey, setDefaultApiKey, listApiKeys, revokeApiKey, rotateApiKey, updateApiKey, validateApiKey, getApiKeyUsageStats };
