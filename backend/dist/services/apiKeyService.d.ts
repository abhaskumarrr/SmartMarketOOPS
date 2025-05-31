/**
 * Enhanced API Key Management Service
 * Handles encryption, storage, and management of Delta Exchange API keys
 */
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
declare function addApiKey(userId: string, apiKey: string, apiSecret: string, options?: ApiKeyOptions): Promise<ApiKeyRecord>;
/**
 * Retrieves a user's API key and secret
 * @param {string} userId - The ID of the user
 * @param {string} keyId - Optional specific key ID to retrieve
 * @param {string} environment - Optional environment filter (testnet/mainnet)
 * @returns {Promise<ApiKeyRecord|null>} The decrypted API key and secret or null if not found
 */
declare function getApiKey(userId: string, keyId?: string, environment?: string): Promise<ApiKeyRecord | null>;
/**
 * Returns a list of user's API keys with masked values
 * @param {string} userId - The ID of the user
 * @param {object} filters - Optional filters like environment, isRevoked, etc.
 * @returns {Promise<ApiKeyRecord[]>} Array of API key records with masked sensitive data
 */
declare function listApiKeys(userId: string, filters?: {
    environment?: string;
    isRevoked?: boolean;
}): Promise<ApiKeyRecord[]>;
/**
 * Revokes an API key
 * @param {string} keyId - The ID of the API key to revoke
 * @param {string} userId - The ID of the user (for verification)
 * @param {string} revokedBy - ID of the user who performed the revocation
 * @param {string} reason - Reason for revocation
 * @returns {Promise<boolean>} True if successful, false otherwise
 */
declare function revokeApiKey(keyId: string, userId: string, revokedBy: string, reason?: string): Promise<boolean>;
/**
 * Rotates an API key (creates a new one and revokes the old one)
 * @param {string} keyId - The ID of the API key to rotate
 * @param {string} userId - The ID of the user
 * @param {string} newApiKey - The new API key
 * @param {string} newApiSecret - The new API secret
 * @returns {Promise<ApiKeyRecord>} The new API key record
 */
declare function rotateApiKey(keyId: string, userId: string, newApiKey: string, newApiSecret: string): Promise<ApiKeyRecord>;
/**
 * Update API key settings
 * @param {string} keyId - The ID of the API key to update
 * @param {string} userId - The ID of the user
 * @param {object} updates - The fields to update
 * @returns {Promise<ApiKeyRecord>} The updated API key record
 */
declare function updateApiKey(keyId: string, userId: string, updates: {
    name?: string;
    scopes?: string[];
    expiry?: Date;
    environment?: string;
    ipRestrictions?: string[];
    rateLimits?: any;
    metadata?: any;
}): Promise<ApiKeyRecord>;
/**
 * Validate if an API key is allowed from a specific IP address
 * @param {string} keyId - The ID of the API key
 * @param {string} ip - The IP address to check
 * @returns {Promise<boolean>} True if allowed, false otherwise
 */
declare function validateIpAccess(keyId: string, ip: string): Promise<boolean>;
/**
 * Masks an API key for display purposes
 * @param {string} apiKey - The API key to mask
 * @returns {string} Masked version of the key
 */
declare function maskApiKey(apiKey: string): string;
/**
 * Validates the format of a Delta Exchange API key
 * @param {string} apiKey - The API key to validate
 * @returns {boolean} True if valid format, false otherwise
 */
declare function validateApiKeyFormat(apiKey: string): boolean;
/**
 * Validates the format of a Delta Exchange API secret
 * @param {string} apiSecret - The API secret to validate
 * @returns {boolean} True if valid format, false otherwise
 */
declare function validateApiSecretFormat(apiSecret: string): boolean;
export { addApiKey, getApiKey, listApiKeys, revokeApiKey, rotateApiKey, updateApiKey, validateIpAccess, maskApiKey, validateApiKeyFormat, validateApiSecretFormat, ApiKeyRecord, ApiKeyOptions };
