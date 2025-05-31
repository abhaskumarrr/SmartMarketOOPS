/**
 * API Key Encryption Utility
 * Specialized encryption functions for API keys with additional security measures
 */
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
export declare function hashApiCredential(value: string, salt: string): string;
/**
 * Encrypt API key and secret using secure encryption
 * @param {ApiKeyData} data - API key data to encrypt
 * @param {string} userId - User ID for whom the keys belong
 * @returns {EncryptedApiKeyData} Encrypted API key data
 */
export declare function encryptApiKeyData(data: ApiKeyData, userId: string): EncryptedApiKeyData;
/**
 * Decrypt API key and secret data
 * @param {EncryptedApiKeyData} encryptedData - Encrypted API key data
 * @param {string} userId - User ID for whom the keys belong
 * @returns {ApiKeyData} Decrypted API key data
 */
export declare function decryptApiKeyData(encryptedData: EncryptedApiKeyData, userId: string): ApiKeyData;
/**
 * Validate if an API key has the correct format
 * @param {string} apiKey - The API key to validate
 * @returns {boolean} True if valid format, false otherwise
 */
export declare function validateApiKeyFormat(apiKey: string): boolean;
/**
 * Validate if an API secret has the correct format
 * @param {string} apiSecret - The API secret to validate
 * @returns {boolean} True if valid format, false otherwise
 */
export declare function validateApiSecretFormat(apiSecret: string): boolean;
/**
 * Mask an API key for display purposes
 * @param {string} apiKey - The API key to mask
 * @returns {string} Masked version of the key
 */
export declare function maskApiKey(apiKey: string): string;
/**
 * Rotate encryption key and re-encrypt API key data
 * @param {EncryptedApiKeyData} encryptedData - Data encrypted with old key
 * @param {string} userId - User ID for whom the keys belong
 * @param {Buffer} newMasterKey - New master key (optional, uses current master key if not provided)
 * @returns {EncryptedApiKeyData} Data re-encrypted with new key
 */
export declare function rotateEncryptionKey(encryptedData: EncryptedApiKeyData, userId: string, newMasterKey?: Buffer): EncryptedApiKeyData;
