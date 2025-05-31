/**
 * Enhanced Encryption Utilities
 * Functions for encrypting and decrypting sensitive data with key rotation support
 */
declare const KEY_SIZE = 32;
declare const IV_SIZE = 16;
declare const VERSION_IDENTIFIER: {
    V1: string;
    V2: string;
};
/**
 * Options for encryption
 */
export interface EncryptionOptions {
    useSalt?: boolean;
}
/**
 * Structure of encrypted data
 */
export interface EncryptedData {
    version: string;
    iv: string;
    salt?: string;
    data: string;
    authTag: string;
}
/**
 * Encrypt data using AES-256-GCM with enhanced security
 * @param {string|Object} data - Data to encrypt
 * @param {Buffer} key - Key to use for encryption
 * @param {EncryptionOptions} options - Encryption options
 * @returns {EncryptedData} - Encrypted data object
 */
export declare function encrypt(data: string | object, key?: Buffer, options?: EncryptionOptions): EncryptedData;
/**
 * Decrypt data using AES-256-GCM with support for different versions and key rotation
 * @param {EncryptedData} encryptedData - Encrypted data object
 * @param {Buffer} key - Key to use for decryption
 * @returns {string|object} - Decrypted data, parsed as JSON if applicable
 */
export declare function decrypt(encryptedData: EncryptedData, key?: Buffer): string | object;
/**
 * Legacy decrypt function that accepts string format
 * @param {string} encryptedString - Encrypted data as string
 * @returns {string|object} - Decrypted data
 */
export declare function decryptFromString(encryptedString: string): string | object;
/**
 * Convert encrypted data object to string format
 * @param {EncryptedData} encryptedData - Encrypted data object
 * @returns {string} - String representation
 */
export declare function encryptedDataToString(encryptedData: EncryptedData): string;
/**
 * Re-encrypt data using the current primary key
 * Useful for key rotation when data was encrypted with an old key
 * @param {EncryptedData} encryptedData - Data encrypted with old key
 * @returns {EncryptedData} - Data re-encrypted with new primary key
 */
export declare function reEncrypt(encryptedData: EncryptedData): EncryptedData;
/**
 * Generates a secure random encryption key
 * @returns {string} - Hex-encoded encryption key
 */
export declare function generateSecureKey(): string;
export declare function encryptToString(data: string | object, key?: Buffer, options?: EncryptionOptions): string;
export { KEY_SIZE, IV_SIZE, VERSION_IDENTIFIER };
