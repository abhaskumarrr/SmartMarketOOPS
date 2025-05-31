"use strict";
/**
 * Enhanced Encryption Utilities
 * Functions for encrypting and decrypting sensitive data with key rotation support
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.VERSION_IDENTIFIER = exports.IV_SIZE = exports.KEY_SIZE = void 0;
exports.encrypt = encrypt;
exports.decrypt = decrypt;
exports.decryptFromString = decryptFromString;
exports.encryptedDataToString = encryptedDataToString;
exports.reEncrypt = reEncrypt;
exports.generateSecureKey = generateSecureKey;
exports.encryptToString = encryptToString;
const crypto_1 = __importDefault(require("crypto"));
// Ensure encryption key is set
if (!process.env.ENCRYPTION_KEY) {
    console.error('ENCRYPTION_KEY environment variable is required');
    process.exit(1);
}
// Convert environment variable to buffer - should be a 64 character hex string (32 bytes)
const primaryKey = Buffer.from(process.env.ENCRYPTION_KEY, 'hex');
// Optional secondary key for key rotation
const secondaryKey = process.env.ENCRYPTION_KEY_SECONDARY
    ? Buffer.from(process.env.ENCRYPTION_KEY_SECONDARY, 'hex')
    : null;
// Encryption algorithm
const ALGORITHM = 'aes-256-gcm';
const KEY_SIZE = 32; // 256 bits
exports.KEY_SIZE = KEY_SIZE;
const IV_SIZE = 16; // 128 bits
exports.IV_SIZE = IV_SIZE;
const VERSION_IDENTIFIER = {
    V1: '01', // Original version
    V2: '02' // Enhanced version with additional salt
};
exports.VERSION_IDENTIFIER = VERSION_IDENTIFIER;
/**
 * Encrypt data using AES-256-GCM with enhanced security
 * @param {string|Object} data - Data to encrypt
 * @param {Buffer} key - Key to use for encryption
 * @param {EncryptionOptions} options - Encryption options
 * @returns {EncryptedData} - Encrypted data object
 */
function encrypt(data, key = primaryKey, options = { useSalt: true }) {
    try {
        // Validate key size
        if (key.length !== KEY_SIZE) {
            throw new Error(`Encryption key must be ${KEY_SIZE * 2} characters (${KEY_SIZE} bytes)`);
        }
        // Convert object to string if necessary
        const dataString = typeof data === 'object' ? JSON.stringify(data) : data;
        // Generate random initialization vector
        const iv = crypto_1.default.randomBytes(IV_SIZE);
        // Add version identifier and optional salt for additional security
        const version = options.useSalt ? VERSION_IDENTIFIER.V2 : VERSION_IDENTIFIER.V1;
        // Generate random salt if using V2
        const salt = options.useSalt ? crypto_1.default.randomBytes(16) : Buffer.alloc(0);
        // Create cipher
        const cipher = crypto_1.default.createCipheriv(ALGORITHM, key, iv);
        // Encrypt data
        let encrypted = cipher.update(dataString, 'utf8', 'hex');
        encrypted += cipher.final('hex');
        // Get authentication tag
        const authTag = cipher.getAuthTag().toString('hex');
        // Create and return encrypted data object
        const result = {
            version,
            iv: iv.toString('hex'),
            data: encrypted,
            authTag
        };
        // Add salt if present
        if (options.useSalt) {
            result.salt = salt.toString('hex');
        }
        return result;
    }
    catch (error) {
        console.error('Encryption error:', error);
        throw new Error('Failed to encrypt data: ' + error.message);
    }
}
/**
 * Decrypt data using AES-256-GCM with support for different versions and key rotation
 * @param {EncryptedData} encryptedData - Encrypted data object
 * @param {Buffer} key - Key to use for decryption
 * @returns {string|object} - Decrypted data, parsed as JSON if applicable
 */
function decrypt(encryptedData, key = primaryKey) {
    try {
        // Convert hex strings to buffers
        const iv = Buffer.from(encryptedData.iv, 'hex');
        const authTag = Buffer.from(encryptedData.authTag, 'hex');
        // Create decipher
        const decipher = crypto_1.default.createDecipheriv(ALGORITHM, key, iv);
        decipher.setAuthTag(authTag);
        // Decrypt data
        let decrypted = decipher.update(encryptedData.data, 'hex', 'utf8');
        decrypted += decipher.final('utf8');
        // Try to parse as JSON if it looks like JSON
        try {
            if (decrypted.startsWith('{') || decrypted.startsWith('[')) {
                return JSON.parse(decrypted);
            }
        }
        catch (e) {
            // If parsing fails, return as string
        }
        return decrypted;
    }
    catch (error) {
        // If decryption failed with primary key and secondary key is available, try with secondary key
        if (secondaryKey) {
            try {
                // Create decipher with secondary key
                const iv = Buffer.from(encryptedData.iv, 'hex');
                const authTag = Buffer.from(encryptedData.authTag, 'hex');
                // Create decipher with secondary key
                const decipher = crypto_1.default.createDecipheriv(ALGORITHM, secondaryKey, iv);
                decipher.setAuthTag(authTag);
                // Decrypt data
                let decrypted = decipher.update(encryptedData.data, 'hex', 'utf8');
                decrypted += decipher.final('utf8');
                // Try to parse as JSON if it looks like JSON
                try {
                    if (decrypted.startsWith('{') || decrypted.startsWith('[')) {
                        return JSON.parse(decrypted);
                    }
                }
                catch (e) {
                    // If parsing fails, return as string
                }
                return decrypted;
            }
            catch (secondaryError) {
                console.error('Secondary key decryption error:', secondaryError);
                throw new Error('Failed to decrypt data with all available keys');
            }
        }
        console.error('Decryption error:', error);
        throw new Error('Failed to decrypt data');
    }
}
/**
 * Legacy decrypt function that accepts string format
 * @param {string} encryptedString - Encrypted data as string
 * @returns {string|object} - Decrypted data
 */
function decryptFromString(encryptedString) {
    try {
        // Parse the encrypted data components
        const parts = encryptedString.split(':');
        // Handle different versions
        const version = parts[0];
        // Create encrypted data object based on version
        const encryptedData = {
            version,
            iv: parts[1],
            data: '',
            authTag: ''
        };
        if (version === VERSION_IDENTIFIER.V1) {
            // Legacy format: version:iv:encrypted:authTag
            encryptedData.data = parts[2];
            encryptedData.authTag = parts[3];
        }
        else if (version === VERSION_IDENTIFIER.V2) {
            // Enhanced format: version:iv:salt:encrypted:authTag
            encryptedData.salt = parts[2];
            encryptedData.data = parts[3];
            encryptedData.authTag = parts[4];
        }
        else {
            // Unknown version
            throw new Error('Unknown encryption version');
        }
        return decrypt(encryptedData);
    }
    catch (error) {
        console.error('String decryption error:', error);
        throw new Error('Failed to decrypt string data');
    }
}
/**
 * Convert encrypted data object to string format
 * @param {EncryptedData} encryptedData - Encrypted data object
 * @returns {string} - String representation
 */
function encryptedDataToString(encryptedData) {
    if (encryptedData.version === VERSION_IDENTIFIER.V1) {
        return `${encryptedData.version}:${encryptedData.iv}:${encryptedData.data}:${encryptedData.authTag}`;
    }
    else if (encryptedData.version === VERSION_IDENTIFIER.V2 && encryptedData.salt) {
        return `${encryptedData.version}:${encryptedData.iv}:${encryptedData.salt}:${encryptedData.data}:${encryptedData.authTag}`;
    }
    else {
        throw new Error('Invalid encrypted data structure');
    }
}
/**
 * Re-encrypt data using the current primary key
 * Useful for key rotation when data was encrypted with an old key
 * @param {EncryptedData} encryptedData - Data encrypted with old key
 * @returns {EncryptedData} - Data re-encrypted with new primary key
 */
function reEncrypt(encryptedData) {
    // Decrypt with either primary or secondary key
    const decrypted = decrypt(encryptedData);
    // Re-encrypt with current primary key using latest version format
    return encrypt(decrypted, primaryKey, { useSalt: true });
}
/**
 * Generates a secure random encryption key
 * @returns {string} - Hex-encoded encryption key
 */
function generateSecureKey() {
    return crypto_1.default.randomBytes(KEY_SIZE).toString('hex');
}
// Also export string-based functions for backward compatibility
function encryptToString(data, key = primaryKey, options = { useSalt: true }) {
    const encryptedData = encrypt(data, key, options);
    return encryptedDataToString(encryptedData);
}
//# sourceMappingURL=encryption.js.map