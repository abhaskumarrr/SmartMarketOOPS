"use strict";
/**
 * Security Utilities
 * Common functions for security operations
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.maskSensitiveData = exports.sanitizeInput = exports.generateApiSecret = exports.generateApiKey = exports.secureCompare = exports.hashString = exports.generateSecureToken = exports.generateRandomString = void 0;
const crypto_1 = __importDefault(require("crypto"));
/**
 * Generate a cryptographically secure random string
 * @param length Length of the string to generate
 * @param encoding Encoding to use (hex, base64, etc)
 * @returns Random string of specified length and encoding
 */
const generateRandomString = (length = 32, encoding = 'hex') => {
    return crypto_1.default.randomBytes(Math.ceil(length / 2)).toString(encoding).slice(0, length);
};
exports.generateRandomString = generateRandomString;
/**
 * Generate a secure random token
 * @param length Length of the token to generate
 * @returns Random token string
 */
const generateSecureToken = (length = 64) => {
    return (0, exports.generateRandomString)(length, 'base64');
};
exports.generateSecureToken = generateSecureToken;
/**
 * Hash a string using SHA-256
 * @param data Data to hash
 * @param salt Optional salt to add to the hash
 * @returns Hashed string
 */
const hashString = (data, salt) => {
    const hash = crypto_1.default.createHash('sha256');
    hash.update(data);
    if (salt)
        hash.update(salt);
    return hash.digest('hex');
};
exports.hashString = hashString;
/**
 * Compare a string with a hash using constant-time comparison
 * to prevent timing attacks
 * @param string String to compare
 * @param hash Hash to compare against
 * @returns True if the string matches the hash
 */
const secureCompare = (string, hash) => {
    // Convert strings to buffers for constant-time comparison
    const stringBuffer = Buffer.from(string);
    const hashBuffer = Buffer.from(hash);
    // Use crypto.timingSafeEqual to prevent timing attacks
    try {
        return stringBuffer.length === hashBuffer.length &&
            crypto_1.default.timingSafeEqual(stringBuffer, hashBuffer);
    }
    catch (error) {
        // If buffers are different lengths, timingSafeEqual will throw
        return false;
    }
};
exports.secureCompare = secureCompare;
/**
 * Generate a random API key
 * @returns Random API key string
 */
const generateApiKey = () => {
    return `ak_${(0, exports.generateRandomString)(32)}`;
};
exports.generateApiKey = generateApiKey;
/**
 * Generate a random API secret
 * @returns Random API secret string
 */
const generateApiSecret = () => {
    return `as_${(0, exports.generateRandomString)(64)}`;
};
exports.generateApiSecret = generateApiSecret;
/**
 * Sanitize user input to prevent XSS attacks
 * @param input User input string
 * @returns Sanitized string
 */
const sanitizeInput = (input) => {
    // Basic sanitization - replace HTML tags with entities
    return input
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
};
exports.sanitizeInput = sanitizeInput;
/**
 * Mask sensitive data like API keys or emails
 * @param data Data to mask (e.g., API key, email)
 * @param visibleStartChars Number of characters to show at the start
 * @param visibleEndChars Number of characters to show at the end
 * @returns Masked string
 */
const maskSensitiveData = (data, visibleStartChars = 4, visibleEndChars = 4) => {
    if (!data || data.length <= visibleStartChars + visibleEndChars) {
        return '****';
    }
    const start = data.substring(0, visibleStartChars);
    const end = data.substring(data.length - visibleEndChars);
    return `${start}****${end}`;
};
exports.maskSensitiveData = maskSensitiveData;
//# sourceMappingURL=security.js.map