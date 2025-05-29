/**
 * Security Utilities
 * Common functions for security operations
 */

import crypto from 'crypto';

/**
 * Generate a cryptographically secure random string
 * @param length Length of the string to generate
 * @param encoding Encoding to use (hex, base64, etc)
 * @returns Random string of specified length and encoding
 */
export const generateRandomString = (length = 32, encoding: BufferEncoding = 'hex'): string => {
  return crypto.randomBytes(Math.ceil(length / 2)).toString(encoding).slice(0, length);
};

/**
 * Generate a secure random token
 * @param length Length of the token to generate
 * @returns Random token string
 */
export const generateSecureToken = (length = 64): string => {
  return generateRandomString(length, 'base64');
};

/**
 * Hash a string using SHA-256
 * @param data Data to hash
 * @param salt Optional salt to add to the hash
 * @returns Hashed string
 */
export const hashString = (data: string, salt?: string): string => {
  const hash = crypto.createHash('sha256');
  hash.update(data);
  if (salt) hash.update(salt);
  return hash.digest('hex');
};

/**
 * Compare a string with a hash using constant-time comparison
 * to prevent timing attacks
 * @param string String to compare
 * @param hash Hash to compare against
 * @returns True if the string matches the hash
 */
export const secureCompare = (string: string, hash: string): boolean => {
  // Convert strings to buffers for constant-time comparison
  const stringBuffer = Buffer.from(string);
  const hashBuffer = Buffer.from(hash);
  
  // Use crypto.timingSafeEqual to prevent timing attacks
  try {
    return stringBuffer.length === hashBuffer.length && 
           crypto.timingSafeEqual(stringBuffer, hashBuffer);
  } catch (error) {
    // If buffers are different lengths, timingSafeEqual will throw
    return false;
  }
};

/**
 * Generate a random API key
 * @returns Random API key string
 */
export const generateApiKey = (): string => {
  return `ak_${generateRandomString(32)}`;
};

/**
 * Generate a random API secret
 * @returns Random API secret string
 */
export const generateApiSecret = (): string => {
  return `as_${generateRandomString(64)}`;
};

/**
 * Sanitize user input to prevent XSS attacks
 * @param input User input string
 * @returns Sanitized string
 */
export const sanitizeInput = (input: string): string => {
  // Basic sanitization - replace HTML tags with entities
  return input
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
};

/**
 * Mask sensitive data like API keys or emails
 * @param data Data to mask (e.g., API key, email)
 * @param visibleStartChars Number of characters to show at the start
 * @param visibleEndChars Number of characters to show at the end
 * @returns Masked string
 */
export const maskSensitiveData = (
  data: string,
  visibleStartChars = 4,
  visibleEndChars = 4
): string => {
  if (!data || data.length <= visibleStartChars + visibleEndChars) {
    return '****';
  }
  
  const start = data.substring(0, visibleStartChars);
  const end = data.substring(data.length - visibleEndChars);
  return `${start}****${end}`;
}; 