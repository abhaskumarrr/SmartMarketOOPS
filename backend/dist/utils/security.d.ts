/**
 * Security Utilities
 * Common functions for security operations
 */
/**
 * Generate a cryptographically secure random string
 * @param length Length of the string to generate
 * @param encoding Encoding to use (hex, base64, etc)
 * @returns Random string of specified length and encoding
 */
export declare const generateRandomString: (length?: number, encoding?: BufferEncoding) => string;
/**
 * Generate a secure random token
 * @param length Length of the token to generate
 * @returns Random token string
 */
export declare const generateSecureToken: (length?: number) => string;
/**
 * Hash a string using SHA-256
 * @param data Data to hash
 * @param salt Optional salt to add to the hash
 * @returns Hashed string
 */
export declare const hashString: (data: string, salt?: string) => string;
/**
 * Compare a string with a hash using constant-time comparison
 * to prevent timing attacks
 * @param string String to compare
 * @param hash Hash to compare against
 * @returns True if the string matches the hash
 */
export declare const secureCompare: (string: string, hash: string) => boolean;
/**
 * Generate a random API key
 * @returns Random API key string
 */
export declare const generateApiKey: () => string;
/**
 * Generate a random API secret
 * @returns Random API secret string
 */
export declare const generateApiSecret: () => string;
/**
 * Sanitize user input to prevent XSS attacks
 * @param input User input string
 * @returns Sanitized string
 */
export declare const sanitizeInput: (input: string) => string;
/**
 * Mask sensitive data like API keys or emails
 * @param data Data to mask (e.g., API key, email)
 * @param visibleStartChars Number of characters to show at the start
 * @param visibleEndChars Number of characters to show at the end
 * @returns Masked string
 */
export declare const maskSensitiveData: (data: string, visibleStartChars?: number, visibleEndChars?: number) => string;
