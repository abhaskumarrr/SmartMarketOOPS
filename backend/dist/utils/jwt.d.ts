/**
 * JWT Utilities
 * Functions for generating and validating JWT tokens
 */
interface JwtPayload {
    id: string;
    [key: string]: any;
}
/**
 * Generate JWT access token (15 minutes for enhanced security)
 * @param {string} userId - User ID
 * @param {object} additionalPayload - Additional payload data
 * @returns {string} JWT token
 */
export declare const generateToken: (userId: string, additionalPayload?: object) => string;
/**
 * Generate JWT refresh token (7 days)
 * @param {string} userId - User ID
 * @param {string} sessionId - Session ID for token rotation
 * @returns {string} JWT refresh token
 */
export declare const generateRefreshToken: (userId: string, sessionId?: string) => string;
/**
 * Verify JWT token
 * @param {string} token - JWT token to verify
 * @returns {JwtPayload|null} Decoded token payload or null if invalid
 */
export declare const verifyToken: (token: string) => JwtPayload | null;
/**
 * Verify JWT refresh token
 * @param {string} token - JWT refresh token to verify
 * @returns {JwtPayload|null} Decoded token payload or null if invalid
 */
export declare const verifyRefreshToken: (token: string) => JwtPayload | null;
/**
 * Generate token pair (access + refresh)
 * @param {string} userId - User ID
 * @param {string} sessionId - Session ID
 * @param {object} additionalPayload - Additional payload for access token
 * @returns {object} Token pair
 */
export declare const generateTokenPair: (userId: string, sessionId?: string, additionalPayload?: object) => {
    accessToken: string;
    refreshToken: string;
    expiresIn: number;
    tokenType: string;
};
/**
 * Extract token from Authorization header
 * @param {string} authHeader - Authorization header value
 * @returns {string|null} Token or null if invalid
 */
export declare const extractTokenFromHeader: (authHeader: string) => string | null;
/**
 * Check if token is expired
 * @param {JwtPayload} payload - Decoded JWT payload
 * @returns {boolean} True if token is expired
 */
export declare const isTokenExpired: (payload: JwtPayload) => boolean;
/**
 * Get token expiration time
 * @param {JwtPayload} payload - Decoded JWT payload
 * @returns {Date|null} Expiration date or null
 */
export declare const getTokenExpiration: (payload: JwtPayload) => Date | null;
declare const _default: {
    generateToken: (userId: string, additionalPayload?: object) => string;
    generateRefreshToken: (userId: string, sessionId?: string) => string;
    generateTokenPair: (userId: string, sessionId?: string, additionalPayload?: object) => {
        accessToken: string;
        refreshToken: string;
        expiresIn: number;
        tokenType: string;
    };
    verifyToken: (token: string) => JwtPayload | null;
    verifyRefreshToken: (token: string) => JwtPayload | null;
    extractTokenFromHeader: (authHeader: string) => string | null;
    isTokenExpired: (payload: JwtPayload) => boolean;
    getTokenExpiration: (payload: JwtPayload) => Date | null;
};
export default _default;
