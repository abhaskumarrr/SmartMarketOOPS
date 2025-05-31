/**
 * JWT Utilities
 * Functions for generating and validating JWT tokens
 */
interface JwtPayload {
    id: string;
    [key: string]: any;
}
/**
 * Generate JWT access token
 * @param {string} userId - User ID
 * @returns {string} JWT token
 */
export declare const generateToken: (userId: string) => string;
/**
 * Generate JWT refresh token
 * @param {string} userId - User ID
 * @returns {string} JWT refresh token
 */
export declare const generateRefreshToken: (userId: string) => string;
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
declare const _default: {
    generateToken: (userId: string) => string;
    generateRefreshToken: (userId: string) => string;
    verifyToken: (token: string) => JwtPayload | null;
    verifyRefreshToken: (token: string) => JwtPayload | null;
};
export default _default;
