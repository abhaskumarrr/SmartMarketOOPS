"use strict";
/**
 * JWT Utilities
 * Functions for generating and validating JWT tokens
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getTokenExpiration = exports.isTokenExpired = exports.extractTokenFromHeader = exports.generateTokenPair = exports.verifyRefreshToken = exports.verifyToken = exports.generateRefreshToken = exports.generateToken = void 0;
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const env_1 = __importDefault(require("./env"));
/**
 * Generate JWT access token (15 minutes for enhanced security)
 * @param {string} userId - User ID
 * @param {object} additionalPayload - Additional payload data
 * @returns {string} JWT token
 */
const generateToken = (userId, additionalPayload = {}) => {
    return jsonwebtoken_1.default.sign({
        id: userId,
        type: 'access',
        iat: Math.floor(Date.now() / 1000),
        ...additionalPayload
    }, env_1.default.JWT_SECRET, { expiresIn: '15m' } // 15 minutes for enhanced security
    );
};
exports.generateToken = generateToken;
/**
 * Generate JWT refresh token (7 days)
 * @param {string} userId - User ID
 * @param {string} sessionId - Session ID for token rotation
 * @returns {string} JWT refresh token
 */
const generateRefreshToken = (userId, sessionId) => {
    // Use JWT_REFRESH_SECRET if available, otherwise use JWT_SECRET
    const secret = process.env.JWT_REFRESH_SECRET || env_1.default.JWT_SECRET;
    return jsonwebtoken_1.default.sign({
        id: userId,
        type: 'refresh',
        sessionId,
        iat: Math.floor(Date.now() / 1000)
    }, secret, { expiresIn: '7d' } // 7 days for refresh token
    );
};
exports.generateRefreshToken = generateRefreshToken;
/**
 * Verify JWT token
 * @param {string} token - JWT token to verify
 * @returns {JwtPayload|null} Decoded token payload or null if invalid
 */
const verifyToken = (token) => {
    try {
        return jsonwebtoken_1.default.verify(token, env_1.default.JWT_SECRET);
    }
    catch (error) {
        console.error('Token verification failed:', error instanceof Error ? error.message : error);
        return null;
    }
};
exports.verifyToken = verifyToken;
/**
 * Verify JWT refresh token
 * @param {string} token - JWT refresh token to verify
 * @returns {JwtPayload|null} Decoded token payload or null if invalid
 */
const verifyRefreshToken = (token) => {
    try {
        // Use JWT_REFRESH_SECRET if available, otherwise use JWT_SECRET
        const secret = process.env.JWT_REFRESH_SECRET || env_1.default.JWT_SECRET;
        return jsonwebtoken_1.default.verify(token, secret);
    }
    catch (error) {
        console.error('Refresh token verification failed:', error instanceof Error ? error.message : error);
        return null;
    }
};
exports.verifyRefreshToken = verifyRefreshToken;
/**
 * Generate token pair (access + refresh)
 * @param {string} userId - User ID
 * @param {string} sessionId - Session ID
 * @param {object} additionalPayload - Additional payload for access token
 * @returns {object} Token pair
 */
const generateTokenPair = (userId, sessionId, additionalPayload = {}) => {
    return {
        accessToken: (0, exports.generateToken)(userId, additionalPayload),
        refreshToken: (0, exports.generateRefreshToken)(userId, sessionId),
        expiresIn: 15 * 60, // 15 minutes in seconds
        tokenType: 'Bearer'
    };
};
exports.generateTokenPair = generateTokenPair;
/**
 * Extract token from Authorization header
 * @param {string} authHeader - Authorization header value
 * @returns {string|null} Token or null if invalid
 */
const extractTokenFromHeader = (authHeader) => {
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return null;
    }
    return authHeader.substring(7); // Remove 'Bearer ' prefix
};
exports.extractTokenFromHeader = extractTokenFromHeader;
/**
 * Check if token is expired
 * @param {JwtPayload} payload - Decoded JWT payload
 * @returns {boolean} True if token is expired
 */
const isTokenExpired = (payload) => {
    if (!payload.exp)
        return true;
    return Date.now() >= payload.exp * 1000;
};
exports.isTokenExpired = isTokenExpired;
/**
 * Get token expiration time
 * @param {JwtPayload} payload - Decoded JWT payload
 * @returns {Date|null} Expiration date or null
 */
const getTokenExpiration = (payload) => {
    if (!payload.exp)
        return null;
    return new Date(payload.exp * 1000);
};
exports.getTokenExpiration = getTokenExpiration;
exports.default = {
    generateToken: exports.generateToken,
    generateRefreshToken: exports.generateRefreshToken,
    generateTokenPair: exports.generateTokenPair,
    verifyToken: exports.verifyToken,
    verifyRefreshToken: exports.verifyRefreshToken,
    extractTokenFromHeader: exports.extractTokenFromHeader,
    isTokenExpired: exports.isTokenExpired,
    getTokenExpiration: exports.getTokenExpiration
};
//# sourceMappingURL=jwt.js.map