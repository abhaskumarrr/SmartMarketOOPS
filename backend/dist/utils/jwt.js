"use strict";
/**
 * JWT Utilities
 * Functions for generating and validating JWT tokens
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.verifyRefreshToken = exports.verifyToken = exports.generateRefreshToken = exports.generateToken = void 0;
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const env_1 = __importDefault(require("./env"));
/**
 * Generate JWT access token
 * @param {string} userId - User ID
 * @returns {string} JWT token
 */
const generateToken = (userId) => {
    return jsonwebtoken_1.default.sign({ id: userId }, env_1.default.JWT_SECRET, { expiresIn: '1h' });
};
exports.generateToken = generateToken;
/**
 * Generate JWT refresh token
 * @param {string} userId - User ID
 * @returns {string} JWT refresh token
 */
const generateRefreshToken = (userId) => {
    // Use JWT_REFRESH_SECRET if available, otherwise use JWT_SECRET
    const secret = process.env.JWT_REFRESH_SECRET || env_1.default.JWT_SECRET;
    return jsonwebtoken_1.default.sign({ id: userId }, secret, { expiresIn: '7d' });
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
exports.default = {
    generateToken: exports.generateToken,
    generateRefreshToken: exports.generateRefreshToken,
    verifyToken: exports.verifyToken,
    verifyRefreshToken: exports.verifyRefreshToken
};
//# sourceMappingURL=jwt.js.map