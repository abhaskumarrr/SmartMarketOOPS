"use strict";
/**
 * API Key Management Routes
 * Defines endpoints for managing exchange API keys
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const apiKeyController = __importStar(require("../../controllers/trading/apiKeyController"));
const express_validator_1 = require("express-validator");
const authMiddleware_1 = require("../../middleware/authMiddleware");
const rateLimiter_1 = __importDefault(require("../../middleware/rateLimiter"));
const security_1 = require("../../utils/security");
const router = (0, express_1.Router)();
// Apply authentication middleware to all API key routes
router.use(authMiddleware_1.authenticateJWT);
/**
 * @route GET /api/keys
 * @desc Get all API keys for the authenticated user
 * @access Private
 */
router.get('/', apiKeyController.getAllApiKeys);
/**
 * @route GET /api/keys/:id
 * @desc Get API key by ID
 * @access Private
 */
router.get('/:id', apiKeyController.getApiKeyById);
/**
 * @route POST /api/keys
 * @desc Create a new API key
 * @access Private
 */
router.post('/', [
    // Apply rate limiting for API key creation
    rateLimiter_1.default.apiKeyManagementLimiter,
    // Validation middleware
    (0, express_validator_1.body)('name')
        .notEmpty().withMessage('Name is required')
        .customSanitizer(security_1.sanitizeInput),
    (0, express_validator_1.body)('key').notEmpty().withMessage('API key is required'),
    (0, express_validator_1.body)('secret').notEmpty().withMessage('API secret is required'),
    (0, express_validator_1.body)('environment').isIn(['testnet', 'mainnet']).withMessage('Environment must be either testnet or mainnet'),
    (0, express_validator_1.body)('scopes').optional().isArray().withMessage('Scopes must be an array'),
    (0, express_validator_1.body)('expiry').optional().isISO8601().withMessage('Expiry must be a valid date'),
    (0, express_validator_1.body)('isDefault').optional().isBoolean().withMessage('isDefault must be a boolean'),
    (0, express_validator_1.body)('ipRestrictions').optional(),
    (0, express_validator_1.body)('skipValidation').optional().isBoolean().withMessage('skipValidation must be a boolean')
], apiKeyController.createApiKey);
/**
 * @route PUT /api/keys/:id
 * @desc Update an API key
 * @access Private
 */
router.put('/:id', [
    // Apply rate limiting for API key updates
    rateLimiter_1.default.apiKeyManagementLimiter,
    // Validation middleware
    (0, express_validator_1.param)('id').notEmpty().withMessage('API key ID is required'),
    (0, express_validator_1.body)('name').optional().customSanitizer(security_1.sanitizeInput),
    (0, express_validator_1.body)('scopes').optional(),
    (0, express_validator_1.body)('expiry').optional().isISO8601().withMessage('Expiry must be a valid date'),
    (0, express_validator_1.body)('isDefault').optional().isBoolean().withMessage('isDefault must be a boolean'),
    (0, express_validator_1.body)('ipRestrictions').optional()
], apiKeyController.updateApiKey);
/**
 * @route DELETE /api/keys/:id
 * @desc Revoke an API key
 * @access Private
 */
router.delete('/:id', [
    // Apply rate limiting for API key revocation
    rateLimiter_1.default.apiKeyManagementLimiter,
    (0, express_validator_1.param)('id').notEmpty().withMessage('API key ID is required'),
    (0, express_validator_1.body)('reason').optional()
], apiKeyController.revokeApiKey);
/**
 * @route POST /api/keys/:id/default
 * @desc Set an API key as default
 * @access Private
 */
router.post('/:id/default', [
    // Apply standard rate limiting
    rateLimiter_1.default.standardLimiter,
    (0, express_validator_1.param)('id').notEmpty().withMessage('API key ID is required')
], apiKeyController.setDefaultApiKey);
/**
 * @route POST /api/keys/:id/rotate
 * @desc Rotate an API key
 * @access Private
 */
router.post('/:id/rotate', [
    // Apply rate limiting for sensitive operations
    rateLimiter_1.default.sensitiveOperationsLimiter,
    (0, express_validator_1.param)('id').notEmpty().withMessage('API key ID is required'),
    (0, express_validator_1.body)('newKey').notEmpty().withMessage('New API key is required'),
    (0, express_validator_1.body)('newSecret').notEmpty().withMessage('New API secret is required'),
    (0, express_validator_1.body)('skipValidation').optional().isBoolean().withMessage('skipValidation must be a boolean')
], apiKeyController.rotateApiKey);
/**
 * @route POST /api/keys/validate
 * @desc Validate an API key without storing it
 * @access Private
 */
router.post('/validate', [
    // Apply stricter rate limiting for API key validation to prevent brute force attacks
    rateLimiter_1.default.apiKeyValidationLimiter,
    (0, express_validator_1.body)('key').notEmpty().withMessage('API key is required'),
    (0, express_validator_1.body)('secret').notEmpty().withMessage('API secret is required'),
    (0, express_validator_1.body)('environment').optional().isIn(['testnet', 'mainnet']).withMessage('Environment must be either testnet or mainnet')
], apiKeyController.validateApiKey);
exports.default = router;
//# sourceMappingURL=apiKeyRoutes.js.map