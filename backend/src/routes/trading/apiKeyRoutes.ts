/**
 * API Key Management Routes
 * Defines endpoints for managing exchange API keys
 */

import { Router } from 'express';
import * as apiKeyController from '../../controllers/trading/apiKeyController';
import { body, param } from 'express-validator';
import { authenticateJWT } from '../../middleware/authMiddleware';
import rateLimiter from '../../middleware/rateLimiter';
import { sanitizeInput } from '../../utils/security';

const router = Router();

// Apply authentication middleware to all API key routes
router.use(authenticateJWT);

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
  rateLimiter.apiKeyManagementLimiter,
  // Validation middleware
  body('name')
    .notEmpty().withMessage('Name is required')
    .customSanitizer(sanitizeInput),
  body('key').notEmpty().withMessage('API key is required'),
  body('secret').notEmpty().withMessage('API secret is required'),
  body('environment').isIn(['testnet', 'mainnet']).withMessage('Environment must be either testnet or mainnet'),
  body('scopes').optional().isArray().withMessage('Scopes must be an array'),
  body('expiry').optional().isISO8601().withMessage('Expiry must be a valid date'),
  body('isDefault').optional().isBoolean().withMessage('isDefault must be a boolean'),
  body('ipRestrictions').optional(),
  body('skipValidation').optional().isBoolean().withMessage('skipValidation must be a boolean')
], apiKeyController.createApiKey);

/**
 * @route PUT /api/keys/:id
 * @desc Update an API key
 * @access Private
 */
router.put('/:id', [
  // Apply rate limiting for API key updates
  rateLimiter.apiKeyManagementLimiter,
  // Validation middleware
  param('id').notEmpty().withMessage('API key ID is required'),
  body('name').optional().customSanitizer(sanitizeInput),
  body('scopes').optional(),
  body('expiry').optional().isISO8601().withMessage('Expiry must be a valid date'),
  body('isDefault').optional().isBoolean().withMessage('isDefault must be a boolean'),
  body('ipRestrictions').optional()
], apiKeyController.updateApiKey);

/**
 * @route DELETE /api/keys/:id
 * @desc Revoke an API key
 * @access Private
 */
router.delete('/:id', [
  // Apply rate limiting for API key revocation
  rateLimiter.apiKeyManagementLimiter,
  param('id').notEmpty().withMessage('API key ID is required'),
  body('reason').optional()
], apiKeyController.revokeApiKey);

/**
 * @route POST /api/keys/:id/default
 * @desc Set an API key as default
 * @access Private
 */
router.post('/:id/default', [
  // Apply standard rate limiting
  rateLimiter.standardLimiter,
  param('id').notEmpty().withMessage('API key ID is required')
], apiKeyController.setDefaultApiKey);

/**
 * @route POST /api/keys/:id/rotate
 * @desc Rotate an API key
 * @access Private
 */
router.post('/:id/rotate', [
  // Apply rate limiting for sensitive operations
  rateLimiter.sensitiveOperationsLimiter,
  param('id').notEmpty().withMessage('API key ID is required'),
  body('newKey').notEmpty().withMessage('New API key is required'),
  body('newSecret').notEmpty().withMessage('New API secret is required'),
  body('skipValidation').optional().isBoolean().withMessage('skipValidation must be a boolean')
], apiKeyController.rotateApiKey);

/**
 * @route POST /api/keys/validate
 * @desc Validate an API key without storing it
 * @access Private
 */
router.post('/validate', [
  // Apply stricter rate limiting for API key validation to prevent brute force attacks
  rateLimiter.apiKeyValidationLimiter,
  body('key').notEmpty().withMessage('API key is required'),
  body('secret').notEmpty().withMessage('API secret is required'),
  body('environment').optional().isIn(['testnet', 'mainnet']).withMessage('Environment must be either testnet or mainnet')
], apiKeyController.validateApiKey);

export default router; 