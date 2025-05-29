/**
 * API Key Routes
 * Endpoints for managing Delta Exchange API keys
 */

import express from 'express';
import { 
  createApiKey, 
  getApiKeys, 
  getApiKey, 
  deleteApiKey,
  validateApiKey
} from '../controllers/apiKeyController';
import { protect, requireVerified, requirePermission } from '../middleware/auth';

const router = express.Router();

// All routes require authentication
router.use(protect);

// All routes require verified email
router.use(requireVerified);

// Create a new API key
router.post('/', requirePermission(['apiKeys:create']), createApiKey);

// Get all API keys for current user
router.get('/', requirePermission(['apiKeys:read']), getApiKeys);

// Get a specific API key
router.get('/:id', requirePermission(['apiKeys:read']), getApiKey);

// Delete an API key
router.delete('/:id', requirePermission(['apiKeys:delete']), deleteApiKey);

// Validate an API key with Delta Exchange
router.post('/validate', requirePermission(['apiKeys:read']), validateApiKey);

export default router; 