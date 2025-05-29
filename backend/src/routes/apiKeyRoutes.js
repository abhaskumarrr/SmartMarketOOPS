/**
 * API Key Routes
 * Endpoints for managing Delta Exchange API keys
 */

const express = require('express');
const router = express.Router();
const { 
  createApiKey, 
  getApiKeys, 
  getApiKey, 
  deleteApiKey,
  validateApiKey
} = require('../controllers/apiKeyController');
const { auth } = require('../middleware/auth');

// All routes require authentication
router.use(auth);

// Create a new API key
router.post('/', createApiKey);

// Get all API keys for current user
router.get('/', getApiKeys);

// Get a specific API key
router.get('/:id', getApiKey);

// Delete an API key
router.delete('/:id', deleteApiKey);

// Validate an API key with Delta Exchange
router.post('/validate', validateApiKey);

module.exports = router; 