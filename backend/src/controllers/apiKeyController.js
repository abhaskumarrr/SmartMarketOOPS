/**
 * API Key Controller
 * Handles management and validation of Delta Exchange API keys
 */

const prisma = require('../utils/prismaClient');
const { encrypt, decrypt } = require('../utils/encryption');
const crypto = require('crypto');
const axios = require('axios');

/**
 * Create a new API key
 * @route POST /api/keys
 * @access Private
 */
const createApiKey = async (req, res) => {
  try {
    const { apiKey, apiSecret, label, testnet } = req.body;
    const userId = req.user.id;

    // Validate input
    if (!apiKey || !apiSecret) {
      return res.status(400).json({
        success: false,
        message: 'API key and secret are required'
      });
    }

    // Check if user already has an API key with this label
    if (label) {
      const existingKey = await prisma.apiKey.findFirst({
        where: {
          userId,
          encryptedData: {
            contains: label
          }
        }
      });

      if (existingKey) {
        return res.status(400).json({
          success: false,
          message: `An API key with label "${label}" already exists`
        });
      }
    }

    // Encrypt the API key and secret
    const encryptedData = encrypt({
      apiKey,
      apiSecret,
      label: label || 'Default',
      testnet: !!testnet
    });

    // Generate a unique identifier for reference
    const keyIdentifier = crypto.randomBytes(16).toString('hex');

    // Create API key record
    const newApiKey = await prisma.apiKey.create({
      data: {
        key: keyIdentifier,
        encryptedData,
        userId,
        scopes: 'trade,account,market', // Default scopes
        expiry: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000) // 90 days from now
      }
    });

    res.status(201).json({
      success: true,
      data: {
        id: newApiKey.id,
        key: keyIdentifier,
        scopes: newApiKey.scopes,
        expiry: newApiKey.expiry,
        createdAt: newApiKey.createdAt
      }
    });
  } catch (error) {
    console.error('Create API key error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while creating API key',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get all API keys for current user
 * @route GET /api/keys
 * @access Private
 */
const getApiKeys = async (req, res) => {
  try {
    const userId = req.user.id;

    // Get all API keys for user
    const apiKeys = await prisma.apiKey.findMany({
      where: { userId }
    });

    // Map API keys to response format (with decrypted labels)
    const formattedKeys = apiKeys.map(key => {
      const decryptedData = decrypt(key.encryptedData);
      return {
        id: key.id,
        key: key.key,
        label: decryptedData.label || 'Default',
        testnet: decryptedData.testnet || false,
        scopes: key.scopes,
        expiry: key.expiry,
        createdAt: key.createdAt
      };
    });

    res.status(200).json({
      success: true,
      data: formattedKeys
    });
  } catch (error) {
    console.error('Get API keys error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching API keys',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Get a specific API key
 * @route GET /api/keys/:id
 * @access Private
 */
const getApiKey = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    // Find API key by id and user id
    const apiKey = await prisma.apiKey.findFirst({
      where: {
        id,
        userId
      }
    });

    if (!apiKey) {
      return res.status(404).json({
        success: false,
        message: 'API key not found'
      });
    }

    // Decrypt API key data
    const decryptedData = decrypt(apiKey.encryptedData);

    res.status(200).json({
      success: true,
      data: {
        id: apiKey.id,
        key: apiKey.key,
        label: decryptedData.label || 'Default',
        testnet: decryptedData.testnet || false,
        scopes: apiKey.scopes,
        expiry: apiKey.expiry,
        createdAt: apiKey.createdAt
      }
    });
  } catch (error) {
    console.error('Get API key error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while fetching API key',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Delete an API key
 * @route DELETE /api/keys/:id
 * @access Private
 */
const deleteApiKey = async (req, res) => {
  try {
    const { id } = req.params;
    const userId = req.user.id;

    // Find API key by id and user id
    const apiKey = await prisma.apiKey.findFirst({
      where: {
        id,
        userId
      }
    });

    if (!apiKey) {
      return res.status(404).json({
        success: false,
        message: 'API key not found'
      });
    }

    // Delete API key
    await prisma.apiKey.delete({
      where: { id }
    });

    res.status(200).json({
      success: true,
      message: 'API key deleted successfully'
    });
  } catch (error) {
    console.error('Delete API key error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while deleting API key',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Validate an API key with Delta Exchange
 * @route POST /api/keys/validate
 * @access Private
 */
const validateApiKey = async (req, res) => {
  try {
    const { apiKey, apiSecret, testnet } = req.body;

    let baseUrl = testnet 
      ? 'https://testnet.delta.exchange/api/v2' 
      : 'https://api.delta.exchange/v2';

    // Attempt to access a simple endpoint that requires auth (e.g., get account details)
    try {
      const timestamp = Math.floor(Date.now() / 1000);
      const signature = createSignature(apiSecret, 'GET', '/wallet/balances', timestamp, {});
      
      const response = await axios({
        method: 'GET',
        url: `${baseUrl}/wallet/balances`,
        headers: {
          'api-key': apiKey,
          'timestamp': timestamp,
          'signature': signature
        }
      });

      if (response.status === 200) {
        return res.status(200).json({
          success: true,
          message: 'API key is valid',
          data: {
            valid: true
          }
        });
      }
    } catch (apiError) {
      return res.status(400).json({
        success: false,
        message: 'Invalid API credentials',
        data: {
          valid: false,
          error: apiError.response?.data?.message || 'Unable to validate API key'
        }
      });
    }
  } catch (error) {
    console.error('Validate API key error:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while validating API key',
      error: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
  }
};

/**
 * Create signature for Delta Exchange API
 * @param {string} secret - API secret
 * @param {string} method - HTTP method
 * @param {string} path - API path
 * @param {number} timestamp - Unix timestamp
 * @param {Object} params - Query parameters or request body
 * @returns {string} - HMAC signature
 */
function createSignature(secret, method, path, timestamp, params) {
  const messageString = timestamp + method + path;
  
  // Add query params or body params if present
  if (Object.keys(params).length > 0) {
    const sortedParams = Object.keys(params).sort().reduce((acc, key) => {
      acc[key] = params[key];
      return acc;
    }, {});
    const paramString = new URLSearchParams(sortedParams).toString();
    messageString += paramString;
  }
  
  // Create HMAC signature
  return crypto
    .createHmac('sha256', secret)
    .update(messageString)
    .digest('hex');
}

module.exports = {
  createApiKey,
  getApiKeys,
  getApiKey,
  deleteApiKey,
  validateApiKey
}; 