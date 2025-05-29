/**
 * API Key Management Controller
 * Handles API endpoints for managing API keys
 */

import { Request, Response } from 'express';
import * as apiKeyManagement from '../../services/apiKeyManagementService';
import { validationResult } from 'express-validator';

/**
 * Get all API keys for the authenticated user
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export async function getAllApiKeys(req: Request, res: Response): Promise<void> {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    
    // Get filters from query parameters
    const environment = req.query.environment as string;
    const includeRevoked = req.query.includeRevoked === 'true';
    
    const apiKeys = await apiKeyManagement.listApiKeys(userId, {
      environment,
      includeRevoked
    });
    
    res.status(200).json({
      success: true,
      data: apiKeys
    });
  } catch (error) {
    console.error('Error getting API keys:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve API keys'
    });
  }
}

/**
 * Get API key details by ID
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export async function getApiKeyById(req: Request, res: Response): Promise<void> {
  try {
    const userId = req.user?.id;
    const keyId = req.params.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    
    if (!keyId) {
      res.status(400).json({ error: 'API key ID is required' });
      return;
    }
    
    // This only returns the key summary, not the actual sensitive data
    const apiKeys = await apiKeyManagement.listApiKeys(userId);
    const apiKey = apiKeys.find(key => key.id === keyId);
    
    if (!apiKey) {
      res.status(404).json({
        success: false,
        error: 'API key not found'
      });
      return;
    }
    
    // Include usage statistics
    const usageStats = await apiKeyManagement.getApiKeyUsageStats(userId, keyId);
    
    res.status(200).json({
      success: true,
      data: {
        ...apiKey,
        usageStats
      }
    });
  } catch (error) {
    console.error('Error getting API key details:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve API key details'
    });
  }
}

/**
 * Create a new API key
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export async function createApiKey(req: Request, res: Response): Promise<void> {
  try {
    // Validate request
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      res.status(400).json({
        success: false,
        errors: errors.array()
      });
      return;
    }
    
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    
    const {
      name,
      key,
      secret,
      environment = 'testnet',
      scopes = ['trading', 'wallet'],
      expiry,
      isDefault = false,
      ipRestrictions,
      skipValidation = false
    } = req.body;
    
    // Validate required fields
    if (!name || !key || !secret) {
      res.status(400).json({
        success: false,
        error: 'Name, API key, and secret are required'
      });
      return;
    }
    
    // Convert string to array if needed
    const scopesArray = typeof scopes === 'string' ? scopes.split(',') : scopes;
    
    // Convert expiry to Date
    const expiryDate = expiry ? new Date(expiry) : new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); // Default 30 days
    
    // Create API key
    const newApiKey = await apiKeyManagement.addApiKey({
      userId,
      name,
      key,
      secret,
      environment,
      scopes: scopesArray,
      expiry: expiryDate,
      isDefault,
      ipRestrictions: ipRestrictions ? ipRestrictions.split(',') : undefined
    }, !skipValidation);
    
    res.status(201).json({
      success: true,
      message: 'API key created successfully',
      data: newApiKey
    });
  } catch (error) {
    console.error('Error creating API key:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to create API key'
    });
  }
}

/**
 * Update an existing API key
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export async function updateApiKey(req: Request, res: Response): Promise<void> {
  try {
    const userId = req.user?.id;
    const keyId = req.params.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    
    if (!keyId) {
      res.status(400).json({ error: 'API key ID is required' });
      return;
    }
    
    const {
      name,
      scopes,
      expiry,
      isDefault,
      ipRestrictions,
      metadata
    } = req.body;
    
    // Convert scopes if provided
    const scopesArray = scopes && typeof scopes === 'string' 
      ? scopes.split(',') 
      : scopes;
    
    // Convert expiry if provided
    const expiryDate = expiry ? new Date(expiry) : undefined;
    
    // Convert IP restrictions if provided
    const ipRestrictionsArray = ipRestrictions && typeof ipRestrictions === 'string'
      ? ipRestrictions.split(',')
      : ipRestrictions;
    
    // Update the API key
    const updatedKey = await apiKeyManagement.updateApiKey(userId, keyId, {
      name,
      scopes: scopesArray,
      expiry: expiryDate,
      isDefault,
      ipRestrictions: ipRestrictionsArray,
      metadata
    });
    
    res.status(200).json({
      success: true,
      message: 'API key updated successfully',
      data: updatedKey
    });
  } catch (error) {
    console.error('Error updating API key:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to update API key'
    });
  }
}

/**
 * Revoke an API key
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export async function revokeApiKey(req: Request, res: Response): Promise<void> {
  try {
    const userId = req.user?.id;
    const keyId = req.params.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    
    if (!keyId) {
      res.status(400).json({ error: 'API key ID is required' });
      return;
    }
    
    const { reason } = req.body;
    
    // Revoke the API key
    const result = await apiKeyManagement.revokeApiKey(
      userId,
      keyId,
      userId, // revokedBy - same as authenticated user
      reason || 'User requested revocation'
    );
    
    if (result) {
      res.status(200).json({
        success: true,
        message: 'API key revoked successfully'
      });
    } else {
      res.status(404).json({
        success: false,
        error: 'API key not found or already revoked'
      });
    }
  } catch (error) {
    console.error('Error revoking API key:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to revoke API key'
    });
  }
}

/**
 * Set API key as default
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export async function setDefaultApiKey(req: Request, res: Response): Promise<void> {
  try {
    const userId = req.user?.id;
    const keyId = req.params.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    
    if (!keyId) {
      res.status(400).json({ error: 'API key ID is required' });
      return;
    }
    
    // Set as default
    const result = await apiKeyManagement.setDefaultApiKey(userId, keyId);
    
    res.status(200).json({
      success: true,
      message: 'API key set as default successfully'
    });
  } catch (error) {
    console.error('Error setting default API key:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to set API key as default'
    });
  }
}

/**
 * Rotate an API key
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export async function rotateApiKey(req: Request, res: Response): Promise<void> {
  try {
    const userId = req.user?.id;
    const keyId = req.params.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    
    if (!keyId) {
      res.status(400).json({ error: 'API key ID is required' });
      return;
    }
    
    const { newKey, newSecret, skipValidation } = req.body;
    
    if (!newKey || !newSecret) {
      res.status(400).json({
        success: false,
        error: 'New API key and secret are required'
      });
      return;
    }
    
    // Rotate the key
    const newKeyData = await apiKeyManagement.rotateApiKey(
      userId,
      keyId,
      newKey,
      newSecret,
      !skipValidation
    );
    
    res.status(200).json({
      success: true,
      message: 'API key rotated successfully',
      data: newKeyData
    });
  } catch (error) {
    console.error('Error rotating API key:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to rotate API key'
    });
  }
}

/**
 * Validate an API key (without storing it)
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export async function validateApiKey(req: Request, res: Response): Promise<void> {
  try {
    const userId = req.user?.id;
    
    if (!userId) {
      res.status(401).json({ error: 'Authentication required' });
      return;
    }
    
    const { key, secret, environment = 'testnet' } = req.body;
    
    if (!key || !secret) {
      res.status(400).json({
        success: false,
        error: 'API key and secret are required'
      });
      return;
    }
    
    // Validate the key
    const validationResult = await apiKeyManagement.validateApiKey(
      key,
      secret,
      environment
    );
    
    res.status(200).json({
      success: true,
      data: validationResult
    });
  } catch (error) {
    console.error('Error validating API key:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to validate API key'
    });
  }
} 