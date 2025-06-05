/**
 * API Key Controller
 * Wrapper controller that delegates to the trading API key controller
 */

import { Request, Response } from 'express';
import {
  getAllApiKeys,
  getApiKeyById,
  createApiKey as createTradingApiKey,
  revokeApiKey,
  validateApiKey as validateTradingApiKey
} from './trading/apiKeyController';

/**
 * Create a new API key
 */
export const createApiKey = async (req: Request, res: Response): Promise<void> => {
  await createTradingApiKey(req, res);
};

/**
 * Get all API keys for current user
 */
export const getApiKeys = async (req: Request, res: Response): Promise<void> => {
  await getAllApiKeys(req, res);
};

/**
 * Get a specific API key
 */
export const getApiKey = async (req: Request, res: Response): Promise<void> => {
  await getApiKeyById(req, res);
};

/**
 * Delete an API key
 */
export const deleteApiKey = async (req: Request, res: Response): Promise<void> => {
  await revokeApiKey(req, res);
};

/**
 * Validate an API key with Delta Exchange
 */
export const validateApiKey = async (req: Request, res: Response): Promise<void> => {
  await validateTradingApiKey(req, res);
};
