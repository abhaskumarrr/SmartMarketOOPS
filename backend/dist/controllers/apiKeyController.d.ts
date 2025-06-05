/**
 * API Key Controller
 * Wrapper controller that delegates to the trading API key controller
 */
import { Request, Response } from 'express';
/**
 * Create a new API key
 */
export declare const createApiKey: (req: Request, res: Response) => Promise<void>;
/**
 * Get all API keys for current user
 */
export declare const getApiKeys: (req: Request, res: Response) => Promise<void>;
/**
 * Get a specific API key
 */
export declare const getApiKey: (req: Request, res: Response) => Promise<void>;
/**
 * Delete an API key
 */
export declare const deleteApiKey: (req: Request, res: Response) => Promise<void>;
/**
 * Validate an API key with Delta Exchange
 */
export declare const validateApiKey: (req: Request, res: Response) => Promise<void>;
