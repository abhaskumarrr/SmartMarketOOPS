/**
 * API Key Management Controller
 * Handles API endpoints for managing API keys
 */
import { Request, Response } from 'express';
/**
 * Get all API keys for the authenticated user
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export declare function getAllApiKeys(req: Request, res: Response): Promise<void>;
/**
 * Get API key details by ID
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export declare function getApiKeyById(req: Request, res: Response): Promise<void>;
/**
 * Create a new API key
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export declare function createApiKey(req: Request, res: Response): Promise<void>;
/**
 * Update an existing API key
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export declare function updateApiKey(req: Request, res: Response): Promise<void>;
/**
 * Revoke an API key
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export declare function revokeApiKey(req: Request, res: Response): Promise<void>;
/**
 * Set API key as default
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export declare function setDefaultApiKey(req: Request, res: Response): Promise<void>;
/**
 * Rotate an API key
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export declare function rotateApiKey(req: Request, res: Response): Promise<void>;
/**
 * Validate an API key (without storing it)
 * @param {Request} req - Express request object
 * @param {Response} res - Express response object
 */
export declare function validateApiKey(req: Request, res: Response): Promise<void>;
