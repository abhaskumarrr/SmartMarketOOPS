/**
 * Create a new API key
 * @route POST /api/keys
 * @access Private
 */
export function createApiKey(req: any, res: any): Promise<any>;
/**
 * Get all API keys for current user
 * @route GET /api/keys
 * @access Private
 */
export function getApiKeys(req: any, res: any): Promise<void>;
/**
 * Get a specific API key
 * @route GET /api/keys/:id
 * @access Private
 */
export function getApiKey(req: any, res: any): Promise<any>;
/**
 * Delete an API key
 * @route DELETE /api/keys/:id
 * @access Private
 */
export function deleteApiKey(req: any, res: any): Promise<any>;
/**
 * Validate an API key with Delta Exchange
 * @route POST /api/keys/validate
 * @access Private
 */
export function validateApiKey(req: any, res: any): Promise<any>;
