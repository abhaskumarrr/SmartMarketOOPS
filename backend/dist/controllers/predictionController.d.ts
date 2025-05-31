/**
 * Get latest price predictions for a specific symbol
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
export function getLatestPredictions(req: any, res: any, next: Function): Promise<void>;
/**
 * Get historical predictions and their performance
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
export function getPredictionHistory(req: any, res: any, next: Function): Promise<void>;
/**
 * Get current market analysis
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
export function getMarketAnalysis(req: any, res: any, next: Function): Promise<void>;
