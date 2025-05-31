/**
 * Get real-time price data
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
export function getRealTimePrice(req: any, res: any, next: Function): Promise<void>;
/**
 * Get historical OHLCV data
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
export function getHistoricalData(req: any, res: any, next: Function): Promise<void>;
/**
 * Get market pairs data (available trading pairs)
 * @param {Object} req - Express request object
 * @param {Object} res - Express response object
 * @param {Function} next - Express next function
 */
export function getMarketPairs(req: any, res: any, next: Function): Promise<void>;
