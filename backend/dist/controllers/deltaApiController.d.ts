/**
 * Get available products (markets)
 * @route GET /api/delta/products
 * @access Private
 */
export function getProducts(req: any, res: any, next: any): Promise<void>;
/**
 * Get order book for a product
 * @route GET /api/delta/products/:id/orderbook
 * @access Private
 */
export function getOrderBook(req: any, res: any, next: any): Promise<void>;
/**
 * Get recent trades for a product
 * @route GET /api/delta/products/:id/trades
 * @access Private
 */
export function getRecentTrades(req: any, res: any, next: any): Promise<void>;
/**
 * Get account balance
 * @route GET /api/delta/balance
 * @access Private
 */
export function getAccountBalance(req: any, res: any, next: any): Promise<void>;
/**
 * Get positions
 * @route GET /api/delta/positions
 * @access Private
 */
export function getPositions(req: any, res: any, next: any): Promise<void>;
/**
 * Create order
 * @route POST /api/delta/orders
 * @access Private
 */
export function createOrder(req: any, res: any, next: any): Promise<any>;
/**
 * Cancel order
 * @route DELETE /api/delta/orders/:id
 * @access Private
 */
export function cancelOrder(req: any, res: any, next: any): Promise<void>;
/**
 * Get orders
 * @route GET /api/delta/orders
 * @access Private
 */
export function getOrders(req: any, res: any, next: any): Promise<void>;
/**
 * Cancel all orders
 * @route DELETE /api/delta/orders
 * @access Private
 */
export function cancelAllOrders(req: any, res: any, next: any): Promise<void>;
/**
 * Get order history
 * @route GET /api/delta/orders/history
 * @access Private
 */
export function getOrderHistory(req: any, res: any, next: any): Promise<void>;
/**
 * Get trade history
 * @route GET /api/delta/fills
 * @access Private
 */
export function getTradeHistory(req: any, res: any, next: any): Promise<void>;
/**
 * Get comprehensive market data
 * @route GET /api/delta/market-data
 * @access Private
 */
export function getMarketData(req: any, res: any, next: any): Promise<any>;
/**
 * Get leverage settings
 * @route GET /api/delta/leverage
 * @access Private
 */
export function getLeverage(req: any, res: any, next: any): Promise<any>;
/**
 * Set leverage
 * @route POST /api/delta/leverage
 * @access Private
 */
export function setLeverage(req: any, res: any, next: any): Promise<any>;
/**
 * Get available currencies
 * @route GET /api/delta/currencies
 * @access Private
 */
export function getCurrencies(req: any, res: any, next: any): Promise<void>;
/**
 * Close all positions
 * @route POST /api/delta/positions/close-all
 * @access Private
 */
export function closeAllPositions(req: any, res: any, next: any): Promise<void>;
/**
 * Add margin to position
 * @route POST /api/delta/positions/add-margin
 * @access Private
 */
export function addMargin(req: any, res: any, next: any): Promise<any>;
