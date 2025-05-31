export class DeltaExchangeService {
    constructor(apiKey: any, apiSecret: any, isTestnet?: boolean);
    apiKey: any;
    apiSecret: any;
    baseUrl: string;
    /**
     * Sign a request with HMAC signature
     * @param {string} method - HTTP method
     * @param {string} path - Endpoint path
     * @param {Object} params - Query params or body data
     * @returns {Object} - Headers with signature
     */
    createSignedHeaders(method: string, path: string, params?: any): any;
    /**
     * Execute a request to Delta Exchange API
     * @param {string} method - HTTP method
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query params or body data
     * @returns {Promise<Object>} - API response
     */
    request(method: string, endpoint: string, params?: any): Promise<any>;
    /**
     * Fetches all available markets
     * @param {Object} params - Optional parameters
     * @returns {Promise<Array>} - List of available markets
     * @see https://docs.delta.exchange/#get-list-of-products
     */
    fetchMarkets(params?: any): Promise<any[]>;
    /**
     * Fetches ticker information for a specific market
     * @param {string} symbol - Market symbol
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Ticker information
     * @see https://docs.delta.exchange/#get-ticker-for-a-product-by-symbol
     */
    fetchTicker(symbol: string, params?: any): Promise<any>;
    /**
     * Fetches account balance information
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Balance information
     * @see https://docs.delta.exchange/#get-wallet-balances
     */
    fetchBalance(params?: any): Promise<any>;
    /**
     * Fetches data on a single open position
     * @param {string} symbol - Market symbol
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Position information
     * @see https://docs.delta.exchange/#get-position
     */
    fetchPosition(symbol: string, params?: any): Promise<any>;
    /**
     * Fetches the status of the exchange API
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Status information
     */
    fetchStatus(params?: any): Promise<any>;
    /**
     * Fetches all available currencies
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Currencies information
     * @see https://docs.delta.exchange/#get-list-of-all-assets
     */
    fetchCurrencies(params?: any): Promise<any>;
    /**
     * Closes all open positions for a market type
     * @param {Object} params - Optional parameters
     * @returns {Promise<Array>} - List of closed positions
     * @see https://docs.delta.exchange/#close-all-positions
     */
    closeAllPositions(params?: any): Promise<any[]>;
    /**
     * Fetches the margin mode of a trading pair
     * @param {string} symbol - Market symbol
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Margin mode information
     * @see https://docs.delta.exchange/#get-user
     */
    fetchMarginMode(symbol: string, params?: any): Promise<any>;
    /**
     * Fetches funding rates for multiple markets
     * @param {Array<string>} symbols - List of market symbols
     * @param {Object} params - Optional parameters
     * @returns {Promise<Array>} - List of funding rate information
     * @see https://docs.delta.exchange/#get-tickers-for-products
     */
    fetchFundingRates(symbols: Array<string>, params?: any): Promise<any[]>;
    /**
     * Adds margin to a position
     * @param {string} symbol - Market symbol
     * @param {number} amount - Amount of margin to add
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Updated position information
     * @see https://docs.delta.exchange/#add-remove-position-margin
     */
    addMargin(symbol: string, amount: number, params?: any): Promise<any>;
    /**
     * Fetches the set leverage for a market
     * @param {string} symbol - Market symbol
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Leverage information
     * @see https://docs.delta.exchange/#get-order-leverage
     */
    fetchLeverage(symbol: string, params?: any): Promise<any>;
    /**
     * Sets the level of leverage for a market
     * @param {number} leverage - The rate of leverage
     * @param {string} symbol - Market symbol
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Response from the exchange
     * @see https://docs.delta.exchange/#change-order-leverage
     */
    setLeverage(leverage: number, symbol: string, params?: any): Promise<any>;
    /**
     * Creates a new order
     * @param {string} symbol - Market symbol
     * @param {string} type - Order type (limit, market, etc.)
     * @param {string} side - Order side (buy, sell)
     * @param {number} amount - Order amount
     * @param {number} price - Order price (for limit orders)
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - New order information
     * @see https://docs.delta.exchange/#place-order
     */
    createOrder(symbol: string, type: string, side: string, amount: number, price: number, params?: any): Promise<any>;
    /**
     * Cancels an existing order
     * @param {string} id - Order ID
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Cancelled order information
     * @see https://docs.delta.exchange/#cancel-order
     */
    cancelOrder(id: string, params?: any): Promise<any>;
    /**
     * Fetches an order by ID
     * @param {string} id - Order ID
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Order information
     * @see https://docs.delta.exchange/#get-order-by-id
     */
    fetchOrder(id: string, params?: any): Promise<any>;
    /**
     * Fetches open orders
     * @param {string} symbol - Market symbol (optional)
     * @param {Object} params - Optional parameters
     * @returns {Promise<Array>} - List of open orders
     * @see https://docs.delta.exchange/#get-open-orders
     */
    fetchOpenOrders(symbol?: string, params?: any): Promise<any[]>;
    /**
     * Fetches order history
     * @param {string} symbol - Market symbol (optional)
     * @param {Object} params - Optional parameters
     * @returns {Promise<Array>} - List of historical orders
     * @see https://docs.delta.exchange/#get-order-history
     */
    fetchOrderHistory(symbol?: string, params?: any): Promise<any[]>;
    /**
     * Fetches trade history
     * @param {string} symbol - Market symbol (optional)
     * @param {Object} params - Optional parameters
     * @returns {Promise<Array>} - List of trades
     * @see https://docs.delta.exchange/#get-trade-history
     */
    fetchMyTrades(symbol?: string, params?: any): Promise<any[]>;
    /**
     * Fetches the orderbook for a market
     * @param {string} symbol - Market symbol
     * @param {Object} params - Optional parameters
     * @returns {Promise<Object>} - Orderbook information
     * @see https://docs.delta.exchange/#get-l2-orderbook
     */
    fetchOrderBook(symbol: string, params?: any): Promise<any>;
    /**
     * Fetches recent trades for a market
     * @param {string} symbol - Market symbol
     * @param {number} limit - Number of trades to fetch
     * @param {Object} params - Optional parameters
     * @returns {Promise<Array>} - List of recent trades
     * @see https://docs.delta.exchange/#get-trade-history-for-a-product
     */
    fetchTrades(symbol: string, limit?: number, params?: any): Promise<any[]>;
}
/**
 * Creates a Delta Exchange service instance using environment credentials
 * @returns {DeltaExchangeService} - Service instance
 */
export function createDefaultService(): DeltaExchangeService;
/**
 * Creates a Delta Exchange service instance using provided credentials
 * @param {string} apiKey - API Key
 * @param {string} apiSecret - API Secret
 * @param {boolean} isTestnet - Whether to use testnet
 * @returns {DeltaExchangeService} - Service instance
 */
export function createService(apiKey: string, apiSecret: string, isTestnet?: boolean): DeltaExchangeService;
