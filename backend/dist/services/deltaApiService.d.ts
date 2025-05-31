/**
 * Delta Exchange API Service
 * Handles communication with Delta Exchange API (both testnet and mainnet)
 *
 * References:
 * - Official Delta Exchange Documentation: https://docs.delta.exchange
 * - CCXT Delta Exchange Documentation: https://docs.ccxt.com/#/exchanges/delta
 */
import * as DeltaExchange from '../types/deltaExchange';
/**
 * DeltaExchangeAPI Service
 * Provides methods to interact with Delta Exchange API
 */
declare class DeltaExchangeAPI {
    private testnet;
    private baseUrl;
    private rateLimit;
    private userId?;
    private apiKeys;
    private client;
    /**
     * Creates a new instance of the Delta Exchange API client
     * @param {DeltaExchange.ApiOptions} options - Configuration options
     */
    constructor(options?: DeltaExchange.ApiOptions);
    /**
     * Initializes the API client with credentials
     * @param {DeltaExchange.ApiCredentials} credentials - API credentials (optional, will use stored keys if not provided)
     */
    initialize(credentials?: DeltaExchange.ApiCredentials | null): Promise<this>;
    /**
     * Gets server time from Delta Exchange
     * @returns {Promise<DeltaExchange.ServerTime>} Server time information
     */
    getServerTime(): Promise<DeltaExchange.ServerTime>;
    /**
     * Gets all available markets from Delta Exchange
     * @param {Record<string, any>} params - Query parameters
     * @returns {Promise<DeltaExchange.Market[]>} Available markets
     */
    getMarkets(params?: Record<string, any>): Promise<DeltaExchange.Market[]>;
    /**
     * Gets market data for a specific symbol
     * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
     * @returns {Promise<DeltaExchange.Market>} Market data
     */
    getMarketData(symbol: string): Promise<DeltaExchange.Market>;
    /**
     * Gets ticker information for a specific symbol
     * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
     * @returns {Promise<DeltaExchange.Ticker>} Ticker data
     */
    getTicker(symbol: string): Promise<DeltaExchange.Ticker>;
    /**
     * Gets orderbook for a specific symbol
     * @param {string} symbol - Market symbol (e.g., 'BTCUSD')
     * @param {number} depth - Orderbook depth (default: 10)
     * @returns {Promise<DeltaExchange.Orderbook>} Orderbook data
     */
    getOrderbook(symbol: string, depth?: number): Promise<DeltaExchange.Orderbook>;
    /**
     * Gets the user's account information
     * @returns {Promise<DeltaExchange.AccountInfo>} Account information
     */
    getAccountInfo(): Promise<DeltaExchange.AccountInfo>;
    /**
     * Gets the user's wallet balances
     * @returns {Promise<DeltaExchange.WalletBalance[]>} Wallet balances
     */
    getWalletBalances(): Promise<DeltaExchange.WalletBalance[]>;
    /**
     * Gets the user's active positions
     * @returns {Promise<DeltaExchange.Position[]>} Active positions
     */
    getPositions(): Promise<DeltaExchange.Position[]>;
    /**
     * Gets the user's active orders
     * @param {Record<string, any>} params - Query parameters
     * @returns {Promise<DeltaExchange.Order[]>} Active orders
     */
    getActiveOrders(params?: Record<string, any>): Promise<DeltaExchange.Order[]>;
    /**
     * Places a new order
     * @param {DeltaExchange.OrderParams} order - Order details
     * @returns {Promise<DeltaExchange.Order>} Order information
     */
    placeOrder(order: DeltaExchange.OrderParams): Promise<DeltaExchange.Order>;
    /**
     * Cancels an order
     * @param {string} orderId - Order ID to cancel
     * @returns {Promise<any>} Cancellation response
     */
    cancelOrder(orderId: string): Promise<any>;
    /**
     * Cancels all active orders
     * @param {DeltaExchange.CancelAllOrdersParams} params - Filter parameters
     * @returns {Promise<any>} Cancellation response
     */
    cancelAllOrders(params?: DeltaExchange.CancelAllOrdersParams): Promise<any>;
    /**
     * Gets order history for the user
     * @param {DeltaExchange.OrderHistoryParams} params - Query parameters
     * @returns {Promise<DeltaExchange.Order[]>} Order history
     */
    getOrderHistory(params?: DeltaExchange.OrderHistoryParams): Promise<DeltaExchange.Order[]>;
    /**
     * Gets trade history for the user
     * @param {DeltaExchange.TradeHistoryParams} params - Query parameters
     * @returns {Promise<DeltaExchange.Trade[]>} Trade history
     */
    getTradeHistory(params?: DeltaExchange.TradeHistoryParams): Promise<DeltaExchange.Trade[]>;
    /**
     * Makes a request to the Delta Exchange API with retries and rate limit handling
     * @private
     * @param {RequestOptions} options - Request options
     * @param {number} retryCount - Current retry count
     * @returns {Promise<any>} API response
     */
    private _makeRequest;
    /**
     * Adds authentication headers to a request
     * @private
     * @param {AxiosRequestConfig} requestConfig - Axios request configuration
     */
    private _addAuthHeaders;
    /**
     * Logs a request
     * @private
     * @param {AxiosRequestConfig} request - Request configuration
     */
    private _logRequest;
    /**
     * Logs a response
     * @private
     * @param {AxiosResponse} response - Axios response
     */
    private _logResponse;
    /**
     * Logs an error
     * @private
     * @param {any} error - Axios error
     */
    private _logError;
}
export default DeltaExchangeAPI;
