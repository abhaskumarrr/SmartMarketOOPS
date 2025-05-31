/**
 * Delta Exchange WebSocket Service
 * Handles real-time data streaming from Delta Exchange WebSocket API
 *
 * References:
 * - Official Delta Exchange WebSocket Documentation: https://docs.delta.exchange/#websocket-api
 */
import { EventEmitter } from 'events';
import * as DeltaExchange from '../types/deltaExchange';
/**
 * WebSocket options interface
 */
interface WebSocketOptions {
    testnet?: boolean;
    reconnect?: {
        maxRetries?: number;
        initialDelay?: number;
        maxDelay?: number;
        factor?: number;
    };
    apiKeys?: DeltaExchange.ApiCredentials;
}
/**
 * Delta Exchange WebSocket client
 * Handles WebSocket connections, subscriptions, and message processing
 * Implements EventEmitter pattern for easy event handling
 */
declare class DeltaExchangeWebSocket extends EventEmitter {
    private testnet;
    private baseUrl;
    private reconnectOptions;
    private apiKeys;
    private ws;
    private connected;
    private reconnecting;
    private reconnectAttempts;
    private reconnectTimer;
    private heartbeatInterval;
    private subscriptions;
    /**
     * Creates a new instance of the Delta Exchange WebSocket client
     * @param {WebSocketOptions} options - Configuration options
     */
    constructor(options?: WebSocketOptions);
    /**
     * Connects to the Delta Exchange WebSocket server
     * @returns {Promise<boolean>} True if connected successfully
     */
    connect(): Promise<boolean>;
    /**
     * Disconnects from the Delta Exchange WebSocket server
     */
    disconnect(): void;
    /**
     * Subscribes to a channel
     * @param {string} channel - Channel to subscribe to (e.g., 'ticker', 'orderbook')
     * @param {string|string[]} [symbols] - Symbols to subscribe to (e.g., 'BTCUSD')
     * @returns {Promise<boolean>} True if subscription successful
     */
    subscribe(channel: string, symbols?: string | string[] | null): Promise<boolean>;
    /**
     * Unsubscribes from a channel
     * @param {string} channel - Channel to unsubscribe from
     * @param {string|string[]} [symbols] - Symbols to unsubscribe from
     * @returns {boolean} True if unsubscription successful
     */
    unsubscribe(channel: string, symbols?: string | string[] | null): boolean;
    /**
     * Re-subscribes to all previously subscribed channels
     * Useful after reconnection
     * @private
     */
    private _resubscribeAll;
    /**
     * Authenticates the WebSocket connection
     * Required for private channels (orders, positions, etc.)
     * @returns {boolean} True if authentication successful
     */
    authenticate(): boolean;
    /**
     * Sends a ping to keep the connection alive
     * @private
     */
    private _sendPing;
    /**
     * Sends a message to the WebSocket server
     * @private
     * @param {Object} message - Message to send
     * @returns {boolean} True if message sent successfully
     */
    private _sendMessage;
    /**
     * Handles WebSocket open event
     * @private
     * @param {Function} resolve - Promise resolve function
     */
    private _handleOpen;
    /**
     * Handles WebSocket message event
     * @private
     * @param {WebSocket.Data} data - Received message data
     */
    private _handleMessage;
    /**
     * Handles WebSocket error event
     * @private
     * @param {Error} error - WebSocket error
     * @param {Function} reject - Promise reject function
     */
    private _handleError;
    /**
     * Handles WebSocket close event
     * @private
     * @param {number} code - Close code
     * @param {string} reason - Close reason
     */
    private _handleClose;
    /**
     * Schedules a reconnection attempt
     * @private
     */
    private _scheduleReconnect;
    /**
     * Creates a subscription key for tracking subscriptions
     * @private
     * @param {string} channel - Channel
     * @param {string|string[]} symbols - Symbols
     * @returns {string} Subscription key
     */
    private _getSubscriptionKey;
}
export default DeltaExchangeWebSocket;
