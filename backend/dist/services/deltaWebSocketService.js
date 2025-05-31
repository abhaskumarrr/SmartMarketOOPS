"use strict";
/**
 * Delta Exchange WebSocket Service
 * Handles real-time data streaming from Delta Exchange WebSocket API
 *
 * References:
 * - Official Delta Exchange WebSocket Documentation: https://docs.delta.exchange/#websocket-api
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const ws_1 = __importDefault(require("ws"));
const crypto_1 = __importDefault(require("crypto"));
const events_1 = require("events");
const logger_1 = require("../utils/logger");
// Create logger
const logger = (0, logger_1.createLogger)('DeltaExchangeWebSocket');
// Environment configuration
const MAINNET_WS_URL = 'wss://socket.delta.exchange';
const TESTNET_WS_URL = 'wss://testnet-socket.delta.exchange';
// Default reconnection settings
const DEFAULT_RECONNECT_OPTIONS = {
    maxRetries: 10,
    initialDelay: 1000, // ms
    maxDelay: 30000, // ms
    factor: 1.5 // exponential backoff factor
};
/**
 * Delta Exchange WebSocket client
 * Handles WebSocket connections, subscriptions, and message processing
 * Implements EventEmitter pattern for easy event handling
 */
class DeltaExchangeWebSocket extends events_1.EventEmitter {
    /**
     * Creates a new instance of the Delta Exchange WebSocket client
     * @param {WebSocketOptions} options - Configuration options
     */
    constructor(options = {}) {
        super();
        this.testnet = options.testnet || false;
        this.baseUrl = this.testnet ? TESTNET_WS_URL : MAINNET_WS_URL;
        this.reconnectOptions = { ...DEFAULT_RECONNECT_OPTIONS, ...(options.reconnect || {}) };
        this.apiKeys = options.apiKeys || null;
        this.ws = null;
        this.connected = false;
        this.reconnecting = false;
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.heartbeatInterval = null;
        this.subscriptions = new Set();
        // Log initialization
        logger.info(`Initializing Delta Exchange WebSocket client with ${this.testnet ? 'testnet' : 'mainnet'} environment`);
    }
    /**
     * Connects to the Delta Exchange WebSocket server
     * @returns {Promise<boolean>} True if connected successfully
     */
    async connect() {
        if (this.connected) {
            logger.info('WebSocket already connected');
            return true;
        }
        return new Promise((resolve, reject) => {
            try {
                logger.info(`Connecting to WebSocket server: ${this.baseUrl}`);
                this.ws = new ws_1.default(this.baseUrl);
                // Set up event handlers
                this.ws.on('open', () => this._handleOpen(resolve));
                this.ws.on('message', (data) => this._handleMessage(data));
                this.ws.on('error', (error) => this._handleError(error, reject));
                this.ws.on('close', (code, reason) => this._handleClose(code, reason));
                // Set a connection timeout
                const connectionTimeout = setTimeout(() => {
                    if (!this.connected) {
                        logger.error('WebSocket connection timeout');
                        reject(new Error('WebSocket connection timeout'));
                        // Clean up the socket to prevent lingering connections
                        if (this.ws) {
                            this.ws.terminate();
                            this.ws = null;
                        }
                    }
                }, 10000); // 10 second connection timeout
                // Clear the timeout if we connect successfully
                this.once('open', () => clearTimeout(connectionTimeout));
            }
            catch (error) {
                logger.error('WebSocket connection error:', error);
                reject(error);
            }
        });
    }
    /**
     * Disconnects from the Delta Exchange WebSocket server
     */
    disconnect() {
        logger.info('Disconnecting from WebSocket server');
        // Clear any pending reconnect attempts
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        // Clear heartbeat interval
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        // Close the connection if it exists
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.connected = false;
        this.reconnecting = false;
        this.reconnectAttempts = 0;
        // Emit disconnect event
        this.emit('disconnect');
    }
    /**
     * Subscribes to a channel
     * @param {string} channel - Channel to subscribe to (e.g., 'ticker', 'orderbook')
     * @param {string|string[]} [symbols] - Symbols to subscribe to (e.g., 'BTCUSD')
     * @returns {Promise<boolean>} True if subscription successful
     */
    async subscribe(channel, symbols = null) {
        if (!this.connected) {
            await this.connect();
        }
        const subscriptionRequest = {
            type: 'subscribe',
            channels: [channel]
        };
        // Add symbols if provided
        if (symbols) {
            subscriptionRequest.symbols = Array.isArray(symbols) ? symbols : [symbols];
        }
        const subscriptionKey = this._getSubscriptionKey(channel, symbols);
        // Send the subscription request
        logger.info(`Subscribing to ${subscriptionKey}`);
        const success = this._sendMessage(subscriptionRequest);
        if (success) {
            // Track this subscription
            this.subscriptions.add(subscriptionKey);
        }
        return success;
    }
    /**
     * Unsubscribes from a channel
     * @param {string} channel - Channel to unsubscribe from
     * @param {string|string[]} [symbols] - Symbols to unsubscribe from
     * @returns {boolean} True if unsubscription successful
     */
    unsubscribe(channel, symbols = null) {
        if (!this.connected) {
            logger.warn('Cannot unsubscribe when not connected');
            return false;
        }
        const unsubscriptionRequest = {
            type: 'unsubscribe',
            channels: [channel]
        };
        // Add symbols if provided
        if (symbols) {
            unsubscriptionRequest.symbols = Array.isArray(symbols) ? symbols : [symbols];
        }
        const subscriptionKey = this._getSubscriptionKey(channel, symbols);
        // Send the unsubscription request
        logger.info(`Unsubscribing from ${subscriptionKey}`);
        const success = this._sendMessage(unsubscriptionRequest);
        if (success) {
            // Remove this subscription from tracking
            this.subscriptions.delete(subscriptionKey);
        }
        return success;
    }
    /**
     * Re-subscribes to all previously subscribed channels
     * Useful after reconnection
     * @private
     */
    _resubscribeAll() {
        if (!this.connected || this.subscriptions.size === 0) {
            return;
        }
        logger.info(`Resubscribing to ${this.subscriptions.size} channels`);
        // Resubscribe to each channel
        for (const subscriptionKey of this.subscriptions) {
            const [channel, symbolsStr] = subscriptionKey.split(':');
            const symbols = symbolsStr ? symbolsStr.split(',') : null;
            // Create a subscription request
            const subscriptionRequest = {
                type: 'subscribe',
                channels: [channel]
            };
            if (symbols) {
                subscriptionRequest.symbols = symbols;
            }
            // Send the request
            this._sendMessage(subscriptionRequest);
        }
    }
    /**
     * Authenticates the WebSocket connection
     * Required for private channels (orders, positions, etc.)
     * @returns {boolean} True if authentication successful
     */
    authenticate() {
        if (!this.connected) {
            logger.warn('Cannot authenticate when not connected');
            return false;
        }
        if (!this.apiKeys) {
            logger.error('Cannot authenticate without API keys');
            return false;
        }
        const timestamp = Math.floor(Date.now());
        const method = 'GET';
        const path = '/ws/authentication';
        // Create signature
        const message = timestamp + method + path;
        const signature = crypto_1.default
            .createHmac('sha256', this.apiKeys.secret)
            .update(message)
            .digest('hex');
        // Create authentication request
        const authRequest = {
            type: 'authenticate',
            api_key: this.apiKeys.key,
            timestamp: timestamp.toString(),
            signature
        };
        // Send authentication request
        logger.info('Authenticating WebSocket connection');
        return this._sendMessage(authRequest);
    }
    /**
     * Sends a ping to keep the connection alive
     * @private
     */
    _sendPing() {
        if (!this.connected) {
            return;
        }
        const pingMessage = { type: 'ping' };
        this._sendMessage(pingMessage);
    }
    /**
     * Sends a message to the WebSocket server
     * @private
     * @param {Object} message - Message to send
     * @returns {boolean} True if message sent successfully
     */
    _sendMessage(message) {
        if (!this.ws || this.ws.readyState !== ws_1.default.OPEN) {
            logger.error('Cannot send message: WebSocket not open');
            return false;
        }
        try {
            const messageStr = JSON.stringify(message);
            this.ws.send(messageStr);
            logger.debug('Sent message', message);
            return true;
        }
        catch (error) {
            logger.error('Error sending message:', error);
            return false;
        }
    }
    /**
     * Handles WebSocket open event
     * @private
     * @param {Function} resolve - Promise resolve function
     */
    _handleOpen(resolve) {
        this.connected = true;
        this.reconnecting = false;
        this.reconnectAttempts = 0;
        logger.info('WebSocket connected');
        // Set up heartbeat interval
        this.heartbeatInterval = setInterval(() => this._sendPing(), 30000);
        // Resubscribe to channels if we were previously subscribed
        if (this.subscriptions.size > 0) {
            this._resubscribeAll();
        }
        // Authenticate if credentials are available
        if (this.apiKeys) {
            this.authenticate();
        }
        // Emit open event
        this.emit('open');
        // Resolve the connect promise if provided
        if (resolve) {
            resolve(true);
        }
    }
    /**
     * Handles WebSocket message event
     * @private
     * @param {WebSocket.Data} data - Received message data
     */
    _handleMessage(data) {
        try {
            // Parse the message
            const message = JSON.parse(data.toString());
            // Handle ping/pong messages
            if (message.type === 'pong') {
                logger.debug('Received pong');
                return;
            }
            // Handle authentication response
            if (message.type === 'authenticated') {
                logger.info('WebSocket authenticated successfully');
                this.emit('authenticated', message);
                return;
            }
            // Handle subscription response
            if (message.type === 'subscribed') {
                logger.info(`Subscribed to ${message.channel}`);
                this.emit('subscribed', message);
                return;
            }
            // Handle unsubscription response
            if (message.type === 'unsubscribed') {
                logger.info(`Unsubscribed from ${message.channel}`);
                this.emit('unsubscribed', message);
                return;
            }
            // Handle error messages
            if (message.type === 'error') {
                logger.error('WebSocket error message:', message);
                this.emit('error', new Error(message.message || 'Unknown WebSocket error'));
                return;
            }
            // Handle data messages
            if (message.channel) {
                // Emit events both for the specific channel and for all messages
                this.emit(message.channel, message);
                if (message.symbol) {
                    this.emit(`${message.channel}:${message.symbol}`, message);
                }
                this.emit('message', message);
                logger.debug(`Received ${message.channel} message:`, {
                    channel: message.channel,
                    symbol: message.symbol,
                    type: message.type
                });
            }
        }
        catch (error) {
            logger.error('Error processing WebSocket message:', error);
            logger.debug('Raw message:', data.toString());
        }
    }
    /**
     * Handles WebSocket error event
     * @private
     * @param {Error} error - WebSocket error
     * @param {Function} reject - Promise reject function
     */
    _handleError(error, reject) {
        logger.error('WebSocket error:', error);
        // Emit error event
        this.emit('error', error);
        // Reject the connect promise if provided
        if (reject) {
            reject(error);
        }
    }
    /**
     * Handles WebSocket close event
     * @private
     * @param {number} code - Close code
     * @param {string} reason - Close reason
     */
    _handleClose(code, reason) {
        this.connected = false;
        // Clear heartbeat interval
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
        logger.info(`WebSocket closed: ${code} - ${reason || 'No reason provided'}`);
        // Emit close event
        this.emit('close', { code, reason });
        // Attempt to reconnect if not deliberately disconnected
        if (!this.reconnecting && code !== 1000) {
            this._scheduleReconnect();
        }
    }
    /**
     * Schedules a reconnection attempt
     * @private
     */
    _scheduleReconnect() {
        if (this.reconnecting || this.reconnectAttempts >= this.reconnectOptions.maxRetries) {
            if (this.reconnectAttempts >= this.reconnectOptions.maxRetries) {
                logger.error(`Maximum reconnection attempts (${this.reconnectOptions.maxRetries}) reached`);
                this.emit('reconnect_failed');
            }
            return;
        }
        this.reconnecting = true;
        this.reconnectAttempts++;
        // Calculate delay with exponential backoff
        const delay = Math.min(this.reconnectOptions.initialDelay * Math.pow(this.reconnectOptions.factor, this.reconnectAttempts - 1), this.reconnectOptions.maxDelay);
        logger.info(`Scheduling reconnection attempt ${this.reconnectAttempts}/${this.reconnectOptions.maxRetries} in ${delay}ms`);
        // Emit reconnecting event
        this.emit('reconnecting', { attempt: this.reconnectAttempts, delay });
        // Schedule reconnection
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            // Attempt to reconnect
            logger.info(`Attempting to reconnect (${this.reconnectAttempts}/${this.reconnectOptions.maxRetries})`);
            this.connect()
                .then(() => {
                logger.info('Reconnection successful');
                this.emit('reconnect');
            })
                .catch(error => {
                logger.error('Reconnection failed:', error);
                if (this.reconnectAttempts < this.reconnectOptions.maxRetries) {
                    this._scheduleReconnect();
                }
                else {
                    logger.error(`Maximum reconnection attempts (${this.reconnectOptions.maxRetries}) reached`);
                    this.emit('reconnect_failed');
                }
            });
        }, delay);
    }
    /**
     * Creates a subscription key for tracking subscriptions
     * @private
     * @param {string} channel - Channel
     * @param {string|string[]} symbols - Symbols
     * @returns {string} Subscription key
     */
    _getSubscriptionKey(channel, symbols) {
        if (!symbols) {
            return channel;
        }
        const symbolsArr = Array.isArray(symbols) ? symbols : [symbols];
        return `${channel}:${symbolsArr.join(',')}`;
    }
}
exports.default = DeltaExchangeWebSocket;
//# sourceMappingURL=deltaWebSocketService.js.map