/**
 * ReconnectingWebSocket - a robust WebSocket connection wrapper
 * 
 * Provides:
 * - Auto reconnection with exponential backoff
 * - Event-based API (similar to native WebSocket)
 * - Connection timeout detection
 * - Automatic ping/pong health checks
 */

// Connection states
export enum ReadyState {
  CONNECTING = 0,
  OPEN = 1,
  CLOSING = 2,
  CLOSED = 3
}

// Connection close codes
export enum CloseCode {
  NORMAL = 1000,
  GOING_AWAY = 1001,
  PROTOCOL_ERROR = 1002,
  UNSUPPORTED_DATA = 1003,
  NO_STATUS = 1005,
  ABNORMAL = 1006,
  INVALID_FRAME_PAYLOAD_DATA = 1007,
  POLICY_VIOLATION = 1008,
  MESSAGE_TOO_BIG = 1009,
  MISSING_EXTENSION = 1010,
  INTERNAL_ERROR = 1011,
  SERVICE_RESTART = 1012,
  TRY_AGAIN_LATER = 1013,
  BAD_GATEWAY = 1014,
  TLS_HANDSHAKE = 1015
}

interface ReconnectingWebSocketOptions {
  // WebSocket URL
  url: string;
  
  // WebSocket protocols
  protocols?: string | string[];
  
  // Whether to automatically connect on instantiation
  automaticOpen?: boolean;
  
  // Whether to automatically reconnect on close/error
  automaticReconnect?: boolean;
  
  // Initial reconnection delay in ms
  reconnectInterval?: number;
  
  // Maximum reconnection delay in ms
  maxReconnectInterval?: number;
  
  // Reconnection backoff multiplier
  reconnectDecay?: number;
  
  // Maximum number of reconnection attempts (0 = infinite)
  maxReconnectAttempts?: number;
  
  // Connection timeout in ms
  timeoutInterval?: number;
  
  // Binary type ('blob' or 'arraybuffer')
  binaryType?: BinaryType;
  
  // How often to send ping (ms)
  pingInterval?: number;
  
  // How long to wait for pong before considering connection dead (ms)
  pingTimeout?: number;
  
  // Debug mode
  debug?: boolean;
}

// Default options
const defaultOptions: ReconnectingWebSocketOptions = {
  url: '',
  protocols: [],
  automaticOpen: true,
  automaticReconnect: true,
  reconnectInterval: 1000,
  maxReconnectInterval: 30000,
  reconnectDecay: 1.5,
  maxReconnectAttempts: 0,
  timeoutInterval: 2000,
  binaryType: 'blob',
  pingInterval: 30000,
  pingTimeout: 5000,
  debug: false
};

export default class ReconnectingWebSocket {
  // Public properties
  public url: string;
  public readyState: ReadyState;
  public protocol: string;
  public binaryType: BinaryType;
  
  // Private properties
  private ws: WebSocket | null;
  private options: ReconnectingWebSocketOptions;
  private reconnectAttempts: number;
  private forcedClose: boolean;
  private timedOut: boolean;
  private eventListeners: {[key: string]: Array<(ev: Event) => void>};
  private pingTimeoutId: number | null;
  private pingIntervalId: number | null;
  private lastPongTime: number;
  
  constructor(options: Partial<ReconnectingWebSocketOptions>) {
    // Merge options with defaults
    this.options = { ...defaultOptions, ...options };
    
    if (!this.options.url) {
      throw new Error('URL must be provided');
    }
    
    this.url = this.options.url;
    this.readyState = ReadyState.CLOSED;
    this.protocol = '';
    this.binaryType = this.options.binaryType || 'blob';
    
    this.ws = null;
    this.reconnectAttempts = 0;
    this.forcedClose = false;
    this.timedOut = false;
    this.eventListeners = {
      'open': [],
      'close': [],
      'error': [],
      'message': []
    };
    this.pingTimeoutId = null;
    this.pingIntervalId = null;
    this.lastPongTime = 0;
    
    if (this.options.automaticOpen) {
      this.open();
    }
  }
  
  /**
   * Open the WebSocket connection
   */
  public open(): void {
    if (this.ws) {
      return;
    }
    
    this._debug('Opening WebSocket...');
    this.ws = new WebSocket(this.url, this.options.protocols || []);
    this.ws.binaryType = this.binaryType;
    
    this.readyState = ReadyState.CONNECTING;
    this.forcedClose = false;
    this.timedOut = false;
    
    const timeout = setTimeout(() => {
      this._debug('Connection timeout');
      this.timedOut = true;
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
      this.readyState = ReadyState.CLOSED;
      this._dispatchEvent('close', new CloseEvent('close', { code: CloseCode.ABNORMAL, reason: 'Connection timeout', wasClean: false }));
      this._tryReconnect();
    }, this.options.timeoutInterval);
    
    this.ws.onopen = (event: Event) => {
      clearTimeout(timeout);
      this._debug('WebSocket connected');
      this.readyState = ReadyState.OPEN;
      this.reconnectAttempts = 0;
      this._dispatchEvent('open', event);
      this._startPing();
    };
    
    this.ws.onclose = (event: CloseEvent) => {
      clearTimeout(timeout);
      this._stopPing();
      this.ws = null;
      
      if (this.forcedClose) {
        this.readyState = ReadyState.CLOSED;
        this._dispatchEvent('close', event);
      } else {
        this.readyState = ReadyState.CONNECTING;
        this._dispatchEvent('close', event);
        if (!this.timedOut && this.options.automaticReconnect) {
          this._tryReconnect();
        }
      }
    };
    
    this.ws.onerror = (event: Event) => {
      this._debug('WebSocket error:', event);
      this._dispatchEvent('error', event);
    };
    
    this.ws.onmessage = (event: MessageEvent) => {
      this._handleMessage(event);
    };
  }
  
  /**
   * Close the WebSocket connection
   */
  public close(code: number = 1000, reason?: string): void {
    if (!this.ws) {
      return;
    }
    
    this.forcedClose = true;
    this._stopPing();
    
    try {
      this.ws.close(code, reason);
    } catch (e) {
      // Ignore errors
    }
  }
  
  /**
   * Send data over the WebSocket
   */
  public send(data: string | ArrayBuffer | Blob | ArrayBufferView): void {
    if (this.ws && this.readyState === ReadyState.OPEN) {
      this._debug('Sending data:', data);
      this.ws.send(data);
    } else {
      throw new Error('INVALID_STATE_ERR: WebSocket not connected');
    }
  }
  
  /**
   * Add an event listener
   */
  public addEventListener<K extends keyof WebSocketEventMap>(
    type: K, 
    listener: (ev: WebSocketEventMap[K]) => any
  ): void {
    if (!(type in this.eventListeners)) {
      this.eventListeners[type] = [];
    }
    this.eventListeners[type].push(listener as any);
  }
  
  /**
   * Remove an event listener
   */
  public removeEventListener<K extends keyof WebSocketEventMap>(
    type: K, 
    listener: (ev: WebSocketEventMap[K]) => any
  ): void {
    if (!(type in this.eventListeners)) {
      return;
    }
    this.eventListeners[type] = this.eventListeners[type].filter(l => l !== listener);
  }
  
  /**
   * Handle incoming messages
   */
  private _handleMessage(event: MessageEvent): void {
    // Handle ping/pong messages
    if (typeof event.data === 'string' && event.data === '__pong__') {
      this.lastPongTime = Date.now();
      return;
    }
    
    this._dispatchEvent('message', event);
  }
  
  /**
   * Start ping/pong health check
   */
  private _startPing(): void {
    this._stopPing();
    this.lastPongTime = Date.now();
    
    // Send ping periodically
    this.pingIntervalId = window.setInterval(() => {
      if (this.readyState === ReadyState.OPEN) {
        this._debug('Sending ping');
        if (this.ws) {
          this.ws.send('__ping__');
        }
        
        // Set timeout for pong response
        this.pingTimeoutId = window.setTimeout(() => {
          const timeSinceLastPong = Date.now() - this.lastPongTime;
          
          if (timeSinceLastPong > this.options.pingTimeout!) {
            this._debug('Ping timeout, connection is stale');
            this.close(CloseCode.ABNORMAL, 'Ping timeout');
            this._tryReconnect();
          }
        }, this.options.pingTimeout);
      }
    }, this.options.pingInterval);
  }
  
  /**
   * Stop ping/pong health check
   */
  private _stopPing(): void {
    if (this.pingIntervalId !== null) {
      clearInterval(this.pingIntervalId);
      this.pingIntervalId = null;
    }
    
    if (this.pingTimeoutId !== null) {
      clearTimeout(this.pingTimeoutId);
      this.pingTimeoutId = null;
    }
  }
  
  /**
   * Try to reconnect
   */
  private _tryReconnect(): void {
    if (this.options.maxReconnectAttempts > 0 && this.reconnectAttempts >= this.options.maxReconnectAttempts) {
      this._debug('Max reconnect attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    
    // Calculate reconnect delay with exponential backoff
    const delay = Math.min(
      this.options.reconnectInterval! * Math.pow(this.options.reconnectDecay!, this.reconnectAttempts - 1),
      this.options.maxReconnectInterval!
    );
    
    this._debug(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      if (this.forcedClose) {
        return;
      }
      
      this.open();
    }, delay);
  }
  
  /**
   * Dispatch an event to listeners
   */
  private _dispatchEvent(type: string, event: Event): void {
    if (!(type in this.eventListeners)) {
      return;
    }
    
    const listeners = this.eventListeners[type];
    for (const listener of listeners) {
      try {
        listener(event);
      } catch (e) {
        console.error('Error in event listener:', e);
      }
    }
    
    // Also call on* methods if they exist
    const onMethod = `on${type}` as keyof this;
    if (typeof this[onMethod] === 'function') {
      try {
        (this[onMethod] as Function)(event);
      } catch (e) {
        console.error(`Error in on${type} handler:`, e);
      }
    }
  }
  
  /**
   * Debug logging
   */
  private _debug(...args: any[]): void {
    if (this.options.debug) {
      console.log('ReconnectingWebSocket:', ...args);
    }
  }
} 