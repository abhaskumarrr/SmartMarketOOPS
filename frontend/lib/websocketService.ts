import { io, Socket } from 'socket.io-client';
import { EventEmitter } from 'events';

// Event types that can be subscribed to
export type WebSocketEventType = 
  | 'market:data'    // Price and candle data
  | 'trade:executed' // Trade execution notifications
  | 'prediction:new' // New ML predictions
  | 'signal:new'     // New trading signals
  | 'alert:triggered'// Alert triggers
  | 'status:update'; // Bot status updates

// Connection status
export enum ConnectionStatus {
  DISCONNECTED = 'disconnected',
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  ERROR = 'error'
}

interface WebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnectionAttempts?: number;
  reconnectionDelay?: number;
  pingInterval?: number;
  pingTimeout?: number;
}

class WebSocketService extends EventEmitter {
  private socket: Socket | null = null;
  private status: ConnectionStatus = ConnectionStatus.DISCONNECTED;
  private subscriptions: Set<string> = new Set();
  private reconnectAttempts: number = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private pingTimer: NodeJS.Timeout | null = null;
  private pingTimeoutTimer: NodeJS.Timeout | null = null;
  private lastPongTime: number = 0;
  private options: WebSocketOptions;
  private static instance: WebSocketService;
  private reconnecting: boolean = false;
  private initialConnectionAttempt: boolean = true;

  private constructor(options: WebSocketOptions = {}) {
    super();
    this.options = {
      url: options.url || (process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:3001'),
      autoConnect: options.autoConnect !== undefined ? options.autoConnect : true,
      reconnectionAttempts: options.reconnectionAttempts || 10,
      reconnectionDelay: options.reconnectionDelay || 1000,
      pingInterval: options.pingInterval || 30000,
      pingTimeout: options.pingTimeout || 5000
    };

    if (this.options.autoConnect) {
      this.connect();
    }
  }

  /**
   * Get the singleton instance of WebSocketService
   */
  public static getInstance(options?: WebSocketOptions): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService(options);
    }
    return WebSocketService.instance;
  }

  /**
   * Connect to the WebSocket server
   */
  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket && this.status === ConnectionStatus.CONNECTED) {
        resolve();
        return;
      }

      // If already connecting, don't try to connect again
      if (this.status === ConnectionStatus.CONNECTING && !this.initialConnectionAttempt) {
        reject(new Error('Already attempting to connect'));
        return;
      }

      this.initialConnectionAttempt = false;
      this.setStatus(ConnectionStatus.CONNECTING);

      // Cleanup any existing socket
      this.cleanup();

      console.log('Connecting to WebSocket server:', this.options.url);
      this.socket = io(this.options.url as string, {
        transports: ['websocket'],
        reconnection: false, // We'll handle reconnection ourselves for more control
        timeout: 10000,
        forceNew: true,
      });

      // Setup event handlers
      this.socket.on('connect', () => {
        console.log('WebSocket connected');
        this.setStatus(ConnectionStatus.CONNECTED);
        this.reconnectAttempts = 0;
        this.reconnecting = false;
        
        // Resubscribe to previously subscribed channels
        this.resubscribe();
        
        // Start ping-pong health check
        this.startPingPong();
        
        resolve();
      });

      this.socket.on('disconnect', (reason) => {
        console.log(`WebSocket disconnected: ${reason}`);
        this.setStatus(ConnectionStatus.DISCONNECTED);
        this.stopPingPong();
        
        // Some disconnect reasons should trigger reconnection
        if (reason === 'io server disconnect' || reason === 'transport close' || reason === 'transport error') {
          this.handleReconnect();
        }
      });

      this.socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        this.setStatus(ConnectionStatus.ERROR);
        this.stopPingPong();
        this.handleReconnect();
        if (!this.reconnecting) {
          reject(error);
        }
      });

      this.socket.on('error', (error) => {
        console.error('WebSocket error:', error);
        this.setStatus(ConnectionStatus.ERROR);
        this.stopPingPong();
        reject(error);
      });

      // Handle pong responses
      this.socket.on('pong', () => {
        this.lastPongTime = Date.now();
        if (this.pingTimeoutTimer) {
          clearTimeout(this.pingTimeoutTimer);
          this.pingTimeoutTimer = null;
        }
      });

      // Setup data event handlers
      this.socket.on('market:data', (data) => this.emit('market:data', data));
      this.socket.on('trade:executed', (data) => this.emit('trade:executed', data));
      this.socket.on('prediction:new', (data) => this.emit('prediction:new', data));
      this.socket.on('signal:new', (data) => this.emit('signal:new', data));
      this.socket.on('alert:triggered', (data) => this.emit('alert:triggered', data));
      this.socket.on('status:update', (data) => this.emit('status:update', data));
    });
  }

  /**
   * Disconnect from the WebSocket server
   */
  public disconnect(): void {
    this.cleanup();
    this.setStatus(ConnectionStatus.DISCONNECTED);
  }

  /**
   * Clean up socket and timers
   */
  private cleanup(): void {
    this.stopPingPong();
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    if (this.socket) {
      // Remove all listeners to prevent memory leaks
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }
  }

  /**
   * Start ping-pong health check
   */
  private startPingPong(): void {
    this.stopPingPong(); // Clear existing timers if any
    
    this.lastPongTime = Date.now();
    
    this.pingTimer = setInterval(() => {
      if (!this.socket || this.status !== ConnectionStatus.CONNECTED) {
        this.stopPingPong();
        return;
      }
      
      // Send ping
      this.socket.emit('ping');
      
      // Set timeout for pong response
      this.pingTimeoutTimer = setTimeout(() => {
        console.warn('WebSocket ping timeout, no pong received');
        
        // If no pong received for too long, consider connection dead
        const timeSinceLastPong = Date.now() - this.lastPongTime;
        if (timeSinceLastPong > this.options.pingInterval! * 2) {
          console.error('WebSocket connection appears dead, reconnecting...');
          this.cleanup();
          this.setStatus(ConnectionStatus.ERROR);
          this.handleReconnect();
        }
      }, this.options.pingTimeout);
    }, this.options.pingInterval);
  }

  /**
   * Stop ping-pong health check
   */
  private stopPingPong(): void {
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    
    if (this.pingTimeoutTimer) {
      clearTimeout(this.pingTimeoutTimer);
      this.pingTimeoutTimer = null;
    }
  }

  /**
   * Subscribe to a specific data channel
   */
  public subscribe(channel: string, data?: any): void {
    if (!this.socket || this.status !== ConnectionStatus.CONNECTED) {
      // Store subscription for later when connected
      this.subscriptions.add(channel);
      
      // Connect if not already connecting/connected
      if (this.status === ConnectionStatus.DISCONNECTED) {
        this.connect().catch(error => {
          console.error('Failed to connect when subscribing:', error);
        });
      }
      return;
    }

    console.log(`Subscribing to ${channel}`);
    this.socket.emit(`subscribe:${channel}`, data);
    this.subscriptions.add(channel);
  }

  /**
   * Unsubscribe from a specific data channel
   */
  public unsubscribe(channel: string, data?: any): void {
    if (!this.socket || this.status !== ConnectionStatus.CONNECTED) {
      this.subscriptions.delete(channel);
      return;
    }

    console.log(`Unsubscribing from ${channel}`);
    this.socket.emit(`unsubscribe:${channel}`, data);
    this.subscriptions.delete(channel);
  }

  /**
   * Get the current connection status
   */
  public getStatus(): ConnectionStatus {
    return this.status;
  }

  /**
   * Helper to update the status and emit events
   */
  private setStatus(status: ConnectionStatus): void {
    if (this.status !== status) {
      this.status = status;
      this.emit('status', status);
    }
  }

  /**
   * Handle reconnection logic with exponential backoff
   */
  private handleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    if (this.reconnectAttempts >= (this.options.reconnectionAttempts || 10)) {
      console.error('Max reconnection attempts reached');
      this.setStatus(ConnectionStatus.ERROR);
      this.reconnecting = false;
      return;
    }

    this.reconnectAttempts++;
    this.reconnecting = true;
    
    // Calculate delay with exponential backoff: base_delay * (2^attempt) with a max
    const baseDelay = this.options.reconnectionDelay || 1000;
    let delay = Math.min(baseDelay * Math.pow(1.5, this.reconnectAttempts - 1), 30000);
    
    // Add some randomness to prevent simultaneous reconnection attempts from many clients
    delay = delay * (0.8 + Math.random() * 0.4); // +/- 20%
    
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.options.reconnectionAttempts}) in ${Math.round(delay)}ms...`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  /**
   * Resubscribe to all previously subscribed channels
   */
  private resubscribe(): void {
    if (!this.socket || this.status !== ConnectionStatus.CONNECTED) {
      return;
    }

    this.subscriptions.forEach(channel => {
      console.log(`Resubscribing to ${channel}`);
      this.socket?.emit(`subscribe:${channel}`);
    });
  }

  /**
   * Force reconnection regardless of current status
   * This can be used for manual reconnection attempts
   */
  public forceReconnect(): Promise<void> {
    // Reset reconnection attempts to give it a fresh start
    this.reconnectAttempts = 0;
    
    // Clean up existing connection
    this.cleanup();
    
    // Set status to disconnected to ensure we actually try to connect
    this.setStatus(ConnectionStatus.DISCONNECTED);
    
    // Attempt reconnection
    return this.connect();
  }

  /**
   * Check if the connection is healthy and reconnect if necessary
   * This can be called periodically as a backup to the ping-pong mechanism
   */
  public checkConnection(): void {
    if (this.status === ConnectionStatus.CONNECTED) {
      // If we think we're connected, verify by checking last pong time
      const timeSinceLastPong = Date.now() - this.lastPongTime;
      const maxInactiveTime = this.options.pingInterval! * 1.5;
      
      if (timeSinceLastPong > maxInactiveTime && this.pingTimer) {
        console.warn(`WebSocket connection might be stale. Last pong was ${timeSinceLastPong}ms ago`);
        // Force reconnection as a precaution
        this.cleanup();
        this.setStatus(ConnectionStatus.ERROR);
        this.handleReconnect();
      }
    } else if (this.status === ConnectionStatus.DISCONNECTED && !this.reconnectTimer) {
      // If we're disconnected and not already trying to reconnect, attempt reconnection
      this.handleReconnect();
    }
  }
}

export default WebSocketService; 