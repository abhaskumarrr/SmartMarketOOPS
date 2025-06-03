/**
 * Real-Time WebSocket Service for SmartMarketOOPS
 * Task #30: Real-Time Trading Dashboard
 * Handles real-time market data, trading signals, and system updates
 */

import { useAuthStore } from '../stores/authStore';

export interface MarketDataUpdate {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high24h: number;
  low24h: number;
  timestamp: number;
}

export interface TradingSignalUpdate {
  id: string;
  symbol: string;
  signal_type: 'buy' | 'sell' | 'hold' | 'strong_buy' | 'strong_sell';
  confidence: number;
  quality: 'excellent' | 'good' | 'fair' | 'poor';
  price: number;
  timestamp: number;
  transformer_prediction: number;
  ensemble_prediction: number;
  smc_score: number;
  technical_score: number;
  stop_loss?: number;
  take_profit?: number;
  position_size?: number;
  risk_reward_ratio?: number;
}

export interface PortfolioUpdate {
  totalValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  positions: Record<string, {
    symbol: string;
    amount: number;
    averagePrice: number;
    currentPrice: number;
    pnl: number;
    pnlPercent: number;
  }>;
}

export interface SystemUpdate {
  type: 'model_update' | 'system_status' | 'performance_metric';
  data: any;
  timestamp: number;
}

export type WebSocketMessage =
  | { type: 'market_data'; data: MarketDataUpdate }
  | { type: 'trading_signal'; data: TradingSignalUpdate }
  | { type: 'portfolio_update'; data: PortfolioUpdate }
  | { type: 'system_update'; data: SystemUpdate }
  | { type: 'error'; data: { message: string; code?: string } }
  | { type: 'connection_status'; data: { status: 'connected' | 'disconnected' | 'reconnecting' } };

export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private maxReconnectDelay = 30000; // Max 30 seconds
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private messageHandlers = new Map<string, Set<(data: any) => void>>();
  private isConnecting = false;
  private shouldReconnect = true;

  constructor(private baseUrl: string = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001') {}

  /**
   * Connect to WebSocket server with authentication
   */
  async connect(): Promise<void> {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return;
    }

    this.isConnecting = true;
    this.shouldReconnect = true;

    try {
      // Get authentication token
      const authStore = useAuthStore.getState();
      const token = authStore.token;

      if (!token) {
        console.log('No authentication token, connecting to WebSocket without auth for real-time data');
        // Continue without authentication for real-time market data
      }

      // Create WebSocket connection (simplified for demo)
      const wsUrl = this.baseUrl;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      this.ws.onerror = this.handleError.bind(this);

    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.isConnecting = false;
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.shouldReconnect = false;

    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this.isConnecting = false;
    this.reconnectAttempts = 0;
  }

  /**
   * Subscribe to specific message types
   */
  subscribe(messageType: string, handler: (data: any) => void): () => void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, new Set());
    }

    this.messageHandlers.get(messageType)!.add(handler);

    // Return unsubscribe function
    return () => {
      const handlers = this.messageHandlers.get(messageType);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          this.messageHandlers.delete(messageType);
        }
      }
    };
  }

  /**
   * Send message to server
   */
  send(message: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  }

  /**
   * Subscribe to market data for specific symbols
   */
  subscribeToMarketData(symbols: string[]): void {
    this.send({
      type: 'subscribe',
      data: {
        channel: 'market_data',
        symbols
      }
    });
  }

  /**
   * Subscribe to trading signals
   */
  subscribeToTradingSignals(symbols?: string[]): void {
    this.send({
      type: 'subscribe',
      data: {
        channel: 'trading_signals',
        symbols: symbols || ['all']
      }
    });
  }

  /**
   * Subscribe to portfolio updates
   */
  subscribeToPortfolio(): void {
    this.send({
      type: 'subscribe',
      data: {
        channel: 'portfolio'
      }
    });
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): 'connected' | 'connecting' | 'disconnected' {
    if (this.isConnecting) return 'connecting';
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return 'connected';
    return 'disconnected';
  }

  private handleOpen(): void {
    console.log('WebSocket connected');
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    this.reconnectDelay = 1000;

    // Start heartbeat
    this.startHeartbeat();

    // Notify connection status
    this.notifyHandlers('connection_status', { status: 'connected' });
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);

      // Handle heartbeat response
      if ((message as any).type === 'pong') {
        return;
      }

      // Notify specific handlers
      this.notifyHandlers(message.type, message.data);

      // Notify all message handlers
      this.notifyHandlers('*', message);

    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket disconnected:', event.code, event.reason);
    this.isConnecting = false;
    this.ws = null;

    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Notify connection status
    this.notifyHandlers('connection_status', { status: 'disconnected' });

    // Attempt reconnection if needed
    if (this.shouldReconnect && event.code !== 1000) {
      this.scheduleReconnect();
    }
  }

  private handleError(error: Event): void {
    console.error('WebSocket error:', error);
    this.isConnecting = false;
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.notifyHandlers('error', {
        message: 'Failed to reconnect to server after multiple attempts',
        code: 'MAX_RECONNECT_ATTEMPTS'
      });
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), this.maxReconnectDelay);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    this.notifyHandlers('connection_status', { status: 'reconnecting' });

    setTimeout(() => {
      if (this.shouldReconnect) {
        this.connect();
      }
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000); // Send ping every 30 seconds
  }

  private notifyHandlers(messageType: string, data: any): void {
    const handlers = this.messageHandlers.get(messageType);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('Error in message handler:', error);
        }
      });
    }
  }
}

// Singleton instance
export const webSocketService = new WebSocketService();

// Auto-connect when authenticated
if (typeof window !== 'undefined') {
  const authStore = useAuthStore.getState();
  if (authStore.isAuthenticated) {
    webSocketService.connect();
  }

  // Listen for auth state changes
  useAuthStore.subscribe((state) => {
    if (state.isAuthenticated && webSocketService.getConnectionStatus() === 'disconnected') {
      webSocketService.connect();
    } else if (!state.isAuthenticated) {
      webSocketService.disconnect();
    }
  });
}
