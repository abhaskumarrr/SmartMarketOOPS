'use client';

import { io, Socket } from 'socket.io-client';
import { MarketData, Position, Trade } from '@/types/trading';

type WebSocketEventCallback = (data: any) => void;

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connect(): void {
    if (this.socket?.connected) {
      console.log('🔌 WebSocket already connected');
      return;
    }

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:3006';
    console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);

    try {
      this.socket = io(wsUrl, {
        transports: ['websocket', 'polling'],
        autoConnect: true,
        timeout: 5000,
        forceNew: true,
      });

      this.socket.on('connect', () => {
        console.log('✅ WebSocket connected successfully');
        this.reconnectAttempts = 0;
      });

      this.socket.on('disconnect', (reason) => {
        console.log(`❌ WebSocket disconnected: ${reason}`);
        if (reason !== 'io client disconnect') {
          this.handleReconnect();
        }
      });

      this.socket.on('connect_error', (error) => {
        console.error('❌ WebSocket connection error:', error);
        this.handleReconnect();
      });
    } catch (error) {
      console.error('❌ Failed to initialize WebSocket:', error);
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // Market data subscriptions
  subscribeToMarketData(callback: (data: MarketData) => void): void {
    this.socket?.on('market_data', callback);
  }

  unsubscribeFromMarketData(): void {
    this.socket?.off('market_data');
  }

  // Portfolio subscriptions
  subscribeToPortfolioUpdates(callback: (data: any) => void): void {
    this.socket?.on('portfolio_update', callback);
  }

  unsubscribeFromPortfolioUpdates(): void {
    this.socket?.off('portfolio_update');
  }

  // Position subscriptions
  subscribeToPositionUpdates(callback: (data: Position) => void): void {
    this.socket?.on('position_update', callback);
  }

  unsubscribeFromPositionUpdates(): void {
    this.socket?.off('position_update');
  }

  // Trade subscriptions
  subscribeToTradeUpdates(callback: (data: Trade) => void): void {
    this.socket?.on('trade_update', callback);
  }

  unsubscribeFromTradeUpdates(): void {
    this.socket?.off('trade_update');
  }

  // AI Model subscriptions
  subscribeToModelPredictions(callback: (data: any) => void): void {
    this.socket?.on('model_prediction', callback);
  }

  unsubscribeFromModelPredictions(): void {
    this.socket?.off('model_prediction');
  }

  // Generic event subscription
  on(event: string, callback: WebSocketEventCallback): void {
    this.socket?.on(event, callback);
  }

  off(event: string): void {
    this.socket?.off(event);
  }

  emit(event: string, data?: any): void {
    this.socket?.emit(event, data);
  }

  get isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

export const wsService = new WebSocketService();
