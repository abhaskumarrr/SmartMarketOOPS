/**
 * Hook for real-time market data through Socket.IO WebSockets
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { create } from 'zustand';
import { io, Socket } from 'socket.io-client';

// Constants for Socket.IO connection
const SOCKET_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:3001';

// Constants for mock data
const BASE_PRICES = {
  'BTCUSD': 48250,
  'ETHUSD': 2870,
  'SOLUSD': 106,
  'BNBUSD': 570,
  'DOTUSD': 7.8,
  'ADAUSD': 0.45,
  'LINKUSD': 14.20,
  'XRPUSD': 0.52,
};

export interface MarketTick {
  symbol: string;
  price: number;
  changePercentage24h: number;
  volume: number;
  timestamp: number;
}

export interface TradeSignal {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  price: number;
  size: number;
  timestamp: number;
  source: 'algorithm' | 'manual' | 'bot';
  strategy?: string;
  confidence?: number;
}

export interface PortfolioUpdate {
  totalBalance: number;
  availableBalance: number;
  totalPnL: number;
  totalPnLPercentage: number;
  positions: {
    symbol: string;
    side: 'long' | 'short';
    size: number;
    entryPrice: number;
    currentPrice: number;
    pnl: number;
    pnlPercentage: number;
  }[];
  timestamp: number;
}

// Helper functions for mock data generation
function createMockMarketTick(symbol: string): MarketTick {
  const basePrice = BASE_PRICES[symbol as keyof typeof BASE_PRICES] || 100;
  return {
    symbol,
    price: basePrice * (1 + (Math.random() - 0.5) * 0.02),
    changePercentage24h: Math.random() * 5 - 2.5,
    volume: 1000000 + Math.random() * 500000,
    timestamp: Date.now()
  };
}

function createMockTrade(symbol: string): TradeSignal {
  const basePrice = BASE_PRICES[symbol as keyof typeof BASE_PRICES] || 100;
  return {
    id: `trade-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    symbol,
    side: Math.random() > 0.5 ? 'buy' : 'sell',
    price: basePrice * (1 + (Math.random() - 0.5) * 0.01),
    size: Math.random() * (symbol === 'BTCUSD' ? 2 : 20),
    timestamp: Date.now(),
    source: ['algorithm', 'manual', 'bot'][Math.floor(Math.random() * 3)] as 'algorithm' | 'manual' | 'bot',
    strategy: ['momentum', 'trend_following', 'mean_reversion', 'breakout'][Math.floor(Math.random() * 4)],
    confidence: Math.random() * 100
  };
}

// Zustand store for real-time data
interface RealTimeStore {
  isConnected: boolean;
  setIsConnected: (connected: boolean) => void;
  lastMarketData: MarketTick | null;
  setLastMarketData: (data: MarketTick) => void;
  lastTrade: TradeSignal | null;
  setLastTrade: (trade: TradeSignal) => void;
  lastPortfolioUpdate: PortfolioUpdate | null;
  setLastPortfolioUpdate: (update: PortfolioUpdate) => void;
  marketTicks: MarketTick[];
  addMarketTick: (tick: MarketTick) => void;
  tradeSignals: TradeSignal[];
  addTradeSignal: (signal: TradeSignal) => void;
  portfolioHistory: PortfolioUpdate[];
  addPortfolioUpdate: (update: PortfolioUpdate) => void;
  clearData: () => void;
}

const useRealTimeStore = create<RealTimeStore>((set) => ({
  isConnected: false,
  setIsConnected: (connected) => set({ isConnected: connected }),
  lastMarketData: null,
  setLastMarketData: (data) => set({ lastMarketData: data }),
  lastTrade: null,
  setLastTrade: (trade) => set({ lastTrade: trade }),
  lastPortfolioUpdate: null,
  setLastPortfolioUpdate: (update) => set({ lastPortfolioUpdate: update }),
  marketTicks: [],
  addMarketTick: (tick) => set((state) => ({ 
    marketTicks: [tick, ...state.marketTicks.slice(0, 99)] 
  })),
  tradeSignals: [],
  addTradeSignal: (signal) => set((state) => ({ 
    tradeSignals: [signal, ...state.tradeSignals.slice(0, 49)] 
  })),
  portfolioHistory: [],
  addPortfolioUpdate: (update) => set((state) => ({ 
    portfolioHistory: [update, ...state.portfolioHistory.slice(0, 99)] 
  })),
  clearData: () => set({
    marketTicks: [],
    tradeSignals: [],
    portfolioHistory: [],
    lastMarketData: null,
    lastTrade: null,
    lastPortfolioUpdate: null
  })
}));

export function useRealTimeData() {
  const socketRef = useRef<Socket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const {
    isConnected,
    setIsConnected,
    lastMarketData,
    setLastMarketData,
    lastTrade,
    setLastTrade,
    lastPortfolioUpdate,
    setLastPortfolioUpdate,
    marketTicks,
    addMarketTick,
    tradeSignals,
    addTradeSignal,
    portfolioHistory,
    addPortfolioUpdate,
    clearData
  } = useRealTimeStore();

  // Connect to Socket.IO server
  const connect = useCallback(() => {
    // Don't connect if WebSockets are disabled
    if (process.env.NEXT_PUBLIC_ENABLE_WEBSOCKETS !== 'true') {
      console.log('WebSockets disabled in environment');
      return;
    }
    
    // Close existing connection
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    // Clear any pending reconnection
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    try {
      console.log(`🔌 Connecting to Socket.IO server at ${SOCKET_URL}`);
      
      // Create Socket.IO connection with configuration
      socketRef.current = io(SOCKET_URL, {
        transports: ['websocket', 'polling'],
        reconnectionAttempts: 5,
        reconnectionDelay: 3000,
        timeout: 10000,
        autoConnect: true
      });
      
      const socket = socketRef.current;
      
      socket.on('connect', () => {
        console.log('✅ Socket.IO connected successfully');
        setIsConnected(true);
        
        // Subscribe to market data channels
        socket.emit('subscribe', {
          channels: ['market_data', 'trade_signals', 'portfolio_updates']
        });
        
        console.log('📡 Subscribed to real-time data channels');
      });
      
      socket.on('disconnect', (reason) => {
        console.log(`❌ Socket.IO disconnected: ${reason}`);
        setIsConnected(false);
        
        // Attempt to reconnect after a delay for certain disconnect reasons
        if (reason === 'io server disconnect') {
          // Server initiated disconnect, don't reconnect automatically
          console.log('Server disconnected, not reconnecting automatically');
        } else {
          // Client initiated or network issues, attempt to reconnect
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log('🔄 Attempting to reconnect Socket.IO...');
            connect();
          }, 5000);
        }
      });
      
      socket.on('connect_error', (error) => {
        console.error('❌ Socket.IO connection error:', error);
        setIsConnected(false);
      });
      
      // Market data updates
      socket.on('marketTick', (data: MarketTick) => {
        console.log('📊 Received market tick:', data.symbol, data.price);
        addMarketTick(data);
        setLastMarketData(data);
      });
      
      socket.on('market_data', (data: MarketTick | MarketTick[]) => {
        if (Array.isArray(data)) {
          data.forEach(tick => {
            addMarketTick(tick);
            setLastMarketData(tick);
          });
          console.log('📊 Received market data batch:', data.length, 'items');
        } else {
          addMarketTick(data);
          setLastMarketData(data);
          console.log('📊 Received market data:', data.symbol, data.price);
        }
      });
      
      // Trade signals
      socket.on('tradeSignal', (signal: TradeSignal) => {
        console.log('📈 Received trade signal:', signal.symbol, signal.side);
        addTradeSignal(signal);
        setLastTrade(signal);
      });
      
      socket.on('trade_signal', (signal: TradeSignal) => {
        console.log('📈 Received trade signal:', signal.symbol, signal.side);
        addTradeSignal(signal);
        setLastTrade(signal);
      });
      
      // Portfolio updates
      socket.on('portfolioUpdate', (data: PortfolioUpdate) => {
        console.log('💼 Received portfolio update:', data.totalBalance);
        addPortfolioUpdate(data);
        setLastPortfolioUpdate(data);
      });
      
      socket.on('portfolio_update', (data: PortfolioUpdate) => {
        console.log('💼 Received portfolio update:', data.totalBalance);
        addPortfolioUpdate(data);
        setLastPortfolioUpdate(data);
      });
      
      // Generic market update handler
      socket.on('market:update', (data: any) => {
        console.log('📡 Received market update:', data);
        if (data.success && data.data) {
          // Handle multiple market data items
          if (Array.isArray(data.data)) {
            data.data.forEach((item: any) => {
              const marketTick: MarketTick = {
                symbol: item.symbol,
                price: item.price,
                changePercentage24h: item.changePercentage24h || item.change || 0,
                volume: item.volume24h || item.volume || 0,
                timestamp: new Date(item.timestamp || Date.now()).getTime()
              };
              addMarketTick(marketTick);
              setLastMarketData(marketTick);
            });
          }
        }
      });
      
    } catch (error) {
      console.error('❌ Failed to create Socket.IO connection:', error);
      setIsConnected(false);
    }
  }, [setIsConnected, addMarketTick, addTradeSignal, addPortfolioUpdate, setLastMarketData, setLastTrade, setLastPortfolioUpdate]);

  // Disconnect from Socket.IO server
  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    setIsConnected(false);
    console.log('🔌 Disconnected from Socket.IO server');
  }, [setIsConnected]);

  // Subscribe to specific market symbols
  const subscribeToSymbol = useCallback((symbol: string) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('subscribe_symbol', { symbol });
      console.log(`📡 Subscribed to symbol: ${symbol}`);
    }
  }, [isConnected]);

  // Unsubscribe from specific market symbols
  const unsubscribeFromSymbol = useCallback((symbol: string) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('unsubscribe_symbol', { symbol });
      console.log(`📡 Unsubscribed from symbol: ${symbol}`);
    }
  }, [isConnected]);

  // Initialize Socket.IO connection
  useEffect(() => {
    connect();
    
    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    isMockMode: false, // No mock mode with real Socket.IO
    lastMarketData,
    lastTrade,
    lastPortfolioUpdate,
    marketTicks,
    tradeSignals,
    portfolioHistory,
    clearData,
    reconnect: connect,
    disconnect,
    subscribeToSymbol,
    unsubscribeFromSymbol
  };
} 