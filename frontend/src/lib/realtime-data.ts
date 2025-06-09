/**
 * Real-time Data Service
 * Provides WebSocket connections to the backend for real-time market data
 */
import { io, Socket } from 'socket.io-client';
import { create } from 'zustand';
import { initializeMockData, updateMockData } from './mock-data';

// Types
export interface MarketTick {
  symbol: string;
  price: number;
  volume: number;
  timestamp: number;
  change24h: number;
  high24h: number;
  low24h: number;
}

export interface TradeSignal {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  price: number;
  confidence: number;
  timestamp: number;
  indicators: {
    name: string;
    value: number;
    signal: 'buy' | 'sell' | 'neutral';
  }[];
}

export interface PortfolioUpdate {
  totalBalance: number;
  availableBalance: number;
  totalPnL: number;
  totalPnLPercentage: number;
  dayPnL: number;
  dayPnLPercentage: number;
  positions: {
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    pnl: number;
    pnlPercentage: number;
  }[];
  timestamp: number;
}

// Connection state and data store
interface RealtimeDataState {
  socket: Socket | null;
  isConnected: boolean;
  isConnecting: boolean;
  lastError: string | null;
  marketData: Record<string, MarketTick>;
  tradeSignals: TradeSignal[];
  portfolioData: PortfolioUpdate | null;
  usingMockData: boolean;
  mockInterval: number | null;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  startMockDataMode: () => void;
  stopMockDataMode: () => void;
}

export const useRealtimeData = create<RealtimeDataState>((set, get) => ({
  socket: null,
  isConnected: false,
  isConnecting: false,
  lastError: null,
  marketData: {},
  tradeSignals: [],
  portfolioData: null,
  usingMockData: false,
  mockInterval: null,

  connect: () => {
    if (get().socket?.connected || get().isConnecting) return;

    set({ isConnecting: true });
    
    try {
      const socketUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3001';
      console.log(`Connecting to WebSocket at ${socketUrl}`);
      
      const socket = io(socketUrl, {
        reconnectionAttempts: 5,
        reconnectionDelay: 3000,
        timeout: 10000,
        transports: ['websocket'],
      });

      socket.on('connect', () => {
        console.log('✅ WebSocket connected');
        set({ 
          socket, 
          isConnected: true, 
          isConnecting: false,
          lastError: null 
        });

        // Stop mock data if it was running
        get().stopMockDataMode();
      });

      socket.on('disconnect', (reason) => {
        console.log(`❌ WebSocket disconnected: ${reason}`);
        set({ isConnected: false });
        
        // If disconnected and not using mock data, start mock data mode
        if (!get().usingMockData) {
          get().startMockDataMode();
        }
      });

      socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        set({ 
          isConnecting: false, 
          lastError: `Connection failed: ${error.message}` 
        });

        // Start mock data mode
        get().startMockDataMode();
      });

      // Market data updates
      socket.on('marketTick', (data: MarketTick | MarketTick[]) => {
        if (Array.isArray(data)) {
          const marketData = { ...get().marketData };
          data.forEach(tick => {
            marketData[tick.symbol] = tick;
          });
          set({ marketData });
        } else {
          set(state => ({
            marketData: {
              ...state.marketData,
              [data.symbol]: data
            }
          }));
        }
      });

      // Trade signals
      socket.on('tradeSignal', (signal: TradeSignal) => {
        set(state => ({
          tradeSignals: [signal, ...state.tradeSignals.slice(0, 19)]
        }));
      });

      // Portfolio updates
      socket.on('portfolioUpdate', (data: PortfolioUpdate) => {
        set({ portfolioData: data });
      });

      set({ socket });
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
      set({ 
        isConnecting: false, 
        lastError: error instanceof Error ? error.message : 'Unknown error' 
      });

      // Start mock data as fallback
      get().startMockDataMode();
    }
  },

  disconnect: () => {
    const { socket, stopMockDataMode } = get();
    
    // Stop mock data
    stopMockDataMode();
    
    if (socket) {
      socket.disconnect();
      set({ 
        socket: null, 
        isConnected: false,
        isConnecting: false 
      });
    }
  },

  reconnect: () => {
    const { disconnect, connect } = get();
    disconnect();
    setTimeout(connect, 500);
  },

  startMockDataMode: () => {
    const { usingMockData, mockInterval } = get();
    
    // Already using mock data
    if (usingMockData && mockInterval !== null) return;
    
    console.log('📊 Starting mock data mode');
    
    // Initialize mock data
    const { mockMarketData, mockPortfolio, mockTradeSignals } = initializeMockData();
    
    set({
      usingMockData: true,
      marketData: mockMarketData,
      portfolioData: mockPortfolio,
      tradeSignals: mockTradeSignals,
      isConnected: true, // Pretend we're connected
      isConnecting: false,
      lastError: 'Using simulated data (backend unavailable)'
    });
    
    // Set up interval to update mock data
    const interval = window.setInterval(() => {
      const { mockMarketData, mockPortfolio, mockTradeSignals } = updateMockData();
      
      set({
        marketData: mockMarketData,
        portfolioData: mockPortfolio,
        tradeSignals: mockTradeSignals
      });
    }, 3000); // Update every 3 seconds
    
    set({ mockInterval: interval });
  },

  stopMockDataMode: () => {
    const { mockInterval } = get();
    
    if (mockInterval !== null) {
      console.log('📊 Stopping mock data mode');
      window.clearInterval(mockInterval);
      set({ 
        usingMockData: false, 
        mockInterval: null 
      });
    }
  }
}));

// Initialize connection on import (client-side only)
if (typeof window !== 'undefined') {
  // Delay connection to ensure hydration completes
  setTimeout(() => {
    useRealtimeData.getState().connect();
  }, 1000);
}

// Utility hooks for specific data access
export function useMarketTick(symbol: string): MarketTick | null {
  return useRealtimeData(state => state.marketData[symbol] || null);
}

export function useAllMarketData(): Record<string, MarketTick> {
  return useRealtimeData(state => state.marketData);
}

export function useTradeSignals(): TradeSignal[] {
  return useRealtimeData(state => state.tradeSignals);
}

export function usePortfolioData(): PortfolioUpdate | null {
  return useRealtimeData(state => state.portfolioData);
}

export function useRealtimeStatus() {
  const { isConnected, isConnecting, lastError, connect, reconnect, usingMockData } = useRealtimeData();
  return { isConnected, isConnecting, lastError, connect, reconnect, usingMockData };
} 