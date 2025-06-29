/**
 * Hook for real-time market data through Socket.IO WebSockets
 * Connects to SmartMarketOOPS backend WebSocket server
 */

import { useState, useEffect, useCallback } from 'react';
import { wsService } from '../services/websocket';
import { create } from 'zustand';
import { createMockMarketTick, createMockTrade, createMockPortfolioUpdate } from '../lib/mockDataGenerator';



// Zustand store for real-time data
interface RealTimeStore {
  isConnected: boolean;
  setIsConnected: (connected: boolean) => void;
  lastMarketData: Record<string, MarketTick>;
  setLastMarketData: (symbol: string, data: MarketTick) => void;
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
  lastMarketData: {},
  setLastMarketData: (symbol, data) => set((state) => ({ 
    lastMarketData: { ...state.lastMarketData, [symbol]: data } 
  })),
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
    lastMarketData: {},
    lastTrade: null,
    lastPortfolioUpdate: null
  })
}));

export function useRealTimeData() {
  const mockIntervalRef = useRef<NodeJS.Timeout | null>(null);
  
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

  // Start mock data generation for development
  const startMockData = useCallback(() => {
    if (mockIntervalRef.current) {
      clearInterval(mockIntervalRef.current);
    }

    console.log('Starting mock data generation for development');
    
    mockIntervalRef.current = setInterval(() => {
      // Generate mock market data for multiple symbols
      const symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD'];
      
      symbols.forEach(symbol => {
        const tick = createMockMarketTick(symbol);
        addMarketTick(tick);
        setLastMarketData(symbol, tick);
      });

      // Occasionally generate trade signals
      if (Math.random() > 0.7) {
        const randomSymbol = symbols[Math.floor(Math.random() * symbols.length)];
        const trade = createMockTrade(randomSymbol);
        addTradeSignal(trade);
        setLastTrade(trade);
      }

      // Occasionally generate portfolio updates
      if (Math.random() > 0.8) {
        const portfolioUpdate = createMockPortfolioUpdate(symbols);
        
        addPortfolioUpdate(portfolioUpdate);
        setLastPortfolioUpdate(portfolioUpdate);
      }
    }, 2000); // Update every 2 seconds
  }, [addMarketTick, setLastMarketData, addTradeSignal, setLastTrade, addPortfolioUpdate, setLastPortfolioUpdate]);

  // Connect to WebSocket service
  const connect = useCallback(() => {
    const enableWebSockets = process.env.NEXT_PUBLIC_ENABLE_WEBSOCKETS === 'true';

    if (!enableWebSockets) {
      console.log('WebSockets disabled, using mock data for development');
      setIsConnected(true); // Simulate connection for UI
      startMockData();
      return;
    }

    wsService.connect();

    wsService.on('connect', () => {
      console.log('WebSocket connected successfully');
      setIsConnected(true);
      wsService.emit('subscribe', { channels: ['market_data', 'trade_signals', 'portfolio_updates'] });
      console.log('Subscribed to real-time data channels');
    });

    wsService.on('disconnect', (reason) => {
      console.log(`WebSocket disconnected: ${reason}`);
      setIsConnected(false);
    });

    wsService.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setIsConnected(false);
      console.log('Falling back to mock data due to connection error');
      startMockData();
    });

    wsService.on('marketTick', (data: MarketTick) => {
      addMarketTick(data);
      setLastMarketData(data.symbol, data);
    });

    wsService.on('market_data', (data: MarketTick | MarketTick[]) => {
      if (Array.isArray(data)) {
        data.forEach(tick => {
          addMarketTick(tick);
          setLastMarketData(tick.symbol, tick);
        });
      } else {
        addMarketTick(data);
        setLastMarketData(data.symbol, data);
      }
    });

    wsService.on('tradeSignal', (signal: TradeSignal) => {
      addTradeSignal(signal);
      setLastTrade(signal);
    });

    wsService.on('trade_signal', (signal: TradeSignal) => {
      addTradeSignal(signal);
      setLastTrade(signal);
    });

    wsService.on('portfolioUpdate', (data: PortfolioUpdate) => {
      addPortfolioUpdate(data);
      setLastPortfolioUpdate(data);
    });

    wsService.on('portfolio_update', (data: PortfolioUpdate) => {
      addPortfolioUpdate(data);
      setLastPortfolioUpdate(data);
    });

    wsService.on('market:update', (data: any) => {
      if (data.success && data.data) {
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
            setLastMarketData(marketTick.symbol, marketTick);
          });
        }
      }
    });
  }, [setIsConnected, addMarketTick, addTradeSignal, addPortfolioUpdate, setLastMarketData, setLastTrade, setLastPortfolioUpdate, startMockData]);

  // Disconnect from WebSocket service
  const disconnect = useCallback(() => {
    wsService.disconnect();

    if (mockIntervalRef.current) {
      clearInterval(mockIntervalRef.current);
      mockIntervalRef.current = null;
    }
    
    setIsConnected(false);
    console.log('Disconnected from WebSocket server');
  }, [setIsConnected]);

  // Subscribe to specific market symbols
  const subscribeToSymbol = useCallback((symbol: string) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('subscribe_symbol', { symbol });
      console.log(`Subscribed to symbol: ${symbol}`);
    }
  }, [isConnected]);

  // Unsubscribe from specific market symbols
  const unsubscribeFromSymbol = useCallback((symbol: string) => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit('unsubscribe_symbol', { symbol });
      console.log(`Unsubscribed from symbol: ${symbol}`);
    }
  }, [isConnected]);

  // Initialize connection
  useEffect(() => {
    connect();
    
    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    isMockMode: process.env.NEXT_PUBLIC_ENABLE_WEBSOCKETS !== 'true',
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