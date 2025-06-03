/**
 * Enhanced Real-Time Trading Store using Zustand
 * Task #30: Real-Time Trading Dashboard
 * Optimized for M2 MacBook Air 8GB development with WebSocket integration
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { webSocketService, type MarketDataUpdate, type TradingSignalUpdate, type PortfolioUpdate } from '../services/websocket';
import { mlIntelligenceService, type MLIntelligenceData, type MLPerformanceMetrics } from '../services/mlIntelligenceService';

// Types
interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high24h: number;
  low24h: number;
  timestamp: number;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  amount: number;
  price: number;
  timestamp: number;
  status: 'pending' | 'filled' | 'cancelled';
  profit?: number;
}

interface Portfolio {
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

interface MLPrediction {
  symbol: string;
  prediction: 'buy' | 'sell' | 'hold';
  confidence: number;
  timestamp: number;
  features: Record<string, number>;
}

interface TradingSignal {
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

interface ConnectionStatus {
  status: 'connected' | 'connecting' | 'disconnected' | 'reconnecting';
  lastConnected?: number;
  reconnectAttempts?: number;
}

interface MLIntelligenceState {
  currentIntelligence: Record<string, MLIntelligenceData>;
  performanceMetrics: MLPerformanceMetrics | null;
  intelligenceHistory: MLIntelligenceData[];
  isMLConnected: boolean;
  lastMLUpdate: number;
}

interface TradingState {
  // Market data
  marketData: Record<string, MarketData>;
  selectedSymbol: string;

  // Trading
  trades: Trade[];
  portfolio: Portfolio;

  // ML predictions and signals
  predictions: Record<string, MLPrediction>;
  tradingSignals: TradingSignal[];
  latestSignals: Record<string, TradingSignal>;

  // Real-time connection
  connectionStatus: ConnectionStatus;

  // ML Intelligence
  mlIntelligence: MLIntelligenceState;

  // UI state
  isConnected: boolean;
  loading: boolean;
  error: string | null;

  // Performance metrics
  performanceMetrics: {
    totalSignals: number;
    successfulSignals: number;
    averageConfidence: number;
    winRate: number;
    totalReturn: number;
  };

  // Settings
  settings: {
    autoTrade: boolean;
    riskLevel: 'low' | 'medium' | 'high';
    maxPositionSize: number;
    stopLoss: number;
    takeProfit: number;
    enableRealTimeSignals: boolean;
    signalQualityThreshold: 'fair' | 'good' | 'excellent';
  };
}

interface TradingActions {
  // Market data actions
  updateMarketData: (symbol: string, data: Partial<MarketData>) => void;
  updateMarketDataFromWS: (data: MarketDataUpdate) => void;
  setSelectedSymbol: (symbol: string) => void;

  // Trading actions
  addTrade: (trade: Omit<Trade, 'id' | 'timestamp'>) => void;
  updateTrade: (id: string, updates: Partial<Trade>) => void;
  updatePortfolio: (portfolio: Partial<Portfolio>) => void;
  updatePortfolioFromWS: (data: PortfolioUpdate) => void;

  // ML prediction actions
  updatePrediction: (symbol: string, prediction: MLPrediction) => void;
  clearOldPredictions: () => void;

  // Trading signal actions
  addTradingSignal: (signal: TradingSignal) => void;
  updateTradingSignalFromWS: (data: TradingSignalUpdate) => void;
  clearOldSignals: () => void;

  // Real-time connection actions
  updateConnectionStatus: (status: ConnectionStatus) => void;
  initializeWebSocket: () => void;
  disconnectWebSocket: () => void;

  // ML Intelligence actions
  updateMLIntelligence: (symbol: string, intelligence: MLIntelligenceData) => void;
  updateMLPerformanceMetrics: (metrics: MLPerformanceMetrics) => void;
  requestMLIntelligence: (symbol: string) => Promise<void>;
  clearMLIntelligenceHistory: () => void;

  // Performance tracking
  updatePerformanceMetrics: (metrics: Partial<TradingState['performanceMetrics']>) => void;

  // UI actions
  setConnected: (connected: boolean) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Settings actions
  updateSettings: (settings: Partial<TradingState['settings']>) => void;

  // Utility actions
  reset: () => void;
  clearTrades: () => void;
  cleanup: () => void;
}

type TradingStore = TradingState & TradingActions;

// Initial state
const initialState: TradingState = {
  marketData: {},
  selectedSymbol: 'BTCUSD',
  trades: [],
  portfolio: {
    totalValue: 10000, // Starting with $10,000 demo account
    totalPnL: 0,
    totalPnLPercent: 0,
    positions: {},
  },
  predictions: {},
  tradingSignals: [],
  latestSignals: {},
  connectionStatus: {
    status: 'disconnected',
  },
  mlIntelligence: {
    currentIntelligence: {},
    performanceMetrics: null,
    intelligenceHistory: [],
    isMLConnected: false,
    lastMLUpdate: 0,
  },
  isConnected: false,
  loading: false,
  error: null,
  performanceMetrics: {
    totalSignals: 0,
    successfulSignals: 0,
    averageConfidence: 0,
    winRate: 0,
    totalReturn: 0,
  },
  settings: {
    autoTrade: false,
    riskLevel: 'medium',
    maxPositionSize: 1000,
    stopLoss: 2, // 2%
    takeProfit: 5, // 5%
    enableRealTimeSignals: true,
    signalQualityThreshold: 'good',
  },
};

// Create the store with memory-efficient middleware
export const useTradingStore = create<TradingStore>()(
  devtools(
    persist(
      immer((set, get) => ({
        ...initialState,

        // Market data actions
        updateMarketData: (symbol: string, data: Partial<MarketData>) =>
          set((state) => {
            state.marketData[symbol] = {
              ...state.marketData[symbol],
              ...data,
              symbol,
              timestamp: Date.now(),
            };
          }),

        updateMarketDataFromWS: (data: MarketDataUpdate) =>
          set((state) => {
            state.marketData[data.symbol] = {
              symbol: data.symbol,
              price: data.price,
              change: data.change,
              changePercent: data.changePercent,
              volume: data.volume,
              high24h: data.high24h,
              low24h: data.low24h,
              timestamp: data.timestamp,
            };
          }),

        setSelectedSymbol: (symbol: string) =>
          set((state) => {
            state.selectedSymbol = symbol;
          }),

        // Trading actions
        addTrade: (trade: Omit<Trade, 'id' | 'timestamp'>) =>
          set((state) => {
            const newTrade: Trade = {
              ...trade,
              id: `trade_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
              timestamp: Date.now(),
            };
            state.trades.unshift(newTrade); // Add to beginning for recent trades first

            // Keep only last 100 trades for memory efficiency
            if (state.trades.length > 100) {
              state.trades = state.trades.slice(0, 100);
            }
          }),

        updateTrade: (id: string, updates: Partial<Trade>) =>
          set((state) => {
            const tradeIndex = state.trades.findIndex(t => t.id === id);
            if (tradeIndex !== -1) {
              state.trades[tradeIndex] = { ...state.trades[tradeIndex], ...updates };
            }
          }),

        updatePortfolio: (portfolio: Partial<Portfolio>) =>
          set((state) => {
            state.portfolio = { ...state.portfolio, ...portfolio };
          }),

        updatePortfolioFromWS: (data: PortfolioUpdate) =>
          set((state) => {
            state.portfolio = {
              totalValue: data.totalValue,
              totalPnL: data.totalPnL,
              totalPnLPercent: data.totalPnLPercent,
              positions: data.positions,
            };
          }),

        // ML prediction actions
        updatePrediction: (symbol: string, prediction: MLPrediction) =>
          set((state) => {
            state.predictions[symbol] = prediction;
          }),

        clearOldPredictions: () =>
          set((state) => {
            const now = Date.now();
            const maxAge = 5 * 60 * 1000; // 5 minutes

            Object.keys(state.predictions).forEach(symbol => {
              if (now - state.predictions[symbol].timestamp > maxAge) {
                delete state.predictions[symbol];
              }
            });
          }),

        // Trading signal actions
        addTradingSignal: (signal: TradingSignal) =>
          set((state) => {
            state.tradingSignals.unshift(signal);
            state.latestSignals[signal.symbol] = signal;

            // Keep only last 100 signals for memory efficiency
            if (state.tradingSignals.length > 100) {
              state.tradingSignals = state.tradingSignals.slice(0, 100);
            }

            // Update performance metrics
            state.performanceMetrics.totalSignals += 1;
            const totalConfidence = state.tradingSignals.reduce((sum, s) => sum + s.confidence, 0);
            state.performanceMetrics.averageConfidence = totalConfidence / state.tradingSignals.length;
          }),

        updateTradingSignalFromWS: (data: TradingSignalUpdate) =>
          set((state) => {
            const signal: TradingSignal = {
              id: data.id,
              symbol: data.symbol,
              signal_type: data.signal_type,
              confidence: data.confidence,
              quality: data.quality,
              price: data.price,
              timestamp: data.timestamp,
              transformer_prediction: data.transformer_prediction,
              ensemble_prediction: data.ensemble_prediction,
              smc_score: data.smc_score,
              technical_score: data.technical_score,
              stop_loss: data.stop_loss,
              take_profit: data.take_profit,
              position_size: data.position_size,
              risk_reward_ratio: data.risk_reward_ratio,
            };

            // Only add if signal quality meets threshold
            const qualityOrder = { poor: 0, fair: 1, good: 2, excellent: 3 };
            const thresholdOrder = qualityOrder[state.settings.signalQualityThreshold];
            const signalOrder = qualityOrder[signal.quality];

            if (signalOrder >= thresholdOrder) {
              state.tradingSignals.unshift(signal);
              state.latestSignals[signal.symbol] = signal;

              // Keep only last 100 signals for memory efficiency
              if (state.tradingSignals.length > 100) {
                state.tradingSignals = state.tradingSignals.slice(0, 100);
              }

              // Update performance metrics
              state.performanceMetrics.totalSignals += 1;
              const totalConfidence = state.tradingSignals.reduce((sum, s) => sum + s.confidence, 0);
              state.performanceMetrics.averageConfidence = totalConfidence / state.tradingSignals.length;
            }
          }),

        clearOldSignals: () =>
          set((state) => {
            const now = Date.now();
            const maxAge = 24 * 60 * 60 * 1000; // 24 hours

            state.tradingSignals = state.tradingSignals.filter(signal =>
              now - signal.timestamp <= maxAge
            );

            // Update latest signals
            Object.keys(state.latestSignals).forEach(symbol => {
              if (now - state.latestSignals[symbol].timestamp > maxAge) {
                delete state.latestSignals[symbol];
              }
            });
          }),

        // Real-time connection actions
        updateConnectionStatus: (status: ConnectionStatus) =>
          set((state) => {
            state.connectionStatus = status;
            state.isConnected = status.status === 'connected';
          }),

        initializeWebSocket: () => {
          const state = get();

          // Subscribe to WebSocket events
          webSocketService.subscribe('connection_status', (data) => {
            get().updateConnectionStatus({
              status: data.status,
              lastConnected: data.status === 'connected' ? Date.now() : state.connectionStatus.lastConnected,
            });
          });

          webSocketService.subscribe('market_data', (data: MarketDataUpdate) => {
            get().updateMarketDataFromWS(data);
          });

          webSocketService.subscribe('trading_signal', (data: TradingSignalUpdate) => {
            if (state.settings.enableRealTimeSignals) {
              get().updateTradingSignalFromWS(data);
            }
          });

          webSocketService.subscribe('portfolio_update', (data: PortfolioUpdate) => {
            get().updatePortfolioFromWS(data);
          });

          webSocketService.subscribe('error', (data) => {
            get().setError(data.message);
          });

          // Connect to WebSocket
          webSocketService.connect();

          // Subscribe to market data for selected symbol
          webSocketService.subscribeToMarketData([state.selectedSymbol]);
          webSocketService.subscribeToTradingSignals();
          webSocketService.subscribeToPortfolio();
        },

        disconnectWebSocket: () => {
          webSocketService.disconnect();
          set((state) => {
            state.connectionStatus = { status: 'disconnected' };
            state.isConnected = false;
          });
        },

        // ML Intelligence actions
        updateMLIntelligence: (symbol: string, intelligence: MLIntelligenceData) =>
          set((state) => {
            state.mlIntelligence.currentIntelligence[symbol] = intelligence;
            state.mlIntelligence.intelligenceHistory.unshift(intelligence);
            state.mlIntelligence.lastMLUpdate = Date.now();
            state.mlIntelligence.isMLConnected = true;

            // Keep only last 50 intelligence records for memory efficiency
            if (state.mlIntelligence.intelligenceHistory.length > 50) {
              state.mlIntelligence.intelligenceHistory = state.mlIntelligence.intelligenceHistory.slice(0, 50);
            }
          }),

        updateMLPerformanceMetrics: (metrics: MLPerformanceMetrics) =>
          set((state) => {
            state.mlIntelligence.performanceMetrics = metrics;
          }),

        requestMLIntelligence: async (symbol: string) => {
          const state = get();
          try {
            const intelligence = await mlIntelligenceService.requestIntelligence(
              symbol,
              state.marketData[symbol]
            );

            if (intelligence) {
              get().updateMLIntelligence(symbol, intelligence);
            }
          } catch (error) {
            console.error('Failed to request ML intelligence:', error);
            set((state) => {
              state.mlIntelligence.isMLConnected = false;
            });
          }
        },

        clearMLIntelligenceHistory: () =>
          set((state) => {
            const now = Date.now();
            const maxAge = 24 * 60 * 60 * 1000; // 24 hours

            state.mlIntelligence.intelligenceHistory = state.mlIntelligence.intelligenceHistory.filter(
              intelligence => now - new Date(intelligence.timestamp).getTime() <= maxAge
            );

            // Clear old current intelligence
            Object.keys(state.mlIntelligence.currentIntelligence).forEach(symbol => {
              const intelligence = state.mlIntelligence.currentIntelligence[symbol];
              if (now - new Date(intelligence.timestamp).getTime() > maxAge) {
                delete state.mlIntelligence.currentIntelligence[symbol];
              }
            });
          }),

        // Performance tracking
        updatePerformanceMetrics: (metrics: Partial<TradingState['performanceMetrics']>) =>
          set((state) => {
            state.performanceMetrics = { ...state.performanceMetrics, ...metrics };
          }),

        // UI actions
        setConnected: (connected: boolean) =>
          set((state) => {
            state.isConnected = connected;
          }),

        setLoading: (loading: boolean) =>
          set((state) => {
            state.loading = loading;
          }),

        setError: (error: string | null) =>
          set((state) => {
            state.error = error;
          }),

        // Settings actions
        updateSettings: (settings: Partial<TradingState['settings']>) =>
          set((state) => {
            state.settings = { ...state.settings, ...settings };
          }),

        // Utility actions
        reset: () =>
          set((state) => {
            Object.assign(state, initialState);
          }),

        clearTrades: () =>
          set((state) => {
            state.trades = [];
          }),

        cleanup: () =>
          set((state) => {
            // Clean up old data for memory efficiency
            get().clearOldPredictions();
            get().clearOldSignals();
            get().clearMLIntelligenceHistory();

            // Limit market data to recent entries
            const now = Date.now();
            const maxAge = 24 * 60 * 60 * 1000; // 24 hours

            Object.keys(state.marketData).forEach(symbol => {
              if (now - state.marketData[symbol].timestamp > maxAge) {
                delete state.marketData[symbol];
              }
            });
          }),
      })),
      {
        name: 'trading-store',
        // Only persist essential data to reduce memory usage
        partialize: (state) => ({
          selectedSymbol: state.selectedSymbol,
          settings: state.settings,
          portfolio: state.portfolio,
          // Don't persist market data, trades, or predictions (they're real-time)
        }),
      }
    ),
    {
      name: 'trading-store',
      enabled: process.env.NODE_ENV === 'development',
    }
  )
);

// Selectors for optimized re-renders
export const useMarketData = (symbol?: string) =>
  useTradingStore((state) =>
    symbol ? state.marketData[symbol] : state.marketData
  );

export const useSelectedMarketData = () =>
  useTradingStore((state) => state.marketData[state.selectedSymbol]);

export const useRecentTrades = (limit: number = 10) =>
  useTradingStore((state) => state.trades.slice(0, limit));

export const usePortfolio = () =>
  useTradingStore((state) => state.portfolio);

export const usePrediction = (symbol?: string) =>
  useTradingStore((state) =>
    symbol ? state.predictions[symbol] : state.predictions[state.selectedSymbol]
  );

export const useConnectionStatus = () =>
  useTradingStore((state) => ({
    isConnected: state.isConnected,
    loading: state.loading,
    error: state.error,
  }));

export const useTradingSettings = () =>
  useTradingStore((state) => state.settings);

// Action hooks for better organization
export const useTradingActions = () =>
  useTradingStore((state) => ({
    updateMarketData: state.updateMarketData,
    setSelectedSymbol: state.setSelectedSymbol,
    addTrade: state.addTrade,
    updateTrade: state.updateTrade,
    updatePortfolio: state.updatePortfolio,
    updatePrediction: state.updatePrediction,
    clearOldPredictions: state.clearOldPredictions,
    setConnected: state.setConnected,
    setLoading: state.setLoading,
    setError: state.setError,
    updateSettings: state.updateSettings,
    reset: state.reset,
    clearTrades: state.clearTrades,
  }));

export default useTradingStore;
