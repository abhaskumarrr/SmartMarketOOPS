/**
 * Real-Time Data Hook
 * Task #30: Real-Time Trading Dashboard
 * Memory-efficient hook for managing real-time data subscriptions
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useTradingStore } from '../stores/tradingStore';
import { webSocketService } from '../services/websocket';

interface UseRealTimeDataOptions {
  symbols?: string[];
  enableSignals?: boolean;
  enablePortfolio?: boolean;
  maxDataPoints?: number;
  cleanupInterval?: number;
}

interface RealTimeDataState {
  isConnected: boolean;
  connectionStatus: string;
  lastUpdate: number;
  dataPoints: number;
  memoryUsage: number;
}

export const useRealTimeData = (options: UseRealTimeDataOptions = {}) => {
  const {
    symbols = ['BTCUSD'],
    enableSignals = true,
    enablePortfolio = true,
    maxDataPoints = 1000,
    cleanupInterval = 5 * 60 * 1000 // 5 minutes
  } = options;

  const [state, setState] = useState<RealTimeDataState>({
    isConnected: false,
    connectionStatus: 'disconnected',
    lastUpdate: 0,
    dataPoints: 0,
    memoryUsage: 0
  });

  // Store references
  const {
    initializeWebSocket,
    disconnectWebSocket,
    cleanup,
    isConnected,
    connectionStatus,
    marketData,
    tradingSignals
  } = useTradingStore();

  // Cleanup timer ref
  const cleanupTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastCleanupRef = useRef<number>(0);

  // Memory monitoring
  const calculateMemoryUsage = useCallback(() => {
    if (typeof window === 'undefined') return 0;
    
    // Estimate memory usage based on data structures
    const marketDataSize = Object.keys(marketData).length * 200; // ~200 bytes per market data entry
    const signalsSize = tradingSignals.length * 500; // ~500 bytes per signal
    const totalSize = marketDataSize + signalsSize;
    
    return totalSize / 1024; // Return in KB
  }, [marketData, tradingSignals]);

  // Update state
  const updateState = useCallback(() => {
    setState(prev => ({
      ...prev,
      isConnected,
      connectionStatus: connectionStatus.status,
      lastUpdate: Date.now(),
      dataPoints: Object.keys(marketData).length + tradingSignals.length,
      memoryUsage: calculateMemoryUsage()
    }));
  }, [isConnected, connectionStatus.status, marketData, tradingSignals, calculateMemoryUsage]);

  // Memory-efficient cleanup
  const performCleanup = useCallback(() => {
    const now = Date.now();
    
    // Only cleanup if enough time has passed
    if (now - lastCleanupRef.current < cleanupInterval) {
      return;
    }

    lastCleanupRef.current = now;
    
    // Perform cleanup
    cleanup();
    
    // Force garbage collection if available (development only)
    if (process.env.NODE_ENV === 'development' && window.gc) {
      window.gc();
    }
  }, [cleanup, cleanupInterval]);

  // Subscribe to real-time data
  const subscribe = useCallback(() => {
    if (!isConnected) return;

    // Subscribe to market data for specified symbols
    webSocketService.subscribeToMarketData(symbols);

    // Subscribe to trading signals if enabled
    if (enableSignals) {
      webSocketService.subscribeToTradingSignals(symbols);
    }

    // Subscribe to portfolio updates if enabled
    if (enablePortfolio) {
      webSocketService.subscribeToPortfolio();
    }
  }, [symbols, enableSignals, enablePortfolio, isConnected]);

  // Initialize WebSocket connection
  useEffect(() => {
    initializeWebSocket();
    
    return () => {
      disconnectWebSocket();
    };
  }, [initializeWebSocket, disconnectWebSocket]);

  // Subscribe to data when connected
  useEffect(() => {
    if (isConnected) {
      subscribe();
    }
  }, [isConnected, subscribe]);

  // Update state when data changes
  useEffect(() => {
    updateState();
  }, [updateState]);

  // Setup cleanup timer
  useEffect(() => {
    cleanupTimerRef.current = setInterval(performCleanup, cleanupInterval);
    
    return () => {
      if (cleanupTimerRef.current) {
        clearInterval(cleanupTimerRef.current);
      }
    };
  }, [performCleanup, cleanupInterval]);

  // Memory pressure handling
  useEffect(() => {
    const memoryThreshold = 5 * 1024; // 5MB threshold
    
    if (state.memoryUsage > memoryThreshold) {
      console.warn(`Memory usage high: ${state.memoryUsage.toFixed(2)}KB, performing cleanup`);
      performCleanup();
    }
  }, [state.memoryUsage, performCleanup]);

  // Connection recovery
  const reconnect = useCallback(() => {
    disconnectWebSocket();
    setTimeout(() => {
      initializeWebSocket();
    }, 1000);
  }, [disconnectWebSocket, initializeWebSocket]);

  // Manual cleanup trigger
  const manualCleanup = useCallback(() => {
    performCleanup();
  }, [performCleanup]);

  // Get connection health
  const getConnectionHealth = useCallback(() => {
    const now = Date.now();
    const timeSinceLastUpdate = now - state.lastUpdate;
    
    return {
      isHealthy: isConnected && timeSinceLastUpdate < 60000, // Healthy if updated within 1 minute
      timeSinceLastUpdate,
      dataFreshness: timeSinceLastUpdate < 30000 ? 'fresh' : 
                     timeSinceLastUpdate < 60000 ? 'stale' : 'very_stale'
    };
  }, [isConnected, state.lastUpdate]);

  return {
    // Connection state
    isConnected: state.isConnected,
    connectionStatus: state.connectionStatus,
    lastUpdate: state.lastUpdate,
    
    // Data metrics
    dataPoints: state.dataPoints,
    memoryUsage: state.memoryUsage,
    
    // Actions
    reconnect,
    cleanup: manualCleanup,
    subscribe,
    
    // Health monitoring
    getConnectionHealth,
    
    // Configuration
    symbols,
    enableSignals,
    enablePortfolio,
    maxDataPoints
  };
};

// Hook for monitoring real-time performance
export const useRealTimePerformance = () => {
  const [metrics, setMetrics] = useState({
    updateFrequency: 0,
    averageLatency: 0,
    missedUpdates: 0,
    lastMeasurement: Date.now()
  });

  const updateCountRef = useRef(0);
  const latencyMeasurementsRef = useRef<number[]>([]);
  const lastUpdateRef = useRef(Date.now());

  // Track update frequency and latency
  const trackUpdate = useCallback((latency?: number) => {
    const now = Date.now();
    updateCountRef.current++;
    
    if (latency) {
      latencyMeasurementsRef.current.push(latency);
      // Keep only last 100 measurements for memory efficiency
      if (latencyMeasurementsRef.current.length > 100) {
        latencyMeasurementsRef.current = latencyMeasurementsRef.current.slice(-100);
      }
    }

    // Calculate metrics every 10 seconds
    if (now - metrics.lastMeasurement > 10000) {
      const timeDiff = now - metrics.lastMeasurement;
      const frequency = (updateCountRef.current / timeDiff) * 1000; // Updates per second
      const avgLatency = latencyMeasurementsRef.current.length > 0 
        ? latencyMeasurementsRef.current.reduce((a, b) => a + b, 0) / latencyMeasurementsRef.current.length
        : 0;

      setMetrics({
        updateFrequency: frequency,
        averageLatency: avgLatency,
        missedUpdates: 0, // TODO: Implement missed update detection
        lastMeasurement: now
      });

      // Reset counters
      updateCountRef.current = 0;
      latencyMeasurementsRef.current = [];
    }

    lastUpdateRef.current = now;
  }, [metrics.lastMeasurement]);

  return {
    metrics,
    trackUpdate
  };
};

// Hook for real-time data validation
export const useDataValidation = () => {
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const validateMarketData = useCallback((data: any) => {
    const errors: string[] = [];

    if (!data.symbol || typeof data.symbol !== 'string') {
      errors.push('Invalid symbol');
    }

    if (!data.price || typeof data.price !== 'number' || data.price <= 0) {
      errors.push('Invalid price');
    }

    if (!data.timestamp || typeof data.timestamp !== 'number') {
      errors.push('Invalid timestamp');
    }

    if (data.volume !== undefined && (typeof data.volume !== 'number' || data.volume < 0)) {
      errors.push('Invalid volume');
    }

    setValidationErrors(errors);
    return errors.length === 0;
  }, []);

  const validateTradingSignal = useCallback((signal: any) => {
    const errors: string[] = [];

    if (!signal.id || typeof signal.id !== 'string') {
      errors.push('Invalid signal ID');
    }

    if (!signal.symbol || typeof signal.symbol !== 'string') {
      errors.push('Invalid symbol');
    }

    if (!signal.signal_type || !['buy', 'sell', 'hold', 'strong_buy', 'strong_sell'].includes(signal.signal_type)) {
      errors.push('Invalid signal type');
    }

    if (signal.confidence === undefined || typeof signal.confidence !== 'number' || 
        signal.confidence < 0 || signal.confidence > 1) {
      errors.push('Invalid confidence');
    }

    if (!signal.quality || !['excellent', 'good', 'fair', 'poor'].includes(signal.quality)) {
      errors.push('Invalid quality');
    }

    setValidationErrors(errors);
    return errors.length === 0;
  }, []);

  return {
    validationErrors,
    validateMarketData,
    validateTradingSignal,
    clearErrors: () => setValidationErrors([])
  };
};
