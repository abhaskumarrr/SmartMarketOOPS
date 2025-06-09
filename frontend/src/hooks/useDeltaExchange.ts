/**
 * Hook for Delta Exchange API interaction with real-time WebSocket integration
 */

import { useState, useEffect, useCallback } from 'react';
import deltaService, { 
  DeltaMarketData, 
  DeltaOrder, 
  DeltaPosition, 
  DeltaBalance,
  DeltaBotStatus,
  DeltaOrderRequest,
  DeltaBotConfig
} from '@/services/deltaService';
import { useRealTimeData } from './useRealTimeData';

export interface UseDeltaExchangeProps {
  symbols?: string[];
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export function useDeltaExchange({
  symbols = ['BTCUSD'],
  autoRefresh = true,
  refreshInterval = 10000,
}: UseDeltaExchangeProps = {}) {
  // Connection state
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Data states
  const [marketData, setMarketData] = useState<Record<string, DeltaMarketData>>({});
  const [orders, setOrders] = useState<DeltaOrder[]>([]);
  const [positions, setPositions] = useState<DeltaPosition[]>([]);
  const [balances, setBalances] = useState<DeltaBalance[]>([]);
  const [bots, setBots] = useState<DeltaBotStatus[]>([]);

  // Real-time WebSocket integration
  const { 
    isConnected: wsConnected, 
    lastMarketData,
    lastTrade,
    lastPortfolioUpdate,
    marketTicks,
    subscribeToSymbol,
    unsubscribeFromSymbol
  } = useRealTimeData();

  // Initialize and check connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        setIsLoading(true);
        
        // Check both Delta Exchange API and real-time data health
        const [deltaConnected, realTimeHealthy] = await Promise.all([
          deltaService.testConnection(),
          deltaService.checkRealTimeDataHealth()
        ]);
        
        setIsConnected(deltaConnected);
        
        if (!deltaConnected) {
          setError('Failed to connect to Delta Exchange API');
          console.error('Failed to connect to Delta Exchange API - please ensure the backend is running');
        } else {
          setError(null);
          console.log('✅ Connected to Delta Exchange API');
          
          if (realTimeHealthy) {
            console.log('✅ Real-time data service is healthy');
          } else {
            console.warn('⚠️ Real-time data service may be unavailable');
          }
          
          // Initial data fetch when connection is established
          await fetchData();
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        setIsConnected(false);
        console.error('Error connecting to Delta Exchange API:', err);
      } finally {
        setIsLoading(false);
      }
    };

    checkConnection();
  }, []);

  // Set up auto-refresh interval
  useEffect(() => {
    if (!autoRefresh || !isConnected) return;

    const intervalId = setInterval(() => {
      fetchData();
    }, refreshInterval);

    return () => clearInterval(intervalId);
  }, [autoRefresh, isConnected, refreshInterval]);

  // Subscribe to real-time market data for symbols
  useEffect(() => {
    if (wsConnected && symbols.length > 0) {
      // Subscribe to all symbols
      symbols.forEach(symbol => {
        subscribeToSymbol(symbol);
      });

      return () => {
        // Unsubscribe from all symbols on cleanup
        symbols.forEach(symbol => {
          unsubscribeFromSymbol(symbol);
        });
      };
    }
  }, [wsConnected, symbols, subscribeToSymbol, unsubscribeFromSymbol]);

  // Update market data from real-time WebSocket updates
  useEffect(() => {
    if (lastMarketData && lastMarketData.symbol) {
      setMarketData(prev => ({
        ...prev,
        [lastMarketData.symbol]: {
          symbol: lastMarketData.symbol,
          price: lastMarketData.price,
          change: lastMarketData.changePercentage24h,
          changePercent: lastMarketData.changePercentage24h,
          volume: lastMarketData.volume,
          high24h: 0, // Real-time data doesn't include these fields
          low24h: 0,
          timestamp: lastMarketData.timestamp,
          source: 'websocket'
        }
      }));
      
      console.log(`📊 Updated market data for ${lastMarketData.symbol}: $${lastMarketData.price}`);
    }
  }, [lastMarketData]);

  // Update market data from market tick history
  useEffect(() => {
    if (marketTicks.length > 0) {
      const latestTicks: Record<string, DeltaMarketData> = {};
      
      // Get the latest tick for each symbol
      marketTicks.forEach(tick => {
        if (!latestTicks[tick.symbol] || tick.timestamp > latestTicks[tick.symbol].timestamp) {
          latestTicks[tick.symbol] = {
            symbol: tick.symbol,
            price: tick.price,
            change: tick.changePercentage24h,
            changePercent: tick.changePercentage24h,
            volume: tick.volume,
            high24h: 0,
            low24h: 0,
            timestamp: tick.timestamp,
            source: 'websocket-history'
          };
        }
      });

      // Update market data with latest ticks
      setMarketData(prev => ({
        ...prev,
        ...latestTicks
      }));
    }
  }, [marketTicks]);

  // Handle portfolio updates from WebSocket
  useEffect(() => {
    if (lastPortfolioUpdate) {
      console.log('💼 Portfolio update received:', lastPortfolioUpdate);
      
      // Update positions from portfolio data
      if (lastPortfolioUpdate.positions) {
        const updatedPositions: DeltaPosition[] = lastPortfolioUpdate.positions.map((pos, index) => ({
          id: index,
          symbol: pos.symbol,
          side: pos.side === 'long' ? 'buy' : 'sell',
          size: pos.size,
          entryPrice: pos.entryPrice,
          markPrice: pos.currentPrice,
          pnl: pos.pnl,
          pnlPercentage: pos.pnlPercentage,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString()
        }));
        
        setPositions(updatedPositions);
      }
    }
  }, [lastPortfolioUpdate]);

  // Fetch all necessary data
  const fetchData = useCallback(async () => {
    if (!isConnected) return;
    
    try {
      setIsLoading(true);
      
      // Fetch market data (prioritizes real-time data)
      if (symbols.length > 0) {
        const data = await deltaService.getMultipleMarketData(symbols);
        const marketDataMap: Record<string, DeltaMarketData> = {};
        
        data.forEach(item => {
          marketDataMap[item.symbol] = item;
        });
        
        // Merge with existing real-time data, preferring real-time timestamps
        setMarketData(prev => {
          const merged = { ...prev };
          Object.keys(marketDataMap).forEach(symbol => {
            const apiData = marketDataMap[symbol];
            const existingData = merged[symbol];
            
            // Only update if API data is newer or no existing data
            if (!existingData || apiData.timestamp > existingData.timestamp) {
              merged[symbol] = apiData;
            }
          });
          return merged;
        });
      }
      
      // Fetch orders, positions, balances, and bots
      const [ordersData, positionsData, balancesData, botsData] = await Promise.all([
        deltaService.getOpenOrders(),
        deltaService.getPositions(),
        deltaService.getBalances(),
        deltaService.getBots()
      ]);
      
      setOrders(ordersData);
      
      // Only update positions if we don't have fresher WebSocket data
      if (!lastPortfolioUpdate || Date.now() - lastPortfolioUpdate.timestamp > 30000) {
        setPositions(positionsData);
      }
      
      setBalances(balancesData);
      setBots(botsData);
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  }, [isConnected, symbols, lastPortfolioUpdate]);

  // Place an order with real-time feedback
  const placeOrder = useCallback(async (order: DeltaOrderRequest): Promise<DeltaOrder | null> => {
    if (!isConnected) {
      setError('Not connected to Delta Exchange API');
      return null;
    }
    
    try {
      setIsLoading(true);
      const result = await deltaService.placeOrder(order);
      
      if (result) {
        // Immediately add the order to the local state for instant feedback
        setOrders(prev => [result, ...prev]);
        
        // Refresh orders in the background to get updated status
        setTimeout(async () => {
          const updatedOrders = await deltaService.getOpenOrders();
          setOrders(updatedOrders);
        }, 2000);
        
        console.log('✅ Order placed successfully:', result);
      }
      
      return result;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to place order');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [isConnected]);

  // Cancel an order with real-time feedback
  const cancelOrder = useCallback(async (orderId: number): Promise<boolean> => {
    if (!isConnected) {
      setError('Not connected to Delta Exchange API');
      return false;
    }
    
    try {
      setIsLoading(true);
      const success = await deltaService.cancelOrder(orderId);
      
      if (success) {
        // Immediately update the order status locally
        setOrders(prev => prev.map(order => 
          order.id === orderId 
            ? { ...order, state: 'cancelled', updatedAt: new Date().toISOString() }
            : order
        ));
        
        // Refresh orders in the background
        setTimeout(async () => {
          const updatedOrders = await deltaService.getOpenOrders();
          setOrders(updatedOrders);
        }, 1000);
        
        console.log('✅ Order cancelled successfully:', orderId);
      }
      
      return success;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel order');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [isConnected]);

  // Bot management functions (unchanged)
  const createBot = useCallback(async (config: DeltaBotConfig): Promise<string | null> => {
    if (!isConnected) {
      setError('Not connected to Delta Exchange API');
      return null;
    }
    
    try {
      setIsLoading(true);
      const botId = await deltaService.createBot(config);
      
      if (botId) {
        // Refresh bots after creating a new one
        const updatedBots = await deltaService.getBots();
        setBots(updatedBots);
      }
      
      return botId;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create bot');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [isConnected]);

  const startBot = useCallback(async (botId: string): Promise<boolean> => {
    if (!isConnected) {
      setError('Not connected to Delta Exchange API');
      return false;
    }
    
    try {
      setIsLoading(true);
      const success = await deltaService.startBot(botId);
      
      if (success) {
        // Refresh bots after starting
        const updatedBots = await deltaService.getBots();
        setBots(updatedBots);
      }
      
      return success;
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to start bot ${botId}`);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [isConnected]);

  const stopBot = useCallback(async (botId: string): Promise<boolean> => {
    if (!isConnected) {
      setError('Not connected to Delta Exchange API');
      return false;
    }
    
    try {
      setIsLoading(true);
      const success = await deltaService.stopBot(botId);
      
      if (success) {
        // Refresh bots after stopping
        const updatedBots = await deltaService.getBots();
        setBots(updatedBots);
      }
      
      return success;
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to stop bot ${botId}`);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [isConnected]);

  // Get real-time price for a symbol
  const getRealTimePrice = useCallback((symbol: string): number => {
    return marketData[symbol]?.price || 0;
  }, [marketData]);

  // Check if real-time data is available
  const hasRealTimeData = useCallback((symbol: string): boolean => {
    const data = marketData[symbol];
    return data && data.source === 'websocket' || data?.source === 'websocket-history';
  }, [marketData]);

  // Expose the API with real-time enhancements
  return {
    // States
    isConnected: isConnected && wsConnected, // Both API and WebSocket should be connected
    isLoading,
    error,
    
    // Data with real-time updates
    marketData,
    orders,
    positions,
    balances,
    bots,
    
    // Real-time specific data
    lastTrade,
    lastPortfolioUpdate,
    marketTicks,
    
    // Actions
    refreshData: fetchData,
    placeOrder,
    cancelOrder,
    createBot,
    startBot,
    stopBot,
    
    // Real-time utilities
    getRealTimePrice,
    hasRealTimeData,
    subscribeToSymbol,
    unsubscribeFromSymbol
  };
} 