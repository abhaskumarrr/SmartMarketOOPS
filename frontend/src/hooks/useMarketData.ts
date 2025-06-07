'use client';

import { useState, useEffect } from 'react';
import { MarketData } from '@/types/trading';
import { apiService } from '@/services/api';
import { wsService } from '@/services/websocket';

export function useMarketData(symbol?: string) {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMarketData = async () => {
      try {
        console.log('🔄 Fetching market data...', symbol ? `for ${symbol}` : 'for all symbols');
        setLoading(true);
        setError(null);

        const data = await apiService.getMarketData(symbol);
        console.log('✅ Market data received:', data);

        setMarketData(data);
        setError(null);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch market data';
        console.error('❌ Market data fetch error:', err);
        setError(errorMessage);

        // Set fallback data to prevent infinite loading
        setMarketData([]);
      } finally {
        setLoading(false);
      }
    };

    fetchMarketData();

    // Subscribe to real-time market data updates
    try {
      console.log('🔌 Setting up WebSocket for market data updates...');
      wsService.connect();
      wsService.subscribeToMarketData((data: MarketData) => {
        console.log('📡 Real-time market data update received:', data);
        setMarketData(prev => {
          const index = prev.findIndex(item => item.symbol === data.symbol);
          if (index >= 0) {
            const updated = [...prev];
            updated[index] = data;
            return updated;
          } else {
            return [...prev, data];
          }
        });
      });
    } catch (wsError) {
      console.error('❌ WebSocket setup error for market data:', wsError);
    }

    return () => {
      try {
        wsService.unsubscribeFromMarketData();
      } catch (cleanupError) {
        console.error('❌ WebSocket cleanup error for market data:', cleanupError);
      }
    };
  }, [symbol]);

  const refreshMarketData = async () => {
    try {
      setLoading(true);
      const data = await apiService.getMarketData(symbol);
      setMarketData(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh market data');
    } finally {
      setLoading(false);
    }
  };

  return {
    marketData,
    loading,
    error,
    refreshMarketData,
  };
}
