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
        setLoading(true);
        const data = await apiService.getMarketData(symbol);
        setMarketData(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch market data');
        console.error('Market data fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMarketData();

    // Subscribe to real-time market data updates
    wsService.connect();
    wsService.subscribeToMarketData((data: MarketData) => {
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

    return () => {
      wsService.unsubscribeFromMarketData();
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
