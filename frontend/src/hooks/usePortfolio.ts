'use client';

import { useState, useEffect } from 'react';
import { Portfolio } from '@/types/trading';
import { apiService } from '@/services/api';
import { wsService } from '@/services/websocket';

export function usePortfolio() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPortfolio = async () => {
      try {
        console.log('🔄 Fetching portfolio data...');
        setLoading(true);
        setError(null);

        const data = await apiService.getPortfolio();
        console.log('✅ Portfolio data received:', data);

        setPortfolio(data);
        setError(null);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch portfolio';
        console.error('❌ Portfolio fetch error:', err);
        setError(errorMessage);

        // Set fallback data to prevent infinite loading
        setPortfolio({
          totalBalance: 0,
          availableBalance: 0,
          totalPnl: 0,
          totalPnlPercentage: 0,
          dailyPnl: 0,
          dailyPnlPercentage: 0,
          positions: [],
        });
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolio();

    // Subscribe to real-time portfolio updates
    try {
      console.log('🔌 Setting up WebSocket for portfolio updates...');
      wsService.connect();
      wsService.subscribeToPortfolioUpdates((data: Portfolio) => {
        console.log('📊 Real-time portfolio update received:', data);
        setPortfolio(data);
      });
    } catch (wsError) {
      console.error('❌ WebSocket setup error:', wsError);
    }

    return () => {
      try {
        wsService.unsubscribeFromPortfolioUpdates();
      } catch (cleanupError) {
        console.error('❌ WebSocket cleanup error:', cleanupError);
      }
    };
  }, []);

  const refreshPortfolio = async () => {
    try {
      setLoading(true);
      const data = await apiService.getPortfolio();
      setPortfolio(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh portfolio');
    } finally {
      setLoading(false);
    }
  };

  return {
    portfolio,
    loading,
    error,
    refreshPortfolio,
  };
}
