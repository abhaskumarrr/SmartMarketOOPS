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
        setLoading(true);
        const data = await apiService.getPortfolio();
        setPortfolio(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch portfolio');
        console.error('Portfolio fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolio();

    // Subscribe to real-time portfolio updates
    wsService.connect();
    wsService.subscribeToPortfolioUpdates((data: Portfolio) => {
      setPortfolio(data);
    });

    return () => {
      wsService.unsubscribeFromPortfolioUpdates();
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
