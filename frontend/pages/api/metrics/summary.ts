// API route for performance metrics summary
// This endpoint serves overall trading performance metrics for the dashboard

import { NextApiRequest, NextApiResponse } from 'next';

interface MetricsData {
  totalProfit: number;
  totalTrades: number;
  winRate: number;
  avgProfit: number;
  avgLoss: number;
  profitFactor: number;
  maxDrawdown: number;
  bestTrade: number;
  worstTrade: number;
  avgTradeTime: string;
  dailyVolume: number;
}

// Sample metrics data
const SAMPLE_METRICS: MetricsData = {
  totalProfit: 12450.75,
  totalTrades: 87,
  winRate: 64.3,
  avgProfit: 242.18,
  avgLoss: -156.32,
  profitFactor: 2.13,
  maxDrawdown: -8.76,
  bestTrade: 985.43,
  worstTrade: -532.10,
  avgTradeTime: '3h 12m',
  dailyVolume: 32567.89
};

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // Add a small delay to simulate network latency
    setTimeout(() => {
      res.status(200).json({
        success: true,
        data: SAMPLE_METRICS,
        timestamp: new Date().toISOString()
      });
    }, 500);
  } catch (error) {
    console.error('Error in metrics summary API:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve metrics data',
      timestamp: new Date().toISOString()
    });
  }
} 