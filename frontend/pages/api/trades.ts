// API route for trade data
// This endpoint serves trading data for the dashboard

import { NextApiRequest, NextApiResponse } from 'next';

interface Trade {
  id: string;
  timestamp: number;
  userId: string;
  symbol: string;
  type: 'market' | 'limit';
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  status: 'pending' | 'completed' | 'cancelled';
  pnl?: number;
  fee: number;
  strategyId?: string;
  botId?: string;
  signalId?: string;
}

// Temporary sample trade data
const SAMPLE_TRADES: Trade[] = [
  {
    id: 'trade-1',
    timestamp: Date.now() - 86400000 * 5,
    userId: 'user-1',
    symbol: 'BTCUSD',
    type: 'market',
    side: 'buy',
    price: 42500,
    quantity: 0.5,
    status: 'completed',
    pnl: 650,
    fee: 21.25,
    strategyId: 'strategy-1'
  },
  {
    id: 'trade-2',
    timestamp: Date.now() - 86400000 * 3,
    userId: 'user-1',
    symbol: 'BTCUSD',
    type: 'limit',
    side: 'sell',
    price: 44300,
    quantity: 0.5,
    status: 'completed',
    pnl: -120,
    fee: 22.15,
    strategyId: 'strategy-2'
  },
  {
    id: 'trade-3',
    timestamp: Date.now() - 86400000 * 1,
    userId: 'user-1',
    symbol: 'BTCUSD',
    type: 'market',
    side: 'buy',
    price: 43800,
    quantity: 0.75,
    status: 'completed',
    pnl: 330,
    fee: 32.85,
    signalId: 'signal-3'
  }
];

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // In production, this would fetch from backend API
    // For now, return sample data
    
    // Add a small delay to simulate network latency
    setTimeout(() => {
      res.status(200).json({
        success: true,
        data: SAMPLE_TRADES,
        message: 'Trade data retrieved successfully'
      });
    }, 300);
  } catch (error) {
    console.error('Error in trades API:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve trade data',
      error: process.env.NODE_ENV === 'development' ? String(error) : undefined
    });
  }
} 