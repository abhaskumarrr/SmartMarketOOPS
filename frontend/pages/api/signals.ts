// API route for trading signals
// This endpoint serves signal data for the TradingView chart component

import { NextApiRequest, NextApiResponse } from 'next';

interface TradingSignal {
  id: string;
  timestamp: number;
  symbol: string;
  direction: 'buy' | 'sell';
  price: number;
  confidence: number;
  source: string;
  strategy: string;
  status: 'pending' | 'executed' | 'expired';
}

// Temporary sample signal data
const SAMPLE_SIGNALS: TradingSignal[] = [
  {
    id: 'signal-1',
    timestamp: Date.now() - 86400000 * 4.5,
    symbol: 'BTCUSD',
    direction: 'buy',
    price: 43100,
    confidence: 0.87,
    source: 'ML-model',
    strategy: 'trend-following',
    status: 'executed'
  },
  {
    id: 'signal-2',
    timestamp: Date.now() - 86400000 * 2.3,
    symbol: 'BTCUSD',
    direction: 'sell',
    price: 44200,
    confidence: 0.81,
    source: 'ML-model',
    strategy: 'reversal',
    status: 'executed'
  },
  {
    id: 'signal-3',
    timestamp: Date.now() - 86400000 * 0.5,
    symbol: 'BTCUSD',
    direction: 'buy',
    price: 43600,
    confidence: 0.92,
    source: 'ML-model',
    strategy: 'support-resistance',
    status: 'pending'
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
        data: SAMPLE_SIGNALS,
        message: 'Signal data retrieved successfully'
      });
    }, 300);
  } catch (error) {
    console.error('Error in signals API:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve signal data',
      error: process.env.NODE_ENV === 'development' ? String(error) : undefined
    });
  }
} 