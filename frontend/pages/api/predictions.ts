// API route for model predictions
// This endpoint serves prediction data for the TradingView chart component

import { NextApiRequest, NextApiResponse } from 'next';

interface Prediction {
  timestamp: number;
  price: number;
  prediction: number;
  confidence: number;
}

// Temporary sample prediction data
const SAMPLE_PREDICTIONS: Prediction[] = [
  { timestamp: Date.now() - 86400000 * 5, price: 42500, prediction: 43200, confidence: 0.85 },
  { timestamp: Date.now() - 86400000 * 4, price: 43200, prediction: 44100, confidence: 0.78 },
  { timestamp: Date.now() - 86400000 * 3, price: 44100, prediction: 43800, confidence: 0.82 },
  { timestamp: Date.now() - 86400000 * 2, price: 43800, prediction: 43200, confidence: 0.75 },
  { timestamp: Date.now() - 86400000 * 1, price: 43200, prediction: 44000, confidence: 0.80 },
  { timestamp: Date.now(), price: 44000, prediction: 45200, confidence: 0.83 },
];

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // In production, this would fetch from backend API
    // For now, return sample data
    
    // Add a small delay to simulate network latency
    setTimeout(() => {
      res.status(200).json({
        success: true,
        data: SAMPLE_PREDICTIONS,
        message: 'Prediction data retrieved successfully'
      });
    }, 300);
  } catch (error) {
    console.error('Error in predictions API:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve prediction data',
      error: process.env.NODE_ENV === 'development' ? String(error) : undefined
    });
  }
} 