// API route for asset allocation metrics
// This endpoint serves asset allocation data for the dashboard

import { NextApiRequest, NextApiResponse } from 'next';

interface AssetAllocationData {
  name: string;
  value: number;
}

// Sample allocation data
const SAMPLE_ALLOCATION: AssetAllocationData[] = [
  { name: 'BTC', value: 45 },
  { name: 'ETH', value: 30 },
  { name: 'SOL', value: 15 },
  { name: 'AVAX', value: 7 },
  { name: 'MATIC', value: 3 }
];

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // Add a small delay to simulate network latency
    setTimeout(() => {
      res.status(200).json({
        success: true,
        data: SAMPLE_ALLOCATION,
        message: 'Asset allocation data retrieved successfully'
      });
    }, 300);
  } catch (error) {
    console.error('Error in asset allocation API:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve asset allocation data',
      error: process.env.NODE_ENV === 'development' ? String(error) : undefined
    });
  }
} 