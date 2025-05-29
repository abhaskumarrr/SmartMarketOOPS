// API route for daily metrics
// This endpoint serves daily performance metrics for the dashboard

import { NextApiRequest, NextApiResponse } from 'next';

interface DailyMetric {
  date: string;
  pnl: number;
  winRate: number;
  totalTrades: number;
  sharpeRatio: number;
  maxDrawdown: number;
  volume: number;
}

// Generate sample data for the past 30 days
const generateSampleData = (): DailyMetric[] => {
  const data: DailyMetric[] = [];
  const now = new Date();
  
  for (let i = 30; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    
    // Generate random data with some trends
    const dayOfWeek = date.getDay();
    const multiplier = dayOfWeek >= 1 && dayOfWeek <= 5 ? 1.2 : 0.8; // Higher on weekdays
    const trend = Math.sin(i / 5) * multiplier; // Add some cyclical pattern
    
    data.push({
      date: date.toISOString().split('T')[0],
      pnl: Math.round((Math.random() * 1000 - 300) * multiplier + trend * 200) / 100,
      winRate: Math.min(0.95, Math.max(0.3, 0.65 + Math.random() * 0.2 + trend * 0.1)),
      totalTrades: Math.floor(Math.random() * 20 * multiplier) + 5,
      sharpeRatio: Math.round((1 + Math.random() * 1.5 + trend * 0.5) * 100) / 100,
      maxDrawdown: Math.round(-(Math.random() * 5 + 1 - trend) * 100) / 100,
      volume: Math.floor(Math.random() * 10 + 2) * 1000 * multiplier
    });
  }
  
  return data;
};

// Sample metrics data
const SAMPLE_DAILY_METRICS = generateSampleData();

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // Get query parameters
    const { startDate, endDate } = req.query;
    
    // Filter data based on date range if provided
    let filteredData = SAMPLE_DAILY_METRICS;
    
    if (startDate && typeof startDate === 'string') {
      filteredData = filteredData.filter(metric => metric.date >= startDate);
    }
    
    if (endDate && typeof endDate === 'string') {
      filteredData = filteredData.filter(metric => metric.date <= endDate);
    }
    
    // Add a small delay to simulate network latency
    setTimeout(() => {
      res.status(200).json({
        success: true,
        data: filteredData,
        message: 'Daily metrics retrieved successfully'
      });
    }, 300);
  } catch (error) {
    console.error('Error in daily metrics API:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to retrieve daily metrics',
      error: process.env.NODE_ENV === 'development' ? String(error) : undefined
    });
  }
} 