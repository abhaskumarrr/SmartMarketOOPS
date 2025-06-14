import { NextRequest, NextResponse } from 'next/server';

// Mock data for metrics
const mockMetricsData = {
  status: "success",
  data: {
    portfolioValue: 10000.25,
    dailyChange: 2.5,
    weeklyChange: 5.2,
    monthlyChange: -1.8,
    allTimeChange: 15.7,
    transactions: {
      total: 120,
      successful: 118,
      failed: 2
    },
    positions: {
      open: 5,
      closed: 15,
      profitable: 12,
      unprofitable: 8
    },
    assets: [
      { symbol: "BTCUSD", allocation: 45.2, value: 4520.10 },
      { symbol: "ETHUSD", allocation: 30.5, value: 3050.75 },
      { symbol: "SOLUSD", allocation: 15.3, value: 1530.40 },
      { symbol: "ADAUSD", allocation: 9.0, value: 899.00 }
    ],
    performance: {
      daily: [
        { date: "2023-06-01", value: 9800.50 },
        { date: "2023-06-02", value: 9850.25 },
        { date: "2023-06-03", value: 9900.00 },
        { date: "2023-06-04", value: 9950.75 },
        { date: "2023-06-05", value: 10000.25 }
      ]
    },
    timestamp: new Date().toISOString(),
    source: "frontend-mock"
  }
};

export async function GET(request: NextRequest) {
  const url = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';
  
  try {
    // Attempt to fetch from backend first
    const response = await fetch(`${url}/api/metrics`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // Add a short timeout to prevent hanging
      signal: AbortSignal.timeout(2000)
    });
    
    if (response.ok) {
      const data = await response.json();
      return NextResponse.json(data);
    } else {
      console.warn(`Backend metrics API returned status: ${response.status}`);
      // Return mock data if backend response is not OK
      return NextResponse.json(mockMetricsData);
    }
  } catch (error) {
    console.warn('Error fetching metrics from backend, using mock data:', error);
    // Return mock data if there was an error fetching from backend
    return NextResponse.json(mockMetricsData);
  }
}

// Handle CORS preflight
export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
} 