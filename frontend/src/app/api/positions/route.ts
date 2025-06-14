import { NextRequest, NextResponse } from 'next/server';

// Mock positions data for development
const mockPositions = [
  {
    id: '1',
    symbol: 'BTCUSD',
    side: 'long',
    size: 0.5,
    entryPrice: 45000,
    currentPrice: 46500,
    pnl: 750,
    pnlPercent: 3.33,
    status: 'open',
    timestamp: new Date().toISOString(),
  },
  {
    id: '2',
    symbol: 'ETHUSD',
    side: 'short',
    size: 2.0,
    entryPrice: 2800,
    currentPrice: 2750,
    pnl: 100,
    pnlPercent: 1.79,
    status: 'open',
    timestamp: new Date().toISOString(),
  },
  {
    id: '3',
    symbol: 'SOLUSD',
    side: 'long',
    size: 10.0,
    entryPrice: 120,
    currentPrice: 125,
    pnl: 50,
    pnlPercent: 4.17,
    status: 'open',
    timestamp: new Date().toISOString(),
  },
];

export async function GET(request: NextRequest) {
  try {
    // In production, this would fetch from the backend API
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';
    
    try {
      // Try to fetch from backend first
      const response = await fetch(`${backendUrl}/api/positions`, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data);
      }
    } catch (error) {
      console.log('Backend not available, using mock data');
    }
    
    // Fallback to mock data
    return NextResponse.json({
      success: true,
      data: mockPositions,
      total: mockPositions.length,
      timestamp: new Date().toISOString(),
    });
    
  } catch (error) {
    console.error('Error fetching positions:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to fetch positions',
        data: mockPositions // Still return mock data on error
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Mock position creation
    const newPosition = {
      id: Date.now().toString(),
      symbol: body.symbol || 'BTCUSD',
      side: body.side || 'long',
      size: body.size || 1.0,
      entryPrice: body.entryPrice || 45000,
      currentPrice: body.entryPrice || 45000,
      pnl: 0,
      pnlPercent: 0,
      status: 'open',
      timestamp: new Date().toISOString(),
    };
    
    return NextResponse.json({
      success: true,
      data: newPosition,
      message: 'Position created successfully',
    });
    
  } catch (error) {
    console.error('Error creating position:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to create position' 
      },
      { status: 500 }
    );
  }
}
