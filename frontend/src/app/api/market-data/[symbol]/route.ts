import { NextRequest, NextResponse } from 'next/server';

// Mock market data generator
function generateMockOHLCV(symbol: string, timeframe: string = '1h', limit: number = 100) {
  const data = [];
  const now = Date.now();
  const timeframeMs = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000,
  }[timeframe] || 60 * 60 * 1000;

  // Base prices for different symbols
  const basePrices = {
    'BTCUSD': 45000,
    'ETHUSD': 2800,
    'SOLUSD': 120,
  };

  let basePrice = basePrices[symbol as keyof typeof basePrices] || 45000;

  for (let i = limit - 1; i >= 0; i--) {
    const timestamp = now - (i * timeframeMs);
    
    // Generate realistic price movement
    const volatility = 0.02; // 2% volatility
    const change = (Math.random() - 0.5) * volatility;
    basePrice = basePrice * (1 + change);
    
    const open = basePrice;
    const close = basePrice * (1 + (Math.random() - 0.5) * volatility * 0.5);
    const high = Math.max(open, close) * (1 + Math.random() * volatility * 0.3);
    const low = Math.min(open, close) * (1 - Math.random() * volatility * 0.3);
    const volume = Math.random() * 1000000 + 100000;

    data.push({
      timestamp,
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Number(volume.toFixed(0)),
    });
  }

  return data;
}

export async function GET(
  request: NextRequest,
  { params }: { params: { symbol: string } }
) {
  try {
    const { symbol } = params;
    const { searchParams } = new URL(request.url);
    const timeframe = searchParams.get('timeframe') || '1h';
    const limit = parseInt(searchParams.get('limit') || '100');

    // Try to fetch from backend first
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3005';
    
    try {
      const response = await fetch(
        `${backendUrl}/api/trading/market-data/${symbol}?timeframe=${timeframe}&limit=${limit}`,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data);
      }
    } catch (error) {
      console.log('Backend not available, using mock data');
    }

    // Generate mock data
    const mockData = generateMockOHLCV(symbol, timeframe, limit);
    
    return NextResponse.json({
      success: true,
      data: mockData,
      symbol,
      timeframe,
      limit,
      timestamp: new Date().toISOString(),
    });

  } catch (error) {
    console.error('Error fetching market data:', error);
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to fetch market data' 
      },
      { status: 500 }
    );
  }
}

// Handle CORS preflight
export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}
