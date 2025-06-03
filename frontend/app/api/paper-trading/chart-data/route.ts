import { NextRequest, NextResponse } from 'next/server';
import { authenticateRequest, createUnauthorizedResponse, hasRole } from '../../../../lib/auth';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

// Generate mock candlestick data for the chart (fallback)
function generateMockCandleData(symbol: string, timeframe: string, limit: number) {
  const candles = [];
  const now = Date.now();
  const timeframeMs = getTimeframeMs(timeframe);
  
  // Base price for different symbols
  const basePrices: { [key: string]: number } = {
    'ETH/USDT': 2600,
    'BTC/USDT': 45000,
    'SOL/USDT': 120
  };
  
  const basePrice = basePrices[symbol] || 2600;
  let currentPrice = basePrice;
  
  for (let i = limit - 1; i >= 0; i--) {
    const time = now - (i * timeframeMs);
    
    // Generate realistic price movement
    const volatility = 0.02; // 2% volatility
    const change = (Math.random() - 0.5) * volatility * currentPrice;
    const open = currentPrice;
    const close = open + change;
    
    // Generate high and low based on open and close
    const high = Math.max(open, close) + Math.random() * 0.01 * currentPrice;
    const low = Math.min(open, close) - Math.random() * 0.01 * currentPrice;
    
    // Generate volume
    const volume = Math.random() * 1000000 + 500000;
    
    candles.push({
      time,
      open: parseFloat(open.toFixed(2)),
      high: parseFloat(high.toFixed(2)),
      low: parseFloat(low.toFixed(2)),
      close: parseFloat(close.toFixed(2)),
      volume: Math.floor(volume)
    });
    
    currentPrice = close;
  }
  
  return candles;
}

function getTimeframeMs(timeframe: string): number {
  const timeframes: { [key: string]: number } = {
    '1m': 60 * 1000,
    '5m': 5 * 60 * 1000,
    '15m': 15 * 60 * 1000,
    '1h': 60 * 60 * 1000,
    '4h': 4 * 60 * 60 * 1000,
    '1d': 24 * 60 * 60 * 1000
  };
  
  return timeframes[timeframe] || timeframes['1m'];
}

export async function GET(request: NextRequest) {
  try {
    // Authentication check
    const user = await authenticateRequest(request);
    if (!user) {
      console.log('[Chart Data API] Authentication failed');
      return createUnauthorizedResponse('Authentication required to access chart data');
    }

    // Role-based authorization (chart data requires at least viewer role)
    if (!hasRole(user, 'viewer')) {
      console.log(`[Chart Data API] User ${user.email} lacks viewer permissions`);
      return createUnauthorizedResponse('Viewer permissions required');
    }

    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'ETH/USDT';
    const timeframe = searchParams.get('timeframe') || '1m';
    const limit = searchParams.get('limit') || '100';

    console.log(`[Chart Data API] Authenticated user ${user.email} fetching chart data: ${symbol} ${timeframe} (${limit} candles)`);

    // Try to fetch from backend first
    try {
      const response = await fetch(`${BACKEND_URL}/api/paper-trading/chart-data?symbol=${encodeURIComponent(symbol)}&timeframe=${timeframe}&limit=${limit}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(10000), // 10 second timeout
      });

      if (response.ok) {
        const data = await response.json();
        console.log(`[Chart Data API] Successfully fetched chart data from backend for ${symbol}`);
        return NextResponse.json(data);
      } else {
        console.warn(`[Chart Data API] Backend responded with status: ${response.status}, falling back to mock data`);
      }
    } catch (backendError) {
      console.warn('[Chart Data API] Backend unavailable, falling back to mock data:', backendError);
    }

    // Fallback to mock data if backend is unavailable
    console.log(`[Chart Data API] Generating mock chart data for ${symbol}`);
    const candles = generateMockCandleData(symbol, timeframe, parseInt(limit));

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        timeframe,
        candles
      },
      source: 'mock' // Indicate this is mock data
    });
  } catch (error) {
    console.error('[Chart Data API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
