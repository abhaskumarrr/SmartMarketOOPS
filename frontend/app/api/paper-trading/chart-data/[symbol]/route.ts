import { NextRequest, NextResponse } from 'next/server';

// Generate mock candlestick data for the chart
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

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ symbol: string }> }
) {
  try {
    const { searchParams } = new URL(request.url);
    const timeframe = searchParams.get('timeframe') || '1m';
    const limit = parseInt(searchParams.get('limit') || '100');

    const params = await context.params;
    const symbol = params.symbol;
    
    // Generate mock data
    const candles = generateMockCandleData(symbol, timeframe, limit);
    
    return NextResponse.json({
      success: true,
      data: {
        symbol,
        timeframe,
        candles
      }
    });
  } catch (error) {
    console.error('Error generating chart data:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to generate chart data' },
      { status: 500 }
    );
  }
}
