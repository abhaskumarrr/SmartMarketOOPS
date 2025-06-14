import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

export async function GET(req: NextRequest) {
  try {
    // Forward the request to backend
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';
    
    const response = await axios.get(`${backendUrl}/api/markets`, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    return NextResponse.json(response.data);
  } catch (error) {
    console.error('Error fetching markets:', error);
    
    // Handle different types of errors
    if (axios.isAxiosError(error) && error.response) {
      return NextResponse.json({
        success: false,
        error: 'Market fetch failed',
        message: error.response.data?.message || error.message,
        code: error.response.status
      }, { status: error.response.status });
    }
    
    // If backend is down or not available yet, return mock data
    // This allows frontend development to continue without backend dependency
    return NextResponse.json({
      success: true,
      data: [
        { symbol: 'BTCUSD', productId: 1, name: 'Bitcoin', minSize: 0.001, tickSize: 0.5, status: 'active' },
        { symbol: 'ETHUSD', productId: 2, name: 'Ethereum', minSize: 0.01, tickSize: 0.05, status: 'active' },
        { symbol: 'SOLUSD', productId: 3, name: 'Solana', minSize: 0.1, tickSize: 0.01, status: 'active' },
        { symbol: 'AVAXUSD', productId: 4, name: 'Avalanche', minSize: 0.1, tickSize: 0.01, status: 'active' },
        { symbol: 'BNBUSD', productId: 5, name: 'Binance Coin', minSize: 0.01, tickSize: 0.1, status: 'active' }
      ]
    });
  }
}

// Endpoint to lookup a specific market by symbol
export async function POST(req: NextRequest) {
  try {
    const data = await req.json();
    
    if (!data.symbol) {
      return NextResponse.json({
        success: false,
        error: 'Missing symbol',
        message: 'Symbol is required'
      }, { status: 400 });
    }
    
    // Forward the request to backend
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';
    
    const response = await axios.post(`${backendUrl}/api/markets/lookup`, data, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    return NextResponse.json(response.data);
  } catch (error) {
    console.error('Error looking up market:', error);
    
    // Fallback to mock data if backend is unavailable
    const mockMarkets: Record<string, { productId: number; name: string; minSize: number; tickSize: number; status: string }> = {
      'BTCUSD': { productId: 1, name: 'Bitcoin', minSize: 0.001, tickSize: 0.5, status: 'active' },
      'ETHUSD': { productId: 2, name: 'Ethereum', minSize: 0.01, tickSize: 0.05, status: 'active' },
      'SOLUSD': { productId: 3, name: 'Solana', minSize: 0.1, tickSize: 0.01, status: 'active' },
      'AVAXUSD': { productId: 4, name: 'Avalanche', minSize: 0.1, tickSize: 0.01, status: 'active' },
      'BNBUSD': { productId: 5, name: 'Binance Coin', minSize: 0.01, tickSize: 0.1, status: 'active' }
    };
    
    try {
      const { symbol } = await req.json();
      
      if (symbol && typeof symbol === 'string' && mockMarkets[symbol]) {
        return NextResponse.json({
          success: true,
          data: { symbol, ...mockMarkets[symbol] }
        });
      }
    } catch (parseError) {
      console.error('Error parsing request body:', parseError);
    }
    
    return NextResponse.json({
      success: false,
      error: 'Market lookup failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
} 