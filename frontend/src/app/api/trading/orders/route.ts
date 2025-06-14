import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

// Order placement API endpoint
export async function POST(req: NextRequest) {
  try {
    const orderData = await req.json();
    
    // Validate required fields
    if (!orderData.product_id || !orderData.size || !orderData.side || !orderData.order_type) {
      return NextResponse.json({
        success: false,
        error: 'Missing required fields',
        message: 'product_id, size, side, and order_type are required'
      }, { status: 400 });
    }
    
    // Forward the request to backend
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';
    
    const response = await axios.post(`${backendUrl}/api/trading/orders`, orderData, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    return NextResponse.json(response.data);
  } catch (error) {
    console.error('Error placing order:', error);
    
    // Handle different types of errors
    if (axios.isAxiosError(error) && error.response) {
      return NextResponse.json({
        success: false,
        error: 'Order placement failed',
        message: error.response.data?.message || error.message,
        code: error.response.status
      }, { status: error.response.status });
    }
    
    return NextResponse.json({
      success: false,
      error: 'Order placement failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// Market lookup endpoint to get market information by symbol
export async function GET(req: NextRequest) {
  try {
    const searchParams = req.nextUrl.searchParams;
    const symbol = searchParams.get('symbol');
    
    if (!symbol) {
      return NextResponse.json({
        success: false,
        error: 'Missing symbol parameter',
        message: 'Symbol parameter is required'
      }, { status: 400 });
    }
    
    // Forward the request to backend
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';
    
    const response = await axios.get(`${backendUrl}/api/markets/lookup?symbol=${encodeURIComponent(symbol)}`, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    return NextResponse.json(response.data);
  } catch (error) {
    console.error('Error looking up market:', error);
    
    // Handle different types of errors
    if (axios.isAxiosError(error) && error.response) {
      return NextResponse.json({
        success: false,
        error: 'Market lookup failed',
        message: error.response.data?.message || error.message,
        code: error.response.status
      }, { status: error.response.status });
    }
    
    return NextResponse.json({
      success: false,
      error: 'Market lookup failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
} 