import { NextRequest, NextResponse } from 'next/server';
import { authenticateRequest, createUnauthorizedResponse, hasRole } from '../../../../lib/auth';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3001';

export async function GET(request: NextRequest) {
  try {
    // Authentication check
    const user = await authenticateRequest(request);
    if (!user) {
      console.log('[Market Data API] Authentication failed');
      return createUnauthorizedResponse('Authentication required to access market data');
    }

    // Role-based authorization (market data requires at least viewer role)
    if (!hasRole(user, 'viewer')) {
      console.log(`[Market Data API] User ${user.email} lacks viewer permissions`);
      return createUnauthorizedResponse('Viewer permissions required');
    }

    console.log(`[Market Data API] Authenticated user ${user.email} fetching from: ${BACKEND_URL}/api/paper-trading/market-data`);

    const response = await fetch(`${BACKEND_URL}/api/paper-trading/market-data`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      console.error(`[Market Data API] Backend responded with status: ${response.status}`);
      return NextResponse.json(
        {
          success: false,
          error: `Backend service unavailable (${response.status})`,
          details: `Failed to fetch from ${BACKEND_URL}`
        },
        { status: response.status }
      );
    }

    const data = await response.json();
    console.log('[Market Data API] Successfully fetched market data');

    return NextResponse.json(data);
  } catch (error) {
    console.error('[Market Data API] Error:', error);

    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        {
          success: false,
          error: 'Backend service is not running',
          details: `Cannot connect to ${BACKEND_URL}`,
          suggestion: 'Please ensure the backend service is running on port 3001'
        },
        { status: 503 }
      );
    }

    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        {
          success: false,
          error: 'Request timeout',
          details: 'Backend service took too long to respond'
        },
        { status: 504 }
      );
    }

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
