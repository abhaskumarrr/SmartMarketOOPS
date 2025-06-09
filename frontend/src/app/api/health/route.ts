import { NextRequest, NextResponse } from 'next/server';
import os from 'os';

// Define response types
interface DependencyStatus {
  status: 'healthy' | 'degraded' | 'unreachable' | 'unknown';
  url: string;
  response: any; // Allow any response type
}

interface HealthData {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  service: string;
  version: string;
  environment: string;
  uptime: number;
  dependencies: {
    backend: DependencyStatus;
    mlSystem: DependencyStatus;
  };
  system: {
    nodeVersion: string;
    platform: string;
    arch: string;
    memory: {
      used: number;
      total: number;
    };
  };
}

export async function GET(request: NextRequest) {
  // Default response data
  const healthData: HealthData = {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'frontend',
    version: process.env.NEXT_PUBLIC_APP_VERSION || '0.1.0',
    environment: process.env.NEXT_PUBLIC_ENVIRONMENT || 'development',
    uptime: process.uptime(),
    dependencies: {
      backend: {
        status: 'unknown',
        url: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006',
        response: null
      },
      mlSystem: {
        status: 'unknown',
        url: process.env.NEXT_PUBLIC_ML_URL || 'http://localhost:8000',
        response: null
      }
    },
    system: {
      nodeVersion: process.version,
      platform: os.platform(),
      arch: os.arch(),
      memory: {
        used: Math.round(process.memoryUsage().rss / (1024 * 1024)),
        total: Math.round(os.totalmem() / (1024 * 1024))
      }
    }
  };

  // Check backend health
  try {
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006';
    const response = await fetch(`${backendUrl}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // Add a short timeout to prevent hanging
      signal: AbortSignal.timeout(2000)
    });
    
    if (response.ok) {
      const data = await response.json();
      healthData.dependencies.backend = {
        status: 'healthy',
        url: backendUrl,
        response: data
      };
    } else {
      healthData.dependencies.backend = {
        status: 'degraded',
        url: backendUrl,
        response: { status: response.status, statusText: response.statusText }
      };
      healthData.status = 'degraded';
    }
  } catch (error) {
    console.error('Backend health check failed:', error);
    healthData.dependencies.backend = {
      status: 'unreachable',
      url: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006',
      response: null
    };
    healthData.status = 'degraded';
  }

  // Check ML system health
  try {
    const mlUrl = process.env.NEXT_PUBLIC_ML_URL || 'http://localhost:8000';
    const response = await fetch(`${mlUrl}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      // Add a short timeout to prevent hanging
      signal: AbortSignal.timeout(2000)
    });
    
    if (response.ok) {
      const data = await response.json();
      healthData.dependencies.mlSystem = {
        status: 'healthy',
        url: mlUrl,
        response: data
      };
    } else {
      healthData.dependencies.mlSystem = {
        status: 'degraded',
        url: mlUrl,
        response: { status: response.status, statusText: response.statusText }
      };
      healthData.status = 'degraded';
    }
  } catch (error) {
    console.error('ML system health check failed:', error);
    healthData.dependencies.mlSystem = {
      status: 'unreachable',
      url: process.env.NEXT_PUBLIC_ML_URL || 'http://localhost:8000',
      response: null
    };
    
    // Only mark as degraded if not already marked
    if (healthData.status === 'healthy') {
      healthData.status = 'degraded';
    }
  }

  return NextResponse.json(healthData);
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
