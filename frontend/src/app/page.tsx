'use client';

import { useEffect, useState } from 'react';

// Use environment variable or fallback to the correct backend URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3006/api';

interface DashboardData {
  portfolioValue?: string;
  dailyChange?: string;
  activePositions?: number;
  profitablePositions?: number;
  aiAccuracy?: number;
  totalPnl?: string;
  totalPnlPercentage?: string;
  tradingStatus?: string;
}

export default function Home() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('Fetching data from:', `${API_BASE_URL}/dashboard/dashboard-summary`);
        
        const response = await fetch(`${API_BASE_URL}/dashboard/dashboard-summary`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'same-origin'
        });

        console.log('Response status:', response.status);
        console.log('Response headers:', Object.fromEntries(response.headers.entries()));

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('API Response:', result);

        // Extract data from the response
        const dashboardData = result?.data || result;
        setData(dashboardData);
      } catch (err) {
        console.error('API Error:', err);
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-lg">Loading dashboard data...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
          <div className="text-lg text-red-600">Failed to load dashboard data</div>
          <div className="text-sm text-gray-600">Error: {error}</div>
          <div className="text-xs text-gray-500">API URL: {API_BASE_URL}/dashboard/dashboard-summary</div>
          <button 
            onClick={() => window.location.reload()} 
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Welcome to SmartMarketOOPS</h1>
        <p className="text-muted-foreground">
          Your AI-powered trading dashboard
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
          <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="tracking-tight text-sm font-medium">Portfolio Value</h3>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              className="h-4 w-4 text-muted-foreground"
            >
              <path d="M12 2v20m8-10H4" />
            </svg>
          </div>
          <div className="p-6 pt-0">
            <div className="text-2xl font-bold">${data?.portfolioValue || 'N/A'}</div>
            <p className="text-xs text-muted-foreground">
              {data?.dailyChange || 'No change data'}
            </p>
          </div>
        </div>

        <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
          <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="tracking-tight text-sm font-medium">Active Positions</h3>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              className="h-4 w-4 text-muted-foreground"
            >
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
          <div className="p-6 pt-0">
            <div className="text-2xl font-bold">{data?.activePositions || 0}</div>
            <p className="text-xs text-muted-foreground">
              {data?.profitablePositions || 0} profitable
            </p>
          </div>
        </div>

        <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
          <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="tracking-tight text-sm font-medium">AI Accuracy</h3>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              className="h-4 w-4 text-muted-foreground"
            >
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          </div>
          <div className="p-6 pt-0">
            <div className="text-2xl font-bold">{data?.aiAccuracy || 0}%</div>
            <p className="text-xs text-muted-foreground">
              Prediction accuracy
            </p>
          </div>
        </div>

        <div className="rounded-lg border bg-card text-card-foreground shadow-sm">
          <div className="p-6 flex flex-row items-center justify-between space-y-0 pb-2">
            <h3 className="tracking-tight text-sm font-medium">Total P&L</h3>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              className="h-4 w-4 text-muted-foreground"
            >
              <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
          </div>
          <div className="p-6 pt-0">
            <div className="text-2xl font-bold text-green-600">{data?.totalPnl || '$0'}</div>
            <p className="text-xs text-muted-foreground">
              {data?.totalPnlPercentage || '0%'} today
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <div className="rounded-lg border bg-card text-card-foreground shadow-sm col-span-4">
          <div className="p-6">
            <h3 className="text-lg font-semibold">Trading Status</h3>
            <div className="mt-4">
              <div className="flex items-center space-x-2">
                <div className={`h-3 w-3 rounded-full ${
                  data?.tradingStatus === 'active' ? 'bg-green-500' : 'bg-gray-400'
                }`}></div>
                <span className="text-sm font-medium">
                  Status: {data?.tradingStatus === 'active' ? 'Active' : 'Inactive'}
                </span>
              </div>
            </div>
          </div>
      </div>

        <div className="rounded-lg border bg-card text-card-foreground shadow-sm col-span-3">
          <div className="p-6">
            <h3 className="text-lg font-semibold">Quick Actions</h3>
            <div className="mt-4 space-y-2">
              <button className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
                View Portfolio
              </button>
              <button className="w-full px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">
                Start Trading
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-6 p-4 bg-gray-100 rounded-lg">
        <h4 className="text-sm font-medium text-gray-700">Debug Information</h4>
        <div className="mt-2 space-y-1 text-xs text-gray-600">
          <div>API URL: {API_BASE_URL}/dashboard/dashboard-summary</div>
          <div>Data loaded: {data ? 'Yes' : 'No'}</div>
          <div>Raw data: {JSON.stringify(data, null, 2)}</div>
        </div>
      </div>
    </div>
  );
}