'use client';

import React, { useState, useEffect } from 'react';
import { UnifiedPageWrapper } from '../../components/layout/UnifiedPageWrapper';
import { TradingViewChart } from '../../components/charts/TradingViewChart';
import { ErrorBoundary } from '../../components/ErrorBoundary';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  DollarSign,
  Target,
  Shield,
  Zap,
  BarChart3,
  Wallet,
  Clock,
  AlertTriangle
} from 'lucide-react';

interface Portfolio {
  balance: number;
  totalPnL: number;
  totalUnrealizedPnL: number;
  currentBalance: number;
  positions: Position[];
  trades: Trade[];
}

interface Position {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  currentPrice?: number;
  unrealizedPnL?: number;
  stopLoss: number;
  takeProfitLevels: TakeProfitLevel[];
  openTime: string;
  status: 'open' | 'closed';
}

interface TakeProfitLevel {
  percentage: number;
  ratio: number;
  price: number;
  executed: boolean;
}

interface Trade {
  id: string;
  symbol: string;
  side: string;
  size: number;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  timestamp: string;
}

interface MarketData {
  [symbol: string]: {
    symbol: string;
    price: number;
    change: string;
    changePercent: number;
    volume: number;
    high24h: number;
    low24h: number;
    timestamp: string;
  };
}

interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export default function PaperTradingPage() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [marketData, setMarketData] = useState<MarketData>({});
  const [chartData, setChartData] = useState<CandleData[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('ETH/USDT');
  const [isConnected, setIsConnected] = useState<boolean>(false);
  // Use a static date for SSR to prevent hydration mismatch
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date('2024-01-01T00:00:00Z'));
  const [nextCycleCountdown, setNextCycleCountdown] = useState<number>(30);
  const [crosshairPrice, setCrosshairPrice] = useState<number | null>(null);
  const [isClient, setIsClient] = useState(false);

  // Fix hydration mismatch by ensuring client-side rendering for time
  useEffect(() => {
    setIsClient(true);
    // Set the actual current time once client-side
    setLastUpdate(new Date());
  }, []);

  // Helper function to format time consistently
  const formatTime = (date: Date): string => {
    if (!isClient) {
      return '--:--:--'; // Placeholder for SSR
    }
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // Fetch portfolio data
  const fetchPortfolio = async () => {
    try {
      const response = await fetch('/api/paper-trading/portfolio', {
        headers: {
          'Authorization': 'Bearer demo-token', // For development
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();

      if (data.success) {
        setPortfolio(data.data);
        setLastUpdate(new Date());
        setIsConnected(true);
      } else {
        console.error('Portfolio API error:', data.error || 'Unknown error');
        setIsConnected(false);
      }
    } catch (error) {
      console.error('Portfolio API error:', error);
      setIsConnected(false);
    }
  };

  // Fetch market data
  const fetchMarketData = async () => {
    try {
      const response = await fetch('/api/paper-trading/market-data', {
        headers: {
          'Authorization': 'Bearer demo-token', // For development
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();

      if (data.success) {
        setMarketData(data.data);
      } else {
        console.error('Market Data API error:', data.error || 'Unknown error');
      }
    } catch (error) {
      console.error('Market Data API error:', error);
    }
  };

  // Fetch chart data
  const fetchChartData = async (symbol: string) => {
    try {
      const response = await fetch(`/api/paper-trading/chart-data?symbol=${encodeURIComponent(symbol)}&timeframe=1m&limit=100`, {
        headers: {
          'Authorization': 'Bearer demo-token', // For development
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();

      if (data.success) {
        setChartData(data.data.candles);
      } else {
        console.error('Chart Data API error:', data.error || 'Unknown error');
      }
    } catch (error) {
      console.error('Chart Data API error:', error);
    }
  };

  // Initialize data fetching
  useEffect(() => {
    const fetchAllData = async () => {
      await Promise.all([
        fetchPortfolio(),
        fetchMarketData(),
        fetchChartData(selectedSymbol)
      ]);
    };

    fetchAllData();

    // Set up polling intervals
    const portfolioInterval = setInterval(fetchPortfolio, 5000); // Every 5 seconds
    const marketDataInterval = setInterval(fetchMarketData, 2000); // Every 2 seconds
    const chartInterval = setInterval(() => fetchChartData(selectedSymbol), 30000); // Every 30 seconds

    return () => {
      clearInterval(portfolioInterval);
      clearInterval(marketDataInterval);
      clearInterval(chartInterval);
    };
  }, [selectedSymbol]);

  // Countdown timer for next cycle
  useEffect(() => {
    const countdownInterval = setInterval(() => {
      setNextCycleCountdown(prev => {
        if (prev <= 1) {
          return 30; // Reset to 30 seconds
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(countdownInterval);
  }, []);

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const openPositions = portfolio?.positions?.filter(p => p.status === 'open') || [];
  const totalUnrealizedPnL = openPositions.reduce((sum, pos) => sum + (pos.unrealizedPnL || 0), 0);

  return (
    <ErrorBoundary>
      <UnifiedPageWrapper
        connectionStatus={isConnected ? 'connected' : 'disconnected'}
        showConnectionStatus={true}
        disablePadding={true}
      >
        {/* Page Header */}
        <div className="border-b border-slate-800 bg-slate-900/50 px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <TrendingUp className="w-8 h-8 text-emerald-500" />
                <div>
                  <h1 className="text-3xl font-bold text-white">Paper Trading Dashboard</h1>
                  <p className="text-slate-400 mt-1">Live Delta Exchange Indian Testnet Trading</p>
                </div>
                <span className="px-3 py-1 text-sm font-medium bg-emerald-500/20 text-emerald-400 rounded-full">
                  Live
                </span>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Next Cycle Countdown */}
              <div className="flex items-center space-x-2 bg-slate-800 px-3 py-1.5 rounded-lg">
                <Clock className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-blue-400">
                  Next cycle: {nextCycleCountdown}s
                </span>
              </div>

              <div className="text-right">
                <p className="text-sm text-slate-400">Last Update</p>
                <p className="text-white font-medium">{formatTime(lastUpdate)}</p>
              </div>
            </div>
          </div>
        </div>

      {/* Main Content */}
      <div className="flex-1 p-6 space-y-6">
        {/* Portfolio Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Total Balance Card */}
          <div className="trading-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium">Total Balance</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {formatCurrency(portfolio?.currentBalance || 0)}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Initial: {formatCurrency(portfolio?.balance || 0)}
                </p>
              </div>
              <div className="w-12 h-12 bg-blue-500/20 rounded-xl flex items-center justify-center">
                <Wallet className="w-6 h-6 text-blue-400" />
              </div>
            </div>
          </div>

          {/* Realized P&L Card */}
          <div className="trading-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium">Realized P&L</p>
                <p className={`text-3xl font-bold mt-2 ${(portfolio?.totalPnL || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {formatCurrency(portfolio?.totalPnL || 0)}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  From closed trades
                </p>
              </div>
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${(portfolio?.totalPnL || 0) >= 0 ? 'bg-emerald-500/20' : 'bg-red-500/20'}`}>
                {(portfolio?.totalPnL || 0) >= 0 ? (
                  <TrendingUp className="w-6 h-6 text-emerald-400" />
                ) : (
                  <TrendingDown className="w-6 h-6 text-red-400" />
                )}
              </div>
            </div>
          </div>

          {/* Unrealized P&L Card */}
          <div className="trading-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium">Unrealized P&L</p>
                <p className={`text-3xl font-bold mt-2 ${totalUnrealizedPnL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {formatCurrency(totalUnrealizedPnL)}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  From open positions
                </p>
              </div>
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${totalUnrealizedPnL >= 0 ? 'bg-emerald-500/20' : 'bg-red-500/20'}`}>
                <Activity className={`w-6 h-6 ${totalUnrealizedPnL >= 0 ? 'text-emerald-400' : 'text-red-400'}`} />
              </div>
            </div>
          </div>

          {/* Open Positions Card */}
          <div className="trading-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-400 text-sm font-medium">Open Positions</p>
                <p className="text-3xl font-bold text-white mt-2">
                  {openPositions.length}
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  Active trades
                </p>
              </div>
              <div className="w-12 h-12 bg-purple-500/20 rounded-xl flex items-center justify-center">
                <Target className="w-6 h-6 text-purple-400" />
              </div>
            </div>
          </div>
        </div>

        {/* Trading Chart */}
        <div className="trading-card">
          <div className="trading-card-header">
            <div className="trading-card-title">
              <BarChart3 className="w-5 h-5" />
              Trading Chart
            </div>
            <div className="flex items-center space-x-2">
              {/* Symbol Selector */}
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ETH/USDT">ETH/USDT</option>
                <option value="BTC/USDT">BTC/USDT</option>
                <option value="SOL/USDT">SOL/USDT</option>
              </select>

              {/* Crosshair Price Display */}
              {crosshairPrice && (
                <div className="px-3 py-1.5 bg-slate-800 rounded-lg text-sm font-mono text-white">
                  ${crosshairPrice.toFixed(2)}
                </div>
              )}
            </div>
          </div>

          <TradingViewChart
            symbol={selectedSymbol}
            data={chartData}
            positions={openPositions}
            height={500}
            onCrosshairMove={setCrosshairPrice}
          />
        </div>

        {/* Market Data and Positions */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Market Data */}
          <div className="trading-card">
            <div className="trading-card-header">
              <div className="trading-card-title">
                <BarChart3 className="w-5 h-5" />
                Live Market Data
              </div>
              <span className="px-2 py-1 text-xs font-medium bg-emerald-500/20 text-emerald-400 rounded-full">
                Real-time
              </span>
            </div>
            <div className="space-y-4">
              {Object.values(marketData).map((data) => (
                <div key={data.symbol} className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg border border-slate-700/50 hover:border-slate-600 transition-colors">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg flex items-center justify-center">
                      <span className="text-sm font-bold text-white">{data.symbol.split('/')[0]}</span>
                    </div>
                    <div>
                      <h4 className="font-semibold text-white">{data.symbol}</h4>
                      <p className="text-xs text-slate-400">
                        Vol: {data.volume.toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-bold text-white">
                      {formatCurrency(data.price)}
                    </div>
                    <div className={`text-sm font-medium flex items-center ${
                      data.changePercent >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {data.changePercent >= 0 ? (
                        <TrendingUp className="w-3 h-3 mr-1" />
                      ) : (
                        <TrendingDown className="w-3 h-3 mr-1" />
                      )}
                      {formatPercentage(data.changePercent)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Active Positions */}
          <div className="trading-card">
            <div className="trading-card-header">
              <div className="trading-card-title">
                <Target className="w-5 h-5" />
                Active Positions
              </div>
              <span className="px-2 py-1 text-xs font-medium bg-blue-500/20 text-blue-400 rounded-full">
                {openPositions.length} Open
              </span>
            </div>

            {openPositions.length === 0 ? (
              <div className="text-center py-12">
                <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                  <AlertTriangle className="w-8 h-8 text-slate-500" />
                </div>
                <p className="text-slate-400 font-medium">No open positions</p>
                <p className="text-sm text-slate-500 mt-1">Waiting for trading signals...</p>
              </div>
            ) : (
              <div className="space-y-4">
                {openPositions.map((position) => {
                  const pnlPercent = position.unrealizedPnL && position.entryPrice && position.size
                    ? (position.unrealizedPnL / (position.entryPrice * position.size)) * 100
                    : 0;

                  return (
                    <div key={position.id} className="p-4 bg-slate-800/50 rounded-lg border border-slate-700/50 hover:border-slate-600 transition-colors">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg flex items-center justify-center">
                            <span className="text-sm font-bold text-white">{position.symbol.split('/')[0]}</span>
                          </div>
                          <div>
                            <span className="font-semibold text-white">{position.symbol}</span>
                            <div className="flex items-center space-x-2 mt-1">
                              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                                position.side === 'buy'
                                  ? 'bg-emerald-500/20 text-emerald-400'
                                  : 'bg-red-500/20 text-red-400'
                              }`}>
                                {position.side.toUpperCase()}
                              </span>
                              <span className="text-xs text-slate-400 font-mono">
                                {position.size.toFixed(4)}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`text-lg font-bold ${
                            (position.unrealizedPnL || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                          }`}>
                            {position.unrealizedPnL ? formatCurrency(position.unrealizedPnL) : '-'}
                          </div>
                          <div className={`text-sm font-medium ${
                            pnlPercent >= 0 ? 'text-emerald-400' : 'text-red-400'
                          }`}>
                            {formatPercentage(pnlPercent)}
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-slate-400">Entry Price</span>
                          <div className="text-white font-mono font-medium">{formatCurrency(position.entryPrice)}</div>
                        </div>
                        <div>
                          <span className="text-slate-400">Current Price</span>
                          <div className="text-white font-mono font-medium">
                            {position.currentPrice ? formatCurrency(position.currentPrice) : '-'}
                          </div>
                        </div>
                      </div>

                      <div className="mt-4 pt-3 border-t border-slate-700">
                        <div className="flex justify-between items-center text-xs">
                          <div className="flex items-center space-x-1 text-red-400">
                            <Shield className="w-3 h-3" />
                            <span>Stop: {formatCurrency(position.stopLoss)}</span>
                          </div>
                          <div className="flex items-center space-x-1 text-emerald-400">
                            <Target className="w-3 h-3" />
                            <span>{position.takeProfitLevels.filter(tp => !tp.executed).length} TPs Active</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* System Status Footer */}
        <div className="trading-card">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></div>
                <span className={`text-sm font-medium ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
                  {isConnected ? 'Connected to Delta Exchange' : 'Disconnected'}
                </span>
              </div>
              <div className="text-sm text-slate-400">
                Indian Testnet Environment
              </div>
              <div className="text-sm text-slate-400">
                Last update: {formatTime(lastUpdate)}
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <span className="px-3 py-1 rounded-lg text-xs font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/20">
                System Healthy
              </span>
              <span className="px-3 py-1 rounded-lg text-xs font-medium bg-blue-500/20 text-blue-400 border border-blue-500/20">
                AI Active
              </span>
            </div>
          </div>
        </div>
      </div>
      </UnifiedPageWrapper>
    </ErrorBoundary>
  );
}
