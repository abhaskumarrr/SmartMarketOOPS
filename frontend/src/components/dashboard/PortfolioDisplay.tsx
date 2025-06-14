'use client';

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Separator } from '../ui/separator';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  Eye,
  EyeOff,
  RefreshCw,
  AlertTriangle,
  BarChart3
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useRealTimeData } from '@/hooks/useRealTimeData';

interface PortfolioData {
  totalBalance: number;
  availableBalance: number;
  totalPnL: number;
  unrealizedPnL: number;
  realizedPnL: number;
  totalPositions: number;
  winRate: number;
  portfolioValue: number;
  marginUsed: number;
  marginAvailable: number;
  dayChange: number;
  dayChangePercent: number;
}

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  marginUsed: number;
  leverage: number;
}

interface PortfolioDisplayProps {
  className?: string;
  showPrivateMode?: boolean;
}

export default function PortfolioDisplay({ 
  className, 
  showPrivateMode = true 
}: PortfolioDisplayProps) {
  const [privateMode, setPrivateMode] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const { isConnected, lastPortfolioUpdate } = useRealTimeData();

  // Mock data for development - replace with real data from portfolioData
  const [portfolio, setPortfolio] = useState<PortfolioData>({
    totalBalance: 12345.67,
    availableBalance: 8901.23,
    totalPnL: 1234.56,
    unrealizedPnL: 456.78,
    realizedPnL: 777.78,
    totalPositions: 5,
    winRate: 68.5,
    portfolioValue: 13580.23,
    marginUsed: 3444.44,
    marginAvailable: 8901.23,
    dayChange: 234.56,
    dayChangePercent: 1.85
  });

  const [currentPositions, setCurrentPositions] = useState<Position[]>([
    {
      id: '1',
      symbol: 'BTCUSD',
      side: 'long',
      size: 0.5,
      entryPrice: 67500,
      currentPrice: 68200,
      pnl: 350,
      pnlPercent: 1.04,
      marginUsed: 1350,
      leverage: 25
    },
    {
      id: '2',
      symbol: 'ETHUSD',
      side: 'short',
      size: 2.0,
      entryPrice: 3450,
      currentPrice: 3420,
      pnl: 60,
      pnlPercent: 0.87,
      marginUsed: 276,
      leverage: 25
    }
  ]);

  // Update with real-time data when available
  useEffect(() => {
    if (lastPortfolioUpdate) {
      setPortfolio(prev => ({
        ...prev,
        totalBalance: lastPortfolioUpdate.totalBalance,
        availableBalance: lastPortfolioUpdate.availableBalance,
        totalPnL: lastPortfolioUpdate.totalPnL,
        dayChangePercent: lastPortfolioUpdate.totalPnLPercentage
      }));
    }
  }, [lastPortfolioUpdate]);

  useEffect(() => {
    if (lastPortfolioUpdate && lastPortfolioUpdate.positions && lastPortfolioUpdate.positions.length > 0) {
      const mappedPositions = lastPortfolioUpdate.positions.map((pos, index) => ({
        id: index.toString(),
        symbol: pos.symbol,
        side: pos.side,
        size: pos.size,
        entryPrice: pos.entryPrice,
        currentPrice: pos.currentPrice,
        pnl: pos.pnl,
        pnlPercent: pos.pnlPercentage,
        marginUsed: pos.size * pos.entryPrice / 25, // Assuming 25x leverage
        leverage: 25 // Default leverage
      }));
      setCurrentPositions(mappedPositions);
    }
  }, [lastPortfolioUpdate]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    // Simulate refresh delay
    setTimeout(() => {
      setIsRefreshing(false);
    }, 1000);
  };

  const formatCurrency = (value: number, hideIfPrivate = true) => {
    if (privateMode && hideIfPrivate) return '****';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  };

  const formatPercent = (value: number) => {
    if (privateMode) return '**%';
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const getPnLColor = (value: number) => {
    if (value > 0) return 'text-green-500';
    if (value < 0) return 'text-red-500';
    return 'text-gray-500';
  };

  const getPnLIcon = (value: number) => {
    return value >= 0 ? TrendingUp : TrendingDown;
  };

  const marginUsagePercent = (portfolio.marginUsed / (portfolio.marginUsed + portfolio.marginAvailable)) * 100;

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <DollarSign className="h-5 w-5 text-blue-500" />
            Portfolio Overview
            {!isConnected && (
              <Badge variant="destructive" className="ml-2">
                <AlertTriangle className="h-3 w-3 mr-1" />
                Offline
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            {showPrivateMode && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setPrivateMode(!privateMode)}
                className="h-8 w-8 p-0"
              >
                {privateMode ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="h-8 w-8 p-0"
            >
              <RefreshCw className={cn('h-4 w-4', isRefreshing && 'animate-spin')} />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Account Balance Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">Total Balance</span>
              <span className="font-semibold text-lg">
                {formatCurrency(portfolio.totalBalance)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">Available</span>
              <span className="text-sm">
                {formatCurrency(portfolio.availableBalance)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">Portfolio Value</span>
              <span className="text-sm">
                {formatCurrency(portfolio.portfolioValue)}
              </span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">Total P&L</span>
              <div className="flex items-center gap-1">
                {React.createElement(getPnLIcon(portfolio.totalPnL), {
                  className: cn('h-4 w-4', getPnLColor(portfolio.totalPnL))
                })}
                <span className={cn('font-semibold', getPnLColor(portfolio.totalPnL))}>
                  {formatCurrency(portfolio.totalPnL)}
                </span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">Unrealized P&L</span>
              <span className={cn('text-sm', getPnLColor(portfolio.unrealizedPnL))}>
                {formatCurrency(portfolio.unrealizedPnL)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">Realized P&L</span>
              <span className={cn('text-sm', getPnLColor(portfolio.realizedPnL))}>
                {formatCurrency(portfolio.realizedPnL)}
              </span>
            </div>
          </div>
        </div>

        <Separator />

        {/* Day Performance */}
        <div className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-blue-500" />
            <span className="text-sm font-medium">Today's Performance</span>
          </div>
          <div className="flex items-center gap-2">
            <span className={cn('font-semibold', getPnLColor(portfolio.dayChange))}>
              {formatCurrency(portfolio.dayChange)}
            </span>
            <Badge 
              variant={portfolio.dayChangePercent >= 0 ? 'default' : 'destructive'}
              className={portfolio.dayChangePercent >= 0 ? 'bg-green-100 text-green-800 hover:bg-green-200' : ''}
            >
              {formatPercent(portfolio.dayChangePercent)}
            </Badge>
          </div>
        </div>

        {/* Margin Usage */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-500">Margin Usage</span>
            <span className="text-sm font-medium">
              {formatCurrency(portfolio.marginUsed)} / {formatCurrency(portfolio.marginUsed + portfolio.marginAvailable)}
            </span>
          </div>
          <Progress 
            value={marginUsagePercent} 
            className={cn(
              'h-2',
              marginUsagePercent > 80 && 'bg-red-100',
              marginUsagePercent > 60 && marginUsagePercent <= 80 && 'bg-yellow-100'
            )}
          />
          <div className="flex justify-between text-xs text-gray-500">
            <span>{marginUsagePercent.toFixed(1)}% used</span>
            <span className={marginUsagePercent > 80 ? 'text-red-500 font-medium' : ''}>
              {marginUsagePercent > 80 ? 'High Risk' : 'Safe'}
            </span>
          </div>
        </div>

        <Separator />

        {/* Trading Statistics */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="space-y-1">
            <div className="text-2xl font-bold text-blue-500">
              {privateMode ? '**' : portfolio.totalPositions}
            </div>
            <div className="text-xs text-gray-500">Active Positions</div>
          </div>
          <div className="space-y-1">
            <div className="text-2xl font-bold text-green-500">
              {formatPercent(portfolio.winRate).replace('%', '')}%
            </div>
            <div className="text-xs text-gray-500">Win Rate</div>
          </div>
          <div className="space-y-1">
            <div className="text-2xl font-bold">
              {currentPositions.length}
            </div>
            <div className="text-xs text-gray-500">Live Trades</div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex gap-2 pt-2">
          <Button variant="outline" size="sm" className="flex-1">
            <BarChart3 className="h-4 w-4 mr-1" />
            Analytics
          </Button>
          <Button variant="outline" size="sm" className="flex-1">
            Export
          </Button>
        </div>
      </CardContent>
    </Card>
  );
} 