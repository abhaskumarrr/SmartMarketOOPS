'use client';

import React, { useState, useRef, useEffect } from 'react';
import TradingViewWidget, { TradingViewWidgetRef } from '../charts/TradingViewWidget';
import TradeExecutionPanel from './TradeExecutionPanel';
import PositionManagementPanel from './PositionManagementPanel';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Separator } from '../ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import ApiService from '@/services/api';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  BarChart3,
  Settings,
  Maximize2,
  Minimize2,
  RefreshCw,
  Wifi,
  WifiOff,
  Brain,
  Target,
  Shield,
  Zap
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useBreakpoints } from '@/hooks/use-responsive';

const EnhancedTradingDashboard: React.FC = () => {
  const [isChartExpanded, setIsChartExpanded] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [isConnected, setIsConnected] = useState(true);
  const [activeTab, setActiveTab] = useState('trade');
  const [signal, setSignal] = useState<any>(null);
  const [isLoadingSignal, setIsLoadingSignal] = useState(false);
  const [dashboardData, setDashboardData] = useState<any>(null);
  const [isLoadingDashboard, setIsLoadingDashboard] = useState(true);
  const chartRef = useRef<TradingViewWidgetRef>(null);
  const { isMobile, isTablet } = useBreakpoints();

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      const apiService = new ApiService();
      const result = await apiService.getDashboardSummary();
      if (result.success) {
        setDashboardData(result.data);
        setIsConnected(true);
      }
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setIsConnected(false);
    } finally {
      setIsLoadingDashboard(false);
    }
  };

  useEffect(() => {
    if (isMobile && isChartExpanded) {
      setIsChartExpanded(false);
    }
  }, [isMobile, isChartExpanded]);

  const getChartHeight = () => {
    if (isChartExpanded) {
      return window.innerHeight - 140;
    }
    if (isMobile) {
      return 300;
    }
    if (isTablet) {
      return 350;
    }
    return 400;
  };

  const portfolioData = dashboardData ? {
    totalBalance: parseFloat(dashboardData.portfolioValue?.replace(/,/g, '') || '0'),
    availableBalance: parseFloat(dashboardData.availableBalance?.replace(/,/g, '') || '0'),
    totalPnL: parseFloat(dashboardData.totalPnl?.replace(/[+,]/g, '') || '0'),
    totalPnLPercentage: parseFloat(dashboardData.totalPnlPercentage?.replace(/[+%]/g, '') || '0'),
    positions: dashboardData.activePositions || 0,
  } : null;

  const marketSymbols = dashboardData?.marketData || [
    { symbol: 'BTCUSDT', name: 'Bitcoin', price: '48250.45', changePercentage24h: '2.34' },
    { symbol: 'ETHUSD', name: 'Ethereum', price: '2870.12', changePercentage24h: '-1.23' },
    { symbol: 'SOLUSD', name: 'Solana', price: '106.78', changePercentage24h: '5.67' },
  ];

  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
  };

  const handleGetSignal = async () => {
    setIsLoadingSignal(true);
    setSignal(null);
    try {
      const apiService = new ApiService();
      const result = await apiService.getTradingSignal();
      setSignal(result);
    } catch (error) {
      console.error("Failed to get trading signal:", error);
      setSignal({ error: "Failed to fetch signal." });
    } finally {
      setIsLoadingSignal(false);
    }
  };

  const handleChartToggle = () => {
    setIsChartExpanded(!isChartExpanded);
  };

  const handleRefreshData = () => {
    if (chartRef.current) {
      chartRef.current.fitContent();
    }
    fetchDashboardData();
  };

  const getSignalColor = (signal: string) => {
    switch (signal?.toLowerCase()) {
      case 'buy': return 'text-green-500';
      case 'sell': return 'text-red-500';
      case 'hold':
      default: return 'text-yellow-500';
    }
  };

  const getSignalIcon = (signal: string) => {
    switch (signal?.toLowerCase()) {
      case 'buy': return <TrendingUp className="w-4 h-4" />;
      case 'sell': return <TrendingDown className="w-4 h-4" />;
      case 'hold':
      default: return <Target className="w-4 h-4" />;
    }
  };

  return (
    <div className="w-full min-h-screen bg-background text-foreground p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Enhanced Trading Dashboard</h1>
        <div className="flex items-center gap-4">
          <Badge variant={isConnected ? "default" : "destructive"} className="flex items-center gap-1">
            {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </Badge>
          <Button variant="outline" size="sm" onClick={fetchDashboardData}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Portfolio Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <DollarSign className="w-5 h-5" />
              Portfolio Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            {isLoadingDashboard ? (
              <div className="text-center text-muted-foreground">Loading...</div>
            ) : portfolioData ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Balance</p>
                    <p className="text-xl font-bold">${portfolioData.totalBalance.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Available</p>
                    <p className="text-xl font-bold">${portfolioData.availableBalance.toLocaleString()}</p>
                  </div>
                </div>
                <Separator />
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Total P&L</p>
                    <p className={cn(
                      "text-lg font-bold",
                      portfolioData.totalPnL >= 0 ? "text-green-500" : "text-red-500"
                    )}>
                      {portfolioData.totalPnL >= 0 ? '+' : ''}${portfolioData.totalPnL.toLocaleString()}
                    </p>
                    <p className={cn(
                      "text-sm",
                      portfolioData.totalPnLPercentage >= 0 ? "text-green-500" : "text-red-500"
                    )}>
                      {portfolioData.totalPnLPercentage >= 0 ? '+' : ''}{portfolioData.totalPnLPercentage}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Active Positions</p>
                    <p className="text-lg font-bold">{portfolioData.positions}</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground">No data available</div>
            )}
          </CardContent>
        </Card>

        {/* AI Signal Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="w-5 h-5 text-blue-500" />
              AI Trading Signal
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {dashboardData?.aiSignal && (
              <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950 dark:to-purple-950 rounded-lg border">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getSignalIcon(dashboardData.aiSignal)}
                    <span className={cn("font-bold uppercase", getSignalColor(dashboardData.aiSignal))}>
                      {dashboardData.aiSignal}
                    </span>
                  </div>
                  <Badge variant="outline">
                    <Zap className="w-3 h-3 mr-1" />
                    {(dashboardData.aiAccuracy || 75).toFixed(1)}%
                  </Badge>
                </div>
                {dashboardData.signalConfidence && (
                  <div className="text-sm text-muted-foreground">
                    Confidence: {(dashboardData.signalConfidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            )}

            <Button 
              onClick={handleGetSignal} 
              disabled={isLoadingSignal} 
              className="w-full"
            >
              <Brain className="w-4 h-4 mr-2" />
              {isLoadingSignal ? 'Getting Signal...' : 'Get New Prediction'}
            </Button>
            
            {signal && (
              <div className="space-y-2">
                <div className="text-sm font-medium">Latest Prediction:</div>
                <pre className="p-3 bg-muted rounded-md text-xs overflow-auto max-h-32">
                  {JSON.stringify(signal, null, 2)}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Trading Stats Card */}
        {dashboardData && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5 text-green-500" />
                Trading Statistics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Win Rate</p>
                  <p className="text-lg font-bold text-green-500">{dashboardData.winRate || 72.5}%</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Total Trades</p>
                  <p className="text-lg font-bold">{dashboardData.totalTrades || 124}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Active Bots</p>
                  <p className="text-lg font-bold">{dashboardData.activeBots || 3}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Risk Level</p>
                  <p className="text-lg font-bold capitalize">{dashboardData.riskLevel || 'Medium'}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Market Data Card */}
        <Card className="md:col-span-2 lg:col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Market Overview
            </CardTitle>
          </CardHeader>
          <CardContent>
            {dashboardData?.marketData ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {dashboardData.marketData.map((market: any) => (
                  <div key={market.symbol} className="p-4 border rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <p className="font-semibold">{market.symbol}</p>
                        <p className="text-2xl font-bold">${parseFloat(market.price).toLocaleString()}</p>
                      </div>
                      <Badge variant={parseFloat(market.changePercentage24h) >= 0 ? "default" : "destructive"}>
                        {parseFloat(market.changePercentage24h) >= 0 ? '+' : ''}{market.changePercentage24h}%
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      <p>24h Volume: ${parseFloat(market.volume24h || 0).toLocaleString()}</p>
                      <p>High: ${parseFloat(market.high24h || market.price).toLocaleString()}</p>
                      <p>Low: ${parseFloat(market.low24h || market.price).toLocaleString()}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Market data will appear here once loaded
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default EnhancedTradingDashboard; 