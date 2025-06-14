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
  ChevronRight,
  ChevronLeft
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useBreakpoints } from '@/hooks/use-responsive';

interface TradingDashboardProps {
  className?: string;
}

const TradingDashboard: React.FC<TradingDashboardProps> = ({ className }) => {
  const [isChartExpanded, setIsChartExpanded] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [isConnected, setIsConnected] = useState(true);
  const [activeTab, setActiveTab] = useState('trade');
  const chartRef = useRef<TradingViewWidgetRef>(null);
  const { isMobile, isTablet, atLeast, breakpoints } = useBreakpoints();

  // Adjust chart expanded state based on screen size
  useEffect(() => {
    // Automatically collapse chart on small screens
    if (isMobile && isChartExpanded) {
      setIsChartExpanded(false);
    }
  }, [isMobile, isChartExpanded]);

  // Calculate chart height based on container and screen size
  const getChartHeight = () => {
    if (isChartExpanded) {
      return window.innerHeight - 140; // Full-screen minus header
    }
    if (isMobile) {
      return 300; // Smaller height on mobile
    }
    if (isTablet) {
      return 350; // Medium height on tablets
    }
    return 400; // Default height on desktop
  };

  // Mock data - this will be replaced with real data from the trading hooks
  const portfolioData = {
    totalBalance: 12485.67,
    availableBalance: 8234.12,
    totalPnL: 1247.89,
    totalPnLPercentage: 11.2,
    positions: 3,
    todaysPnL: 234.56
  };

  const marketSymbols = [
    { symbol: 'BTCUSDT', name: 'Bitcoin', price: 48250.45, change: 2.34 },
    { symbol: 'ETHUSD', name: 'Ethereum', price: 2870.12, change: -1.23 },
    { symbol: 'SOLUSD', name: 'Solana', price: 106.78, change: 5.67 },
    { symbol: 'BNBUSD', name: 'BNB', price: 570.34, change: 1.89 }
  ];

  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
    // Request new data for the symbol
    // This will be integrated with the real data service
  };

  const handleChartToggle = () => {
    setIsChartExpanded(!isChartExpanded);
  };

  const handleRefreshData = () => {
    if (chartRef.current) {
      chartRef.current.fitContent();
    }
    // Refresh all real-time data
  };

  return (
    <div className={cn("w-full min-h-screen bg-background text-foreground overflow-x-hidden", className)}>
      {/* Top Navigation Bar - Responsive */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between p-3 sm:p-4 border-b bg-card">
        <div className="flex items-center justify-between sm:justify-start gap-2 sm:gap-4 mb-2 sm:mb-0">
          <h1 className="text-lg sm:text-xl font-semibold">Trading Dashboard</h1>
          <Badge variant={isConnected ? "default" : "destructive"} className="flex items-center gap-1">
            {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>
        
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleRefreshData} className="h-8 px-2 sm:px-3">
            <RefreshCw className="w-4 h-4 sm:mr-2" />
            <span className="hidden sm:inline">Refresh</span>
          </Button>
          <Button variant="ghost" size="sm" className="h-8 px-2 sm:px-3">
            <Settings className="w-4 h-4 sm:mr-2" />
            <span className="hidden sm:inline">Settings</span>
          </Button>
        </div>
      </div>

      {/* Main Dashboard Grid - Mobile-first approach */}
      <div className={cn(
        "grid gap-2 sm:gap-4 p-2 sm:p-4 min-h-[calc(100vh-80px)]",
        isChartExpanded 
          ? "grid-cols-1" 
          : "grid-cols-1 lg:grid-cols-12"
      )}>
        
        {/* Chart Section */}
        <Card className={cn(
          "overflow-hidden",
          isChartExpanded 
            ? "col-span-1" 
            : "col-span-1 lg:col-span-8 xl:col-span-9"
        )}>
          <CardHeader className="flex flex-col sm:flex-row items-start sm:items-center justify-between space-y-2 sm:space-y-0 py-2 px-3 sm:px-4 sm:pb-2">
            <div className="flex flex-col sm:flex-row sm:items-center gap-2 w-full sm:w-auto">
              <CardTitle className="text-base sm:text-lg font-semibold">{selectedSymbol}</CardTitle>
              <div className="flex gap-1 sm:gap-2 overflow-x-auto pb-1 sm:pb-0 w-full sm:w-auto">
                {marketSymbols.map((market) => (
                  <Button
                    key={market.symbol}
                    variant={selectedSymbol === market.symbol ? "default" : "ghost"}
                    size="sm"
                    onClick={() => handleSymbolChange(market.symbol)}
                    className="text-xs h-7 px-2"
                  >
                    {market.symbol}
                  </Button>
                ))}
              </div>
            </div>
            
            <div className="flex items-center justify-between sm:justify-end gap-2 w-full sm:w-auto">
              <div className="text-right">
                <div className="text-sm font-medium">
                  ${marketSymbols.find(m => m.symbol === selectedSymbol)?.price.toLocaleString()}
                </div>
                <div className={cn(
                  "text-xs flex items-center",
                  (marketSymbols.find(m => m.symbol === selectedSymbol)?.change || 0) >= 0 
                    ? "text-green-500" 
                    : "text-red-500"
                )}>
                  {(marketSymbols.find(m => m.symbol === selectedSymbol)?.change || 0) >= 0 
                    ? <TrendingUp className="w-3 h-3 mr-1" />
                    : <TrendingDown className="w-3 h-3 mr-1" />
                  }
                  {marketSymbols.find(m => m.symbol === selectedSymbol)?.change}%
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={handleChartToggle} className="h-8 w-8 p-0">
                {isChartExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <TradingViewWidget
              ref={chartRef}
              symbol={selectedSymbol}
              height={getChartHeight()}
              className="w-full"
            />
          </CardContent>
        </Card>

        {/* Right Sidebar - Only show when chart is not expanded */}
        {!isChartExpanded && (
          <div className="col-span-1 lg:col-span-4 xl:col-span-3 space-y-2 sm:space-y-4">
            {/* Portfolio Overview */}
            <Card>
              <CardHeader className="py-2 px-3 sm:px-4">
                <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                  <DollarSign className="w-4 h-4 sm:w-5 sm:h-5" />
                  Portfolio
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 sm:space-y-4 px-3 sm:px-4 py-2 sm:py-3">
                <div className="grid grid-cols-2 gap-3 sm:gap-4">
                  <div>
                    <p className="text-xs sm:text-sm text-muted-foreground">Total Balance</p>
                    <p className="text-base sm:text-lg font-semibold">${portfolioData.totalBalance.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-xs sm:text-sm text-muted-foreground">Available</p>
                    <p className="text-base sm:text-lg font-semibold">${portfolioData.availableBalance.toLocaleString()}</p>
                  </div>
                </div>
                
                <Separator className="my-1 sm:my-2" />
                
                <div className="grid grid-cols-2 gap-3 sm:gap-4">
                  <div>
                    <p className="text-xs sm:text-sm text-muted-foreground">Total P&L</p>
                    <p className={cn(
                      "text-base sm:text-lg font-semibold",
                      portfolioData.totalPnL >= 0 ? "text-green-500" : "text-red-500"
                    )}>
                      {portfolioData.totalPnL >= 0 ? '+' : ''}${portfolioData.totalPnL.toLocaleString()}
                    </p>
                    <p className={cn(
                      "text-xs",
                      portfolioData.totalPnLPercentage >= 0 ? "text-green-500" : "text-red-500"
                    )}>
                      {portfolioData.totalPnLPercentage >= 0 ? '+' : ''}{portfolioData.totalPnLPercentage}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs sm:text-sm text-muted-foreground">Today's P&L</p>
                    <p className={cn(
                      "text-base sm:text-lg font-semibold",
                      portfolioData.todaysPnL >= 0 ? "text-green-500" : "text-red-500"
                    )}>
                      {portfolioData.todaysPnL >= 0 ? '+' : ''}${portfolioData.todaysPnL.toLocaleString()}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Trading Panels - Tabs for small/medium screens, side by side for larger screens */}
            <div className="lg:hidden">
              <Tabs defaultValue="trade" value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="w-full grid grid-cols-2">
                  <TabsTrigger value="trade">Trade</TabsTrigger>
                  <TabsTrigger value="positions">Positions</TabsTrigger>
                </TabsList>
                <TabsContent value="trade" className="mt-2">
                  <Card>
                    <CardContent className="p-0">
                      <TradeExecutionPanel symbol={selectedSymbol} compact={true} />
                    </CardContent>
                  </Card>
                </TabsContent>
                <TabsContent value="positions" className="mt-2">
                  <Card>
                    <CardContent className="p-0">
                      <PositionManagementPanel compact={true} />
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>

            {/* Trading panels side by side for larger screens */}
            <div className="hidden lg:grid lg:grid-cols-1 lg:gap-4">
              <Card>
                <CardHeader className="py-2 px-4">
                  <CardTitle className="text-base">Trade {selectedSymbol}</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <TradeExecutionPanel symbol={selectedSymbol} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="py-2 px-4">
                  <CardTitle className="text-base">Positions</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <PositionManagementPanel />
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradingDashboard; 