'use client';

import React, { useState, useRef } from 'react';
import TradingViewWidget, { TradingViewWidgetRef } from '../charts/TradingViewWidget';
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
  WifiOff
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface TradingDashboardProps {
  className?: string;
}

const TradingDashboard: React.FC<TradingDashboardProps> = ({ className }) => {
  const [isChartExpanded, setIsChartExpanded] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [isConnected, setIsConnected] = useState(true);
  const chartRef = useRef<TradingViewWidgetRef>(null);

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
    <div className={cn("w-full h-screen bg-background text-foreground", className)}>
      {/* Top Navigation Bar */}
      <div className="flex items-center justify-between p-4 border-b bg-card">
        <div className="flex items-center gap-4">
          <h1 className="text-xl font-semibold">Trading Dashboard</h1>
          <Badge variant={isConnected ? "default" : "destructive"} className="flex items-center gap-1">
            {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>
        
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={handleRefreshData}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
          <Button variant="ghost" size="sm">
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className={cn(
        "grid gap-4 p-4 h-[calc(100vh-80px)]",
        isChartExpanded 
          ? "grid-cols-1 grid-rows-1" 
          : "grid-cols-12 grid-rows-12 lg:grid-rows-8"
      )}>
        
        {/* Chart Section */}
        <Card className={cn(
          "overflow-hidden",
          isChartExpanded 
            ? "col-span-12 row-span-12" 
            : "col-span-12 lg:col-span-8 row-span-8 lg:row-span-6"
        )}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <div className="flex items-center gap-4">
              <CardTitle className="text-lg font-semibold">{selectedSymbol}</CardTitle>
              <div className="flex gap-2">
                {marketSymbols.map((market) => (
                  <Button
                    key={market.symbol}
                    variant={selectedSymbol === market.symbol ? "default" : "ghost"}
                    size="sm"
                    onClick={() => handleSymbolChange(market.symbol)}
                    className="text-xs"
                  >
                    {market.symbol}
                  </Button>
                ))}
              </div>
            </div>
            
            <div className="flex items-center gap-2">
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
              <Button variant="ghost" size="sm" onClick={handleChartToggle}>
                {isChartExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            <TradingViewWidget
              ref={chartRef}
              symbol={selectedSymbol}
              height={isChartExpanded ? window.innerHeight - 140 : 400}
              className="w-full"
            />
          </CardContent>
        </Card>

        {/* Right Sidebar - Only show when chart is not expanded */}
        {!isChartExpanded && (
          <>
            {/* Portfolio Overview */}
            <Card className="col-span-12 lg:col-span-4 row-span-4 lg:row-span-3">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <DollarSign className="w-5 h-5" />
                  Portfolio
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Total Balance</p>
                    <p className="text-lg font-semibold">${portfolioData.totalBalance.toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Available</p>
                    <p className="text-lg font-semibold">${portfolioData.availableBalance.toLocaleString()}</p>
                  </div>
                </div>
                
                <Separator />
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-muted-foreground">Total P&L</p>
                    <p className={cn(
                      "text-lg font-semibold",
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
                    <p className="text-sm text-muted-foreground">Today's P&L</p>
                    <p className={cn(
                      "text-lg font-semibold",
                      portfolioData.todaysPnL >= 0 ? "text-green-500" : "text-red-500"
                    )}>
                      {portfolioData.todaysPnL >= 0 ? '+' : ''}${portfolioData.todaysPnL.toLocaleString()}
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Trading Panel */}
            <Card className="col-span-12 lg:col-span-4 row-span-4 lg:row-span-3">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="w-5 h-5" />
                  Trade Execution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="buy" className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="buy" className="text-green-600">Buy</TabsTrigger>
                    <TabsTrigger value="sell" className="text-red-600">Sell</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="buy" className="space-y-4 mt-4">
                    <div className="text-center text-sm text-muted-foreground">
                      Trading controls will be implemented here
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="sell" className="space-y-4 mt-4">
                    <div className="text-center text-sm text-muted-foreground">
                      Trading controls will be implemented here
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>

            {/* Positions & Market Data */}
            <Card className="col-span-12 row-span-4 lg:row-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Positions & Market Overview
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="positions" className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="positions">Positions ({portfolioData.positions})</TabsTrigger>
                    <TabsTrigger value="market">Market</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="positions" className="mt-4">
                    <div className="text-center text-sm text-muted-foreground">
                      Position management interface will be implemented here
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="market" className="mt-4">
                    <div className="space-y-2">
                      {marketSymbols.map((market) => (
                        <div key={market.symbol} className="flex items-center justify-between p-2 rounded hover:bg-accent">
                          <div>
                            <span className="font-medium">{market.symbol}</span>
                            <span className="text-sm text-muted-foreground ml-2">{market.name}</span>
                          </div>
                          <div className="text-right">
                            <div className="font-medium">${market.price.toLocaleString()}</div>
                            <div className={cn(
                              "text-xs",
                              market.change >= 0 ? "text-green-500" : "text-red-500"
                            )}>
                              {market.change >= 0 ? '+' : ''}{market.change}%
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </div>
  );
};

export default TradingDashboard; 