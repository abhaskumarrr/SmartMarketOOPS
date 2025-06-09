"use client"

import React, { useState, useEffect } from 'react'
import { TradingChart } from '@/components/charts/TradingChart'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { useAllMarketData, usePortfolioData, useRealtimeStatus, useTradeSignals } from '@/lib/realtime-data'
import { PositionTracking } from '@/components/trading/PositionTracking'
import { OrderManagement } from '@/components/trading/OrderManagement'

export default function DashboardPage() {
  const marketData = useAllMarketData()
  const portfolioData = usePortfolioData()
  const tradeSignals = useTradeSignals()
  const { isConnected, isConnecting, lastError, reconnect } = useRealtimeStatus()
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSD')
  const [lastUpdate, setLastUpdate] = useState(new Date())

  // Update timestamp whenever we get new data
  useEffect(() => {
    if (Object.keys(marketData).length > 0) {
      setLastUpdate(new Date())
    }
  }, [marketData])

  // Convert market data to sorted array
  const marketDataArray = Object.values(marketData).sort((a, b) => {
    // Sort by volume, highest first
    return b.volume - a.volume
  })

  // Calculate total portfolio value
  const totalPortfolioValue = portfolioData?.totalBalance || 0
  const availableBalance = portfolioData?.availableBalance || 0
  const totalPnL = portfolioData?.totalPnL || 0
  const totalPnLPercentage = portfolioData?.totalPnLPercentage || 0
  const dayPnL = portfolioData?.dayPnL || 0
  const dayPnLPercentage = portfolioData?.dayPnLPercentage || 0

  // Find current price for selected symbol
  const symbolData = marketData[selectedSymbol]
  const currentPrice = symbolData?.price

  return (
    <div className="space-y-6 animate-in">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Trading Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor markets, manage positions, and execute trades
          </p>
        </div>
        <div className="flex items-center gap-2 mt-4 md:mt-0">
          <Badge variant={isConnected ? "default" : "destructive"} className="h-6">
            {isConnected ? "🟢 Connected" : isConnecting ? "🟡 Connecting..." : "🔴 Disconnected"}
          </Badge>
          {!isConnected && (
            <Button size="sm" onClick={reconnect}>
              Reconnect
            </Button>
          )}
          <Badge variant="outline" className="h-6">
            Last update: {lastUpdate.toLocaleTimeString()}
          </Badge>
        </div>
      </div>

      {lastError && (
        <div className="bg-destructive/10 text-destructive p-3 rounded-md border border-destructive/20">
          <p>Connection error: {lastError}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Portfolio summary */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Portfolio Summary</CardTitle>
              <CardDescription>Your account balance and performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="text-sm font-medium">Total Balance</div>
                  <div className="text-3xl font-bold">${totalPortfolioValue.toLocaleString()}</div>
                  <div className={`text-sm ${totalPnLPercentage >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {totalPnLPercentage >= 0 ? '↑' : '↓'} ${Math.abs(totalPnL).toLocaleString()} ({Math.abs(totalPnLPercentage).toFixed(2)}%)
                  </div>
                </div>

                <div>
                  <div className="text-sm font-medium">Available Balance</div>
                  <div className="text-2xl font-bold">${availableBalance.toLocaleString()}</div>
                </div>

                <div>
                  <div className="text-sm font-medium">Today's P&L</div>
                  <div className={`text-2xl font-bold ${dayPnL >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {dayPnL >= 0 ? '+' : '-'}${Math.abs(dayPnL).toLocaleString()} ({Math.abs(dayPnLPercentage).toFixed(2)}%)
                  </div>
                </div>

                <div className="pt-2">
                  <div className="flex justify-between text-sm">
                    <span>Used Margin</span>
                    <span>{portfolioData ? ((totalPortfolioValue - availableBalance) / totalPortfolioValue * 100).toFixed(1) : 0}%</span>
                  </div>
                  <Progress value={portfolioData ? ((totalPortfolioValue - availableBalance) / totalPortfolioValue * 100) : 0} className="h-2 mt-1" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Market Watch</CardTitle>
              <CardDescription>Real-time market prices</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
                {marketDataArray.length > 0 ? (
                  marketDataArray.map((data) => (
                    <div 
                      key={data.symbol} 
                      className={`flex justify-between items-center p-2 rounded-md cursor-pointer transition-colors ${selectedSymbol === data.symbol ? 'bg-muted' : 'hover:bg-muted/50'}`}
                      onClick={() => setSelectedSymbol(data.symbol)}
                    >
                      <div className="font-medium">{data.symbol}</div>
                      <div className="flex flex-col items-end">
                        <div className="font-semibold">${data.price.toLocaleString()}</div>
                        <div className={`text-xs ${data.change24h >= 0 ? 'text-profit' : 'text-loss'}`}>
                          {data.change24h >= 0 ? '↑' : '↓'} {Math.abs(data.change24h).toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-muted-foreground py-4">
                    {isConnected ? 'Loading market data...' : 'Connect to see market data'}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Trading Signals</CardTitle>
              <CardDescription>AI-generated trade suggestions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2">
                {tradeSignals.length > 0 ? (
                  tradeSignals.map((signal) => (
                    <div key={signal.id} className="border rounded-md p-3 space-y-2">
                      <div className="flex justify-between items-center">
                        <div className="font-medium">{signal.symbol}</div>
                        <Badge variant={signal.type === 'buy' ? 'default' : 'destructive'}>
                          {signal.type.toUpperCase()}
                        </Badge>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Price: ${signal.price.toLocaleString()}</span>
                        <span>Confidence: {(signal.confidence * 100).toFixed(0)}%</span>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(signal.timestamp).toLocaleString()}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-muted-foreground py-4">
                    {isConnected ? 'No trading signals yet' : 'Connect to see trading signals'}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Center and right columns - Chart and trading interface */}
        <div className="lg:col-span-2 space-y-6">
          <Card className="p-0 overflow-hidden">
            <CardHeader className="p-4">
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle className="flex items-center">
                    {selectedSymbol}
                    {currentPrice && (
                      <span className="ml-3 text-2xl">
                        ${currentPrice.toLocaleString()}
                      </span>
                    )}
                  </CardTitle>
                  {symbolData && (
                    <CardDescription className={symbolData.change24h >= 0 ? 'text-profit' : 'text-loss'}>
                      {symbolData.change24h >= 0 ? '▲' : '▼'} {Math.abs(symbolData.change24h).toFixed(2)}% (24h)
                    </CardDescription>
                  )}
                </div>
                <div>
                  <Button size="sm" variant="outline" className="mr-2">Add Indicator</Button>
                  <Button size="sm">Trade</Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="h-[400px]">
                <TradingChart symbol={selectedSymbol} height={400} />
              </div>
            </CardContent>
          </Card>

          <Tabs defaultValue="positions" className="w-full">
            <TabsList className="grid grid-cols-2">
              <TabsTrigger value="positions">Positions</TabsTrigger>
              <TabsTrigger value="orders">Orders</TabsTrigger>
            </TabsList>
            <TabsContent value="positions" className="mt-2">
              <PositionTracking />
            </TabsContent>
            <TabsContent value="orders" className="mt-2">
              <OrderManagement />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  )
}


