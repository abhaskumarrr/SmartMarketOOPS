"use client"

import React, { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { TrendingUp, TrendingDown, DollarSign, Activity, Target, AlertTriangle, RefreshCw } from 'lucide-react'
import { apiClient, isApiSuccess, Portfolio } from '@/lib/api'
import { usePortfolioWebSocket } from '@/hooks/useWebSocket'

// All interfaces are now imported from @/lib/api

export function PortfolioDashboard() {
  const [portfolioData, setPortfolioData] = useState<Portfolio | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null)

  // WebSocket for real-time portfolio updates
  const { portfolio: wsPortfolio, isConnected } = usePortfolioWebSocket()

  // Manual refresh function
  const refreshPortfolio = async () => {
    try {
      setIsLoading(true)
      setError(null)

      const response = await apiClient.getPortfolio()

      if (isApiSuccess(response)) {
        setPortfolioData(response.data)
        setLastUpdate(new Date())
      } else {
        throw new Error(response.error || 'Failed to refresh portfolio data')
      }
    } catch (err) {
      console.error('Error refreshing portfolio:', err)
      setError(err instanceof Error ? err.message : 'Failed to refresh portfolio')
    } finally {
      setIsLoading(false)
    }
  }

  // Fetch portfolio data
  useEffect(() => {
    const fetchPortfolioData = async () => {
      try {
        setIsLoading(true)
        setError(null)

        const response = await apiClient.getPortfolio()

        if (isApiSuccess(response)) {
          setPortfolioData(response.data)
          setLastUpdate(new Date())
        } else {
          throw new Error(response.error || 'Failed to fetch portfolio data')
        }
      } catch (err) {
        console.error('Error fetching portfolio data:', err)
        setError(err instanceof Error ? err.message : 'Failed to load portfolio data')
        // Generate mock data for demo
        generateMockPortfolioData()
      } finally {
        setIsLoading(false)
      }
    }

    fetchPortfolioData()

    // Set up real-time updates if WebSocket is not connected
    let interval: NodeJS.Timeout | null = null
    if (!isConnected) {
      interval = setInterval(fetchPortfolioData, 15000) // Update every 15 seconds
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isConnected])

  // Update from WebSocket data
  useEffect(() => {
    if (wsPortfolio) {
      setPortfolioData(wsPortfolio)
      setLastUpdate(new Date())
      setError(null)
    }
  }, [wsPortfolio])

  const generateMockPortfolioData = () => {
    const mockData: Portfolio = {
      totalBalance: 10000,
      availableBalance: 7500,
      totalPnL: 1250,
      totalPnLPercentage: 14.3,
      dayPnL: 320,
      dayPnLPercentage: 3.2,
      positions: [
        {
          id: '1',
          symbol: 'BTCUSD',
          side: 'long',
          size: 0.5,
          entryPrice: 48000,
          currentPrice: 50000,
          pnl: 1000,
          pnlPercentage: 4.17,
          leverage: 10,
          margin: 2400,
          liquidationPrice: 43200,
          timestamp: new Date(Date.now() - 3600000).toISOString(),
        },
        {
          id: '2',
          symbol: 'ETHUSD',
          side: 'short',
          size: 2,
          entryPrice: 3200,
          currentPrice: 3100,
          pnl: 200,
          pnlPercentage: 3.13,
          leverage: 5,
          margin: 1280,
          liquidationPrice: 3520,
          timestamp: new Date(Date.now() - 1800000).toISOString(),
        },
      ],
      recentTrades: [
        {
          id: '1',
          symbol: 'BTCUSD',
          side: 'buy',
          size: 0.1,
          price: 49800,
          timestamp: new Date(Date.now() - 300000).toISOString(),
          status: 'filled',
        },
        {
          id: '2',
          symbol: 'ETHUSD',
          side: 'sell',
          size: 0.5,
          price: 3150,
          timestamp: new Date(Date.now() - 600000).toISOString(),
          status: 'filled',
        },
      ],
    }
    setPortfolioData(mockData)
  }

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <Card key={i} className="animate-pulse">
            <CardHeader className="space-y-0 pb-2">
              <div className="h-4 bg-muted rounded w-3/4"></div>
              <div className="h-8 bg-muted rounded w-1/2 mt-2"></div>
            </CardHeader>
          </Card>
        ))}
      </div>
    )
  }

  if (error || !portfolioData) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-32">
          <div className="text-center">
            <AlertTriangle className="h-8 w-8 text-destructive mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">{error || 'No portfolio data available'}</p>
            <div className="flex gap-2 mt-2">
              <Button variant="outline" size="sm" onClick={refreshPortfolio} disabled={isLoading}>
                <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                {isLoading ? 'Refreshing...' : 'Refresh'}
              </Button>
              <Button variant="outline" size="sm" onClick={() => window.location.reload()}>
                Reload Page
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Balance</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${portfolioData.totalBalance.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              Available: ${portfolioData.availableBalance.toLocaleString()}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
            {portfolioData.totalPnL >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-500" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${portfolioData.totalPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              ${portfolioData.totalPnL.toLocaleString()}
            </div>
            <p className={`text-xs ${portfolioData.totalPnLPercentage >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {portfolioData.totalPnLPercentage >= 0 ? '+' : ''}{portfolioData.totalPnLPercentage.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Day P&L</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${portfolioData.dayPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              ${portfolioData.dayPnL.toLocaleString()}
            </div>
            <p className={`text-xs ${portfolioData.dayPnLPercentage >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {portfolioData.dayPnLPercentage >= 0 ? '+' : ''}{portfolioData.dayPnLPercentage.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Positions</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{portfolioData.positions.length}</div>
            <p className="text-xs text-muted-foreground">
              {portfolioData.positions.filter(p => p.pnl >= 0).length} profitable
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Active Positions */}
      <Card>
        <CardHeader>
          <CardTitle>Active Positions</CardTitle>
          <CardDescription>Your current trading positions</CardDescription>
        </CardHeader>
        <CardContent>
          {portfolioData.positions.length === 0 ? (
            <div className="text-center py-8">
              <Target className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No active positions</p>
            </div>
          ) : (
            <div className="space-y-4">
              {portfolioData.positions.map((position) => (
                <div key={position.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div>
                      <div className="font-medium">{position.symbol}</div>
                      <div className="flex items-center space-x-2">
                        <Badge variant={position.side === 'long' ? 'default' : 'secondary'}>
                          {position.side.toUpperCase()}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {position.leverage}x leverage
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">Size: {position.size}</div>
                    <div className="text-sm text-muted-foreground">
                      Entry: ${position.entryPrice.toLocaleString()}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">
                      Current: ${position.currentPrice.toLocaleString()}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Margin: ${position.margin.toLocaleString()}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold ${position.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      ${position.pnl.toLocaleString()}
                    </div>
                    <div className={`text-sm ${position.pnlPercentage >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                      {position.pnlPercentage >= 0 ? '+' : ''}{position.pnlPercentage.toFixed(2)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Trades */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Trades</CardTitle>
          <CardDescription>Your latest trading activity</CardDescription>
        </CardHeader>
        <CardContent>
          {portfolioData.recentTrades.length === 0 ? (
            <div className="text-center py-8">
              <Activity className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No recent trades</p>
            </div>
          ) : (
            <div className="space-y-2">
              {portfolioData.recentTrades.map((trade) => (
                <div key={trade.id} className="flex items-center justify-between p-3 border rounded">
                  <div className="flex items-center space-x-3">
                    <Badge variant={trade.side === 'buy' ? 'default' : 'destructive'}>
                      {trade.side.toUpperCase()}
                    </Badge>
                    <span className="font-medium">{trade.symbol}</span>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">
                      {trade.size} @ ${trade.price.toLocaleString()}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {new Date(trade.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  <Badge variant={trade.status === 'filled' ? 'default' : 'secondary'}>
                    {trade.status}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
