"use client"

import React, { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { TrendingUp, TrendingDown, Target, AlertTriangle, DollarSign, Activity } from 'lucide-react'

interface Position {
  id: string
  symbol: string
  side: 'long' | 'short'
  size: number
  entryPrice: number
  currentPrice: number
  pnl: number
  pnlPercentage: number
  leverage: number
  margin: number
  liquidationPrice: number
  stopLoss?: number
  takeProfit?: number
  timestamp: string
}

interface PositionMetrics {
  totalPositions: number
  profitablePositions: number
  totalPnL: number
  totalMargin: number
  riskLevel: 'low' | 'medium' | 'high'
}

export function PositionTracking() {
  const [positions, setPositions] = useState<Position[]>([])
  const [metrics, setMetrics] = useState<PositionMetrics | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const fetchPositions = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('/api/positions')
        
        if (!response.ok) {
          throw new Error('Failed to fetch positions')
        }
        
        const data = await response.json()
        setPositions(data.positions || [])
        setMetrics(data.metrics || null)
      } catch (error) {
        console.error('Error fetching positions:', error)
        // Generate mock data for demo
        generateMockPositions()
      } finally {
        setIsLoading(false)
      }
    }

    fetchPositions()
    
    // Set up real-time updates
    const interval = setInterval(fetchPositions, 5000) // Update every 5 seconds
    
    return () => clearInterval(interval)
  }, [])

  const generateMockPositions = () => {
    const mockPositions: Position[] = [
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
        stopLoss: 46000,
        takeProfit: 55000,
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
        takeProfit: 2900,
        timestamp: new Date(Date.now() - 1800000).toISOString(),
      },
      {
        id: '3',
        symbol: 'SOLUSD',
        side: 'long',
        size: 10,
        entryPrice: 95,
        currentPrice: 92,
        pnl: -30,
        pnlPercentage: -3.16,
        leverage: 20,
        margin: 47.5,
        liquidationPrice: 90.25,
        stopLoss: 90,
        takeProfit: 105,
        timestamp: new Date(Date.now() - 900000).toISOString(),
      },
    ]

    const mockMetrics: PositionMetrics = {
      totalPositions: mockPositions.length,
      profitablePositions: mockPositions.filter(p => p.pnl > 0).length,
      totalPnL: mockPositions.reduce((sum, p) => sum + p.pnl, 0),
      totalMargin: mockPositions.reduce((sum, p) => sum + p.margin, 0),
      riskLevel: 'medium',
    }

    setPositions(mockPositions)
    setMetrics(mockMetrics)
  }

  const handleClosePosition = async (positionId: string) => {
    try {
      const response = await fetch(`/api/positions/${positionId}/close`, {
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error('Failed to close position')
      }

      setPositions(prev => prev.filter(p => p.id !== positionId))
      alert('Position closed successfully!')
    } catch (error) {
      console.error('Error closing position:', error)
      alert('Failed to close position. Please try again.')
    }
  }

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'text-green-500'
      case 'medium': return 'text-yellow-500'
      case 'high': return 'text-red-500'
      default: return 'text-muted-foreground'
    }
  }

  const getLiquidationRisk = (position: Position) => {
    const priceDistance = Math.abs(position.currentPrice - position.liquidationPrice)
    const riskPercentage = (priceDistance / position.currentPrice) * 100
    
    if (riskPercentage < 5) return { level: 'high', color: 'bg-red-500' }
    if (riskPercentage < 15) return { level: 'medium', color: 'bg-yellow-500' }
    return { level: 'low', color: 'bg-green-500' }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="space-y-0 pb-2">
                <div className="h-4 bg-muted rounded w-3/4"></div>
                <div className="h-8 bg-muted rounded w-1/2 mt-2"></div>
              </CardHeader>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Position Metrics */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Positions</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.totalPositions}</div>
              <p className="text-xs text-muted-foreground">
                {metrics.profitablePositions} profitable
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
              {metrics.totalPnL >= 0 ? (
                <TrendingUp className="h-4 w-4 text-green-500" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-500" />
              )}
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${metrics.totalPnL >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                ${metrics.totalPnL.toLocaleString()}
              </div>
              <p className="text-xs text-muted-foreground">
                Unrealized P&L
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Margin</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">${metrics.totalMargin.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground">
                Used margin
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Risk Level</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold capitalize ${getRiskColor(metrics.riskLevel)}`}>
                {metrics.riskLevel}
              </div>
              <p className="text-xs text-muted-foreground">
                Portfolio risk
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Active Positions */}
      <Card>
        <CardHeader>
          <CardTitle>Active Positions</CardTitle>
          <CardDescription>Monitor your open trading positions in real-time</CardDescription>
        </CardHeader>
        <CardContent>
          {positions.length === 0 ? (
            <div className="text-center py-12">
              <Target className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No Active Positions</h3>
              <p className="text-muted-foreground">
                You don't have any open positions at the moment.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {positions.map((position) => {
                const liquidationRisk = getLiquidationRisk(position)
                
                return (
                  <Card key={position.id} className="p-4">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div>
                          <div className="flex items-center space-x-2">
                            <h3 className="text-lg font-semibold">{position.symbol}</h3>
                            <Badge variant={position.side === 'long' ? 'default' : 'secondary'}>
                              {position.side.toUpperCase()}
                            </Badge>
                            <Badge variant="outline">
                              {position.leverage}x
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            Opened {new Date(position.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => handleClosePosition(position.id)}
                      >
                        Close Position
                      </Button>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div>
                        <p className="text-sm text-muted-foreground">Size</p>
                        <p className="font-medium">{position.size}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Entry Price</p>
                        <p className="font-medium">${position.entryPrice.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Current Price</p>
                        <p className="font-medium">${position.currentPrice.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Margin</p>
                        <p className="font-medium">${position.margin.toLocaleString()}</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div>
                        <p className="text-sm text-muted-foreground">P&L</p>
                        <p className={`text-lg font-bold ${position.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                          ${position.pnl.toLocaleString()} ({position.pnlPercentage >= 0 ? '+' : ''}{position.pnlPercentage.toFixed(2)}%)
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Liquidation Price</p>
                        <div className="flex items-center space-x-2">
                          <p className="font-medium">${position.liquidationPrice.toLocaleString()}</p>
                          <Badge variant={liquidationRisk.level === 'high' ? 'destructive' : liquidationRisk.level === 'medium' ? 'secondary' : 'default'}>
                            {liquidationRisk.level} risk
                          </Badge>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Stop Loss / Take Profit</p>
                        <p className="font-medium">
                          {position.stopLoss ? `$${position.stopLoss.toLocaleString()}` : 'None'} / {position.takeProfit ? `$${position.takeProfit.toLocaleString()}` : 'None'}
                        </p>
                      </div>
                    </div>

                    {/* Liquidation Risk Bar */}
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Liquidation Risk</span>
                        <span className="capitalize">{liquidationRisk.level}</span>
                      </div>
                      <Progress 
                        value={liquidationRisk.level === 'high' ? 80 : liquidationRisk.level === 'medium' ? 50 : 20} 
                        className="h-2"
                      />
                    </div>
                  </Card>
                )
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
