"use client"

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { usePortfolioData, useRealtimeStatus } from '@/lib/realtime-data'

export function PositionTracking() {
  const portfolioData = usePortfolioData()
  const { isConnected, usingMockData } = useRealtimeStatus()
  const [expandedPosition, setExpandedPosition] = useState<string | null>(null)

  const positions = portfolioData?.positions || []
  
  const toggleExpand = (symbol: string) => {
    setExpandedPosition(expandedPosition === symbol ? null : symbol)
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex justify-between items-center">
          <span>Open Positions</span>
          {usingMockData && (
            <Badge variant="outline" className="text-xs">Simulated Data</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {positions.length > 0 ? (
          <div className="space-y-2">
            {positions.map((position) => (
              <div 
                key={position.symbol} 
                className="rounded-md border border-border p-3 transition-all hover:bg-muted/30 cursor-pointer"
                onClick={() => toggleExpand(position.symbol)}
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{position.symbol}</span>
                    <Badge variant={position.pnl >= 0 ? "default" : "destructive"}>
                      {position.size > 0 ? 'LONG' : 'SHORT'}
                    </Badge>
                  </div>
                  <div className={`text-lg font-semibold ${position.pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                    {position.pnl >= 0 ? '+' : ''}{position.pnl.toLocaleString()} USD
                  </div>
                </div>
                
                <div className="flex justify-between text-sm mt-1">
                  <span>Size: {Math.abs(position.size)} ({Math.abs(position.size * position.currentPrice).toLocaleString()} USD)</span>
                  <span className={position.pnl >= 0 ? 'text-profit' : 'text-loss'}>
                    {position.pnlPercentage >= 0 ? '+' : ''}{position.pnlPercentage.toFixed(2)}%
                  </span>
                </div>
                
                {expandedPosition === position.symbol && (
                  <div className="mt-3 pt-3 border-t border-border space-y-2">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <div className="text-muted-foreground">Entry Price</div>
                        <div>${position.entryPrice.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Current Price</div>
                        <div>${position.currentPrice.toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Liquidation Price</div>
                        <div>${(position.entryPrice * (position.size > 0 ? 0.8 : 1.2)).toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Margin</div>
                        <div>${(Math.abs(position.size * position.entryPrice) * 0.1).toLocaleString()}</div>
                      </div>
                    </div>
                    
                    <div className="flex gap-2 mt-4">
                      <Button size="sm" variant="destructive" className="w-full">Close Position</Button>
                      <Button size="sm" variant="outline" className="w-full">Edit TP/SL</Button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="py-8 text-center text-muted-foreground">
            {isConnected ? (
              <>
                <p>You don't have any open positions.</p>
                <Button className="mt-4" size="sm">Open New Position</Button>
              </>
            ) : (
              <p>Connect to view your positions.</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
