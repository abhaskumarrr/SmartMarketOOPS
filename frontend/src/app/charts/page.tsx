'use client';

import { useState } from 'react'
import { TradingChart } from '@/components/charts/TradingChart'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'

export default function ChartsPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSD')

  const symbols = [
    { value: 'BTCUSD', label: 'Bitcoin', price: 50000, change: 2.34 },
    { value: 'ETHUSD', label: 'Ethereum', price: 3100, change: 1.87 },
    { value: 'SOLUSD', label: 'Solana', price: 98.50, change: -0.45 },
    { value: 'ADAUSD', label: 'Cardano', price: 0.45, change: 3.21 },
    { value: 'DOTUSD', label: 'Polkadot', price: 6.78, change: -1.23 },
    { value: 'LINKUSD', label: 'Chainlink', price: 14.56, change: 4.67 },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Live Charts</h1>
          <p className="text-muted-foreground">
            Real-time cryptocurrency price charts with professional trading tools
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {symbols.map((symbol) => (
                <SelectItem key={symbol.value} value={symbol.value}>
                  {symbol.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Main Trading Chart */}
      <TradingChart symbol={selectedSymbol} height={600} />

      {/* Market Overview Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {symbols.map((symbol) => (
          <Card
            key={symbol.value}
            className={`cursor-pointer transition-all hover:shadow-md ${
              selectedSymbol === symbol.value ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setSelectedSymbol(symbol.value)}
          >
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{symbol.label}</CardTitle>
                <Badge variant={symbol.change >= 0 ? 'default' : 'destructive'}>
                  {symbol.change >= 0 ? '+' : ''}{symbol.change.toFixed(2)}%
                </Badge>
              </div>
              <CardDescription>{symbol.value}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">${symbol.price.toLocaleString()}</div>
              <div className="text-sm text-muted-foreground mt-1">
                24h Volume: ${(Math.random() * 1000000000).toLocaleString()}
              </div>
              <div className="mt-4 flex space-x-2">
                <Button
                  variant={selectedSymbol === symbol.value ? 'default' : 'outline'}
                  size="sm"
                  className="flex-1"
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedSymbol(symbol.value)
                  }}
                >
                  View Chart
                </Button>
                <Button variant="outline" size="sm">
                  Trade
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
