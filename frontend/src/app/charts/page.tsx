'use client';

import { useState, useRef } from 'react'
import TradingViewWidget, { TradingViewWidgetRef } from '@/components/charts/TradingViewWidget'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TimeframeSelector, DEFAULT_TIMEFRAMES } from '@/components/charts/TimeframeSelector'
import { ChartType } from '@/components/charts/ChartToolbar'
import { 
  CandlestickData,
  LineData,
  SeriesDataItemTypeMap,
  CrosshairMode
} from 'lightweight-charts'
import { Separator } from '@/components/ui/separator'

// Mock data generators
function generateMockCandles(symbol: string, count = 100): CandlestickData[] {
  const candles: CandlestickData[] = [];
  let currentTime = new Date();
  currentTime.setHours(0, 0, 0, 0);
  
  // Base prices for different symbols
  const basePrices: Record<string, number> = {
    'BTCUSD': 50000,
    'ETHUSD': 3000,
    'SOLUSD': 100,
    'ADAUSD': 0.5,
    'DOTUSD': 7,
    'LINKUSD': 15,
  };
  
  const basePrice = basePrices[symbol] || 1000;
  let lastClose = basePrice;
  
  for (let i = 0; i < count; i++) {
    const time = new Date(currentTime.getTime() - (count - i) * 60 * 60 * 1000);
    const open = lastClose;
    const high = open * (1 + Math.random() * 0.03);
    const low = open * (1 - Math.random() * 0.03);
    const close = low + Math.random() * (high - low);
    
    candles.push({
      time: Math.floor(time.getTime() / 1000) as any,
      open,
      high,
      low,
      close
    });
    
    lastClose = close;
  }
  
  return candles;
}

export default function ChartsPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSD')
  const [timeframe, setTimeframe] = useState('1h')
  const [chartType, setChartType] = useState<ChartType>('candle')
  const [showVolume, setShowVolume] = useState(true)
  const [showGrid, setShowGrid] = useState(true)
  const chartRef = useRef<TradingViewWidgetRef>(null)
  
  const handleExportImage = () => {
    if (chartRef.current) {
      const chart = chartRef.current.getChart();
      if (chart) {
        const canvas = chart.takeScreenshot();
        const link = document.createElement('a');
        link.download = `${selectedSymbol}_${timeframe}_chart.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
      }
    }
  }
  
  const handleFullscreen = () => {
    const chartElement = document.getElementById('trading-chart-container');
    if (chartElement) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        chartElement.requestFullscreen();
      }
    }
  }

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
          <h1 className="text-3xl font-bold tracking-tight">Advanced Charts</h1>
          <p className="text-muted-foreground">
            Professional technical analysis with real-time data and indicators
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="w-40 h-9 rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
          >
            {symbols.map((symbol) => (
              <option key={symbol.value} value={symbol.value}>
                {symbol.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Main content area - Charts and Controls */}
        <div className="md:col-span-3">
          <Tabs defaultValue="chart" className="w-full">
            <TabsList className="mb-4">
              <TabsTrigger value="chart">Price Chart</TabsTrigger>
              <TabsTrigger value="depth">Depth Chart</TabsTrigger>
              <TabsTrigger value="compare">Compare</TabsTrigger>
            </TabsList>
            
            <TabsContent value="chart" className="space-y-4">
              <Card className="border">
                <CardContent className="p-0">
                  <div id="trading-chart-container" className="relative w-full h-[600px]">
                    {/* Chart Toolbar */}
                    <div className="absolute top-2 left-2 right-2 z-10 rounded-md bg-background/80 backdrop-blur-sm border shadow-sm flex items-center justify-between p-1">
                      <div className="flex items-center space-x-1">
                        {/* Chart Type Toggles */}
                        <div className="flex border rounded-md overflow-hidden">
                          <Button
                            variant={chartType === 'candle' ? "default" : "ghost"}
                            size="sm"
                            className="h-7 px-2 rounded-none"
                            onClick={() => setChartType('candle')}
                          >
                            Candle
                          </Button>
                          <Button
                            variant={chartType === 'line' ? "default" : "ghost"}
                            size="sm"
                            className="h-7 px-2 rounded-none"
                            onClick={() => setChartType('line')}
                          >
                            Line
                          </Button>
                          <Button
                            variant={chartType === 'bar' ? "default" : "ghost"}
                            size="sm"
                            className="h-7 px-2 rounded-none"
                            onClick={() => setChartType('bar')}
                          >
                            Bar
                          </Button>
                        </div>
                        
                        <Separator orientation="vertical" className="mx-1 h-6" />
                        
                        {/* Timeframe Selector */}
                        <TimeframeSelector
                          selectedTimeframe={timeframe}
                          onTimeframeChange={setTimeframe}
                        />
                      </div>
                      
                      <div className="flex items-center space-x-1">
                        {/* Settings */}
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-7"
                          onClick={() => setShowVolume(!showVolume)}
                        >
                          {showVolume ? 'Hide' : 'Show'} Volume
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-7"
                          onClick={() => setShowGrid(!showGrid)}
                        >
                          {showGrid ? 'Hide' : 'Show'} Grid
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-7"
                          onClick={handleExportImage}
                        >
                          Export
                        </Button>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="h-7"
                          onClick={handleFullscreen}
                        >
                          Fullscreen
                        </Button>
                      </div>
                    </div>
                    
                    {/* Trading Chart */}
                    <TradingViewWidget 
                      ref={chartRef}
                      symbol={selectedSymbol} 
                      height={600}
                      autosize
                      data={generateMockCandles(selectedSymbol, 200)}
                    />
                  </div>
                </CardContent>
              </Card>
              
              {/* Technical Analysis Summary */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-lg">Technical Analysis Summary</CardTitle>
                  <CardDescription>
                    Automated technical analysis based on popular indicators
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-1">
                      <h4 className="text-sm font-medium">Moving Averages</h4>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">MA (50)</span>
                        <Badge variant={selectedSymbol === 'BTCUSD' ? 'default' : 'secondary'}>
                          {selectedSymbol === 'BTCUSD' ? 'Buy' : 'Neutral'}
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">MA (200)</span>
                        <Badge variant="destructive">Sell</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">EMA (14)</span>
                        <Badge variant="default">Buy</Badge>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <h4 className="text-sm font-medium">Oscillators</h4>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">RSI (14)</span>
                        <Badge variant="secondary">Neutral</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">MACD</span>
                        <Badge variant="default">Buy</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Stochastic</span>
                        <Badge variant="destructive">Sell</Badge>
                      </div>
                    </div>
                    
                    <div className="space-y-1">
                      <h4 className="text-sm font-medium">Summary</h4>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">1 Hour</span>
                        <Badge variant="default">Buy</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">4 Hour</span>
                        <Badge variant="secondary">Neutral</Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">1 Day</span>
                        <Badge variant="destructive">Sell</Badge>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="depth">
              <Card className="border h-[500px] flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <h3 className="text-lg font-medium">Depth Chart Coming Soon</h3>
                  <p>Visualize buy and sell orders at different price levels</p>
                </div>
              </Card>
            </TabsContent>
            
            <TabsContent value="compare">
              <Card className="border h-[500px] flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <h3 className="text-lg font-medium">Compare Assets Coming Soon</h3>
                  <p>Compare performance of multiple cryptocurrencies</p>
                </div>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
        
        {/* Sidebar - Market Cards */}
        <div className="md:col-span-1">
          <h3 className="text-lg font-medium mb-4">Market Overview</h3>
          <div className="space-y-4">
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
      </div>
    </div>
  )
}
