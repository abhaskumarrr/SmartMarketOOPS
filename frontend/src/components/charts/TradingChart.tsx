"use client"

import React, { useEffect, useRef, useState } from 'react'
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, CandlestickSeries } from 'lightweight-charts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Badge } from '@/components/ui/badge'
import { apiClient, generateMockCandleData, isApiSuccess } from '@/lib/api'
import { useMarketDataWebSocket } from '@/hooks/useWebSocket'

interface TradingChartProps {
  symbol?: string
  height?: number
}

interface CandlestickDataPoint {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export function TradingChart({ symbol = 'BTCUSD', height = 400 }: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h')
  const [isLoading, setIsLoading] = useState(true)
  const [currentPrice, setCurrentPrice] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  // WebSocket for real-time updates
  const { marketData, isConnected, error: wsError } = useMarketDataWebSocket()

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      layout: {
        textColor: 'rgba(255, 255, 255, 0.9)',
        background: { type: 'solid', color: 'transparent' },
      },
      grid: {
        vertLines: { color: 'rgba(197, 203, 206, 0.1)' },
        horzLines: { color: 'rgba(197, 203, 206, 0.1)' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: 'rgba(197, 203, 206, 0.2)',
      },
      timeScale: {
        borderColor: 'rgba(197, 203, 206, 0.2)',
        timeVisible: true,
        secondsVisible: false,
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
    })

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    })

    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
      }
    }
  }, [height])

  // Fetch and update chart data
  useEffect(() => {
    const fetchChartData = async () => {
      setIsLoading(true)
      setError(null)

      try {
        // Try to fetch real data from API
        const response = await apiClient.getCandleData(symbol, selectedTimeframe, 100)

        if (isApiSuccess(response) && response.data.candles) {
          // Transform data to lightweight-charts format
          const chartData: CandlestickDataPoint[] = response.data.candles.map((candle) => ({
            time: candle.timestamp as Time,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.volume,
          }))

          if (candlestickSeriesRef.current) {
            candlestickSeriesRef.current.setData(chartData)

            if (chartData.length > 0) {
              setCurrentPrice(chartData[chartData.length - 1].close)
            }
          }
        } else {
          throw new Error(response.error || 'Failed to fetch chart data')
        }
      } catch (error) {
        console.error('Error fetching chart data:', error)
        setError(error instanceof Error ? error.message : 'Failed to load chart data')

        // Generate mock data as fallback
        generateMockData()
      } finally {
        setIsLoading(false)
      }
    }

    fetchChartData()
  }, [symbol, selectedTimeframe])

  // Generate mock data for demo purposes
  const generateMockData = () => {
    const data: CandlestickDataPoint[] = []
    const basePrice = 50000
    let currentPrice = basePrice
    const now = new Date()

    for (let i = 100; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000) // 1 hour intervals
      const open = currentPrice
      const change = (Math.random() - 0.5) * 1000
      const close = open + change
      const high = Math.max(open, close) + Math.random() * 500
      const low = Math.min(open, close) - Math.random() * 500

      data.push({
        time: Math.floor(time.getTime() / 1000) as Time,
        open,
        high,
        low,
        close,
        volume: Math.random() * 1000000,
      })

      currentPrice = close
    }

    if (candlestickSeriesRef.current) {
      candlestickSeriesRef.current.setData(data)
      setCurrentPrice(data[data.length - 1].close)
    }
  }

  // Real-time updates from WebSocket
  useEffect(() => {
    if (marketData && Array.isArray(marketData)) {
      const symbolData = marketData.find(data => data.symbol === symbol)
      if (symbolData && candlestickSeriesRef.current) {
        // Update current price from real-time data
        setCurrentPrice(symbolData.price)

        // Update the latest candle with current price
        const newDataPoint: CandlestickDataPoint = {
          time: Math.floor(Date.now() / 1000) as Time,
          open: symbolData.price * 0.999, // Approximate open
          high: symbolData.high24h || symbolData.price * 1.001,
          low: symbolData.low24h || symbolData.price * 0.999,
          close: symbolData.price,
        }

        candlestickSeriesRef.current.update(newDataPoint)
      }
    }
  }, [marketData, symbol])

  // Fallback real-time updates if WebSocket is not connected
  useEffect(() => {
    if (isConnected) return // Skip if WebSocket is working

    const interval = setInterval(async () => {
      try {
        const response = await apiClient.getSymbolMarketData(symbol)
        if (isApiSuccess(response) && candlestickSeriesRef.current) {
          const data = response.data
          setCurrentPrice(data.price)

          const newDataPoint: CandlestickDataPoint = {
            time: Math.floor(Date.now() / 1000) as Time,
            open: data.price * 0.999,
            high: data.high24h || data.price * 1.001,
            low: data.low24h || data.price * 0.999,
            close: data.price,
          }

          candlestickSeriesRef.current.update(newDataPoint)
        }
      } catch (error) {
        console.error('Error fetching real-time data:', error)
      }
    }, 10000) // Update every 10 seconds as fallback

    return () => clearInterval(interval)
  }, [symbol, isConnected])

  const timeframes = [
    { value: '1m', label: '1m' },
    { value: '5m', label: '5m' },
    { value: '15m', label: '15m' },
    { value: '1h', label: '1h' },
    { value: '4h', label: '4h' },
    { value: '1d', label: '1d' },
  ]

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="flex items-center space-x-4">
          <CardTitle className="text-2xl font-bold">{symbol}</CardTitle>
          {currentPrice && (
            <div className="text-2xl font-bold text-green-500">
              ${currentPrice.toLocaleString()}
            </div>
          )}
          {/* Connection Status */}
          <div className="flex items-center space-x-2">
            <Badge variant={isConnected ? 'default' : 'secondary'}>
              {isConnected ? '🟢 Live' : '🔴 Offline'}
            </Badge>
            {error && (
              <Badge variant="destructive">
                Error
              </Badge>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-20">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timeframes.map((tf) => (
                <SelectItem key={tf.value} value={tf.value}>
                  {tf.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="sm"
            onClick={() => chartRef.current?.timeScale().fitContent()}
          >
            Fit
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/50 z-10">
              <div className="text-sm text-muted-foreground">Loading chart data...</div>
            </div>
          )}
          <div ref={chartContainerRef} className="w-full" />
        </div>
      </CardContent>
    </Card>
  )
}
