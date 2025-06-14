'use client';

import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { PauseCircle, PlayCircle, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import useOptimizedWebSocket from '@/hooks/useOptimizedWebSocket';
import { useBreakpoints } from '@/hooks/use-responsive';

interface RealTimeDataChartProps {
  className?: string;
  symbol: string;
  title?: string;
  height?: number | string;
  dataKeys?: string[];
  interval?: string;
  compact?: boolean;
  showControls?: boolean;
}

// WebSocket data structure
interface TickerData {
  symbol: string;
  price: number;
  volume: number;
  high: number;
  low: number;
  change: number;
  changePercent: number;
  timestamp: number;
  [key: string]: number | string; // More specific than any
}

// Time intervals
const INTERVALS = [
  { value: '1m', label: '1m' },
  { value: '5m', label: '5m' },
  { value: '15m', label: '15m' },
  { value: '30m', label: '30m' },
  { value: '1h', label: '1h' },
  { value: '4h', label: '4h' },
  { value: '1d', label: '1d' },
];

// Simulates a WebSocket URL for the given symbol and interval
const getWebSocketUrl = (symbol: string, interval: string) => {
  // In a real app, this would be your actual WebSocket endpoint
  return `${process.env.NEXT_PUBLIC_WS_BASE_URL || 'wss://api.example.com'}/market/ticker/${symbol}/${interval}`;
};

const RealTimeDataChart: React.FC<RealTimeDataChartProps> = ({
  className,
  symbol,
  title = 'Real-time Price',
  height = 300,
  dataKeys = ['price'],
  interval: initialInterval = '1m',
  compact = false,
  showControls = true,
}) => {
  const { isMobile, isTablet } = useBreakpoints();
  const [interval, setInterval] = useState(initialInterval);
  const [isPaused, setIsPaused] = useState(false);
  const [chartData, setChartData] = useState<TickerData[]>([]);
  const [visibleKeys, setVisibleKeys] = useState<string[]>(dataKeys);
  const maxDataPoints = useRef(isMobile ? 20 : isTablet ? 50 : 100);
  
  // Dynamic WebSocket URL based on symbol and interval
  const wsUrl = !isPaused ? getWebSocketUrl(symbol, interval) : null;
  
  // Use our optimized WebSocket hook with performance enhancements
  const { 
    isConnected, 
    bufferedData, 
    data: latestData, 
    error,
    reconnect 
  } = useOptimizedWebSocket<TickerData>(wsUrl, {
    bufferInterval: 250,  // Process data every 250ms
    batchSize: 5,         // Process up to 5 messages at once
    enableBuffering: true,
    autoReconnect: true,
    debug: process.env.NODE_ENV === 'development'
  });
  
  // Update data when new data arrives via WebSocket
  useEffect(() => {
    if (!latestData || isPaused) return;
    
    setChartData(prevData => {
      // Add new data point
      const newData = [...prevData, latestData];
      
      // Limit the number of data points to prevent memory issues
      if (newData.length > maxDataPoints.current) {
        return newData.slice(-maxDataPoints.current);
      }
      return newData;
    });
  }, [latestData, isPaused]);
  
  // Process buffered data for smoother updates
  useEffect(() => {
    if (!bufferedData || bufferedData.length === 0 || isPaused) return;
    
    // Only update if we have multiple data points to process
    if (bufferedData.length > 1) {
      setChartData(prevData => {
        const newData = [...prevData];
        
        // Add only unique timestamps to prevent duplicates
        bufferedData.forEach(dataPoint => {
          if (!newData.some(d => d.timestamp === dataPoint.timestamp)) {
            newData.push(dataPoint);
          }
        });
        
        // Sort by timestamp and limit the data points
        return newData
          .sort((a, b) => a.timestamp - b.timestamp)
          .slice(-maxDataPoints.current);
      });
    }
  }, [bufferedData, isPaused]);
  
  // Clear data when symbol or interval changes
  useEffect(() => {
    setChartData([]);
  }, [symbol, interval]);
  
  // Update maxDataPoints when screen size changes
  useEffect(() => {
    maxDataPoints.current = isMobile ? 20 : isTablet ? 50 : 100;
  }, [isMobile, isTablet]);
  
  // Format timestamp for display
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };
  
  // Get the latest price and calculate change
  const latestPrice = useMemo(() => {
    if (chartData.length === 0) return null;
    return chartData[chartData.length - 1].price;
  }, [chartData]);
  
  const priceChange = useMemo(() => {
    if (chartData.length < 2) return null;
    const latest = chartData[chartData.length - 1];
    const first = chartData[0];
    return {
      absolute: latest.price - first.price,
      percent: ((latest.price - first.price) / first.price) * 100
    };
  }, [chartData]);
  
  // Generate colors for each data key
  const lineColors = useMemo(() => {
    const colors = [
      '#3498db', // Blue
      '#2ecc71', // Green
      '#e74c3c', // Red
      '#f39c12', // Orange
      '#9b59b6', // Purple
      '#1abc9c', // Teal
      '#34495e', // Dark Blue
    ];
    
    return dataKeys.reduce((acc, key, index) => {
      acc[key] = colors[index % colors.length];
      return acc;
    }, {} as Record<string, string>);
  }, [dataKeys]);
  
  // Toggle visibility of a data key in the chart
  const toggleDataKey = (key: string) => {
    setVisibleKeys(prev => 
      prev.includes(key)
        ? prev.filter(k => k !== key)
        : [...prev, key]
    );
  };
  
  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className={cn(
        "flex flex-row items-center justify-between",
        compact ? "px-3 py-2" : "px-4 py-3"
      )}>
        <div className="flex flex-col">
          <CardTitle className={cn(
            "flex items-center",
            compact ? "text-sm" : "text-base"
          )}>
            {title}
            <Badge 
              variant={isConnected ? "outline" : "destructive"} 
              className="ml-2 text-xs"
            >
              {isConnected ? "Live" : "Disconnected"}
            </Badge>
          </CardTitle>
          {latestPrice && (
            <div className="flex items-center mt-1">
              <span className={cn(
                "font-mono font-medium",
                compact ? "text-base" : "text-lg"
              )}>
                {latestPrice.toFixed(2)}
              </span>
              {priceChange && (
                <span className={cn(
                  "ml-2 text-xs font-medium",
                  priceChange.absolute >= 0 ? "text-green-500" : "text-red-500"
                )}>
                  {priceChange.absolute >= 0 ? "+" : ""}
                  {priceChange.absolute.toFixed(2)} ({priceChange.percent.toFixed(2)}%)
                </span>
              )}
            </div>
          )}
        </div>
        
        {showControls && (
          <div className="flex items-center gap-1">
            {!compact && (
              <Select value={interval} onValueChange={setInterval}>
                <SelectTrigger className="w-16 h-8">
                  <SelectValue placeholder="1m" />
                </SelectTrigger>
                <SelectContent>
                  {INTERVALS.map((int) => (
                    <SelectItem key={int.value} value={int.value}>
                      {int.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            
            <Button
              variant="ghost"
              size="icon"
              className={cn("h-8 w-8", compact && "p-1")}
              onClick={() => setIsPaused(!isPaused)}
            >
              {isPaused ? (
                <PlayCircle className="h-4 w-4" />
              ) : (
                <PauseCircle className="h-4 w-4" />
              )}
            </Button>
            
            {!isConnected && (
              <Button
                variant="ghost"
                size="icon"
                className={cn("h-8 w-8", compact && "p-1")}
                onClick={reconnect}
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            )}
          </div>
        )}
      </CardHeader>
      
      <CardContent className={cn("p-0", compact ? "h-[160px]" : `h-[${height}px]`)}>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={formatTime} 
                tick={{ fontSize: compact ? 10 : 12 }}
              />
              <YAxis 
                domain={['auto', 'auto']}
                tick={{ fontSize: compact ? 10 : 12 }}
              />
              <Tooltip
                labelFormatter={(value) => formatTime(value as number)}
                contentStyle={{ 
                  backgroundColor: 'rgba(0, 0, 0, 0.8)', 
                  border: 'none',
                  borderRadius: '4px', 
                  fontSize: compact ? '10px' : '12px' 
                }}
              />
              {!compact && <Legend onClick={(e) => {
                if (typeof e.dataKey === 'string') {
                  toggleDataKey(e.dataKey);
                }
              }} />}
              
              {/* Render each visible data key as a line */}
              {dataKeys.filter(key => visibleKeys.includes(key)).map((key) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  name={key.charAt(0).toUpperCase() + key.slice(1)}
                  stroke={lineColors[key]}
                  dot={false}
                  activeDot={{ r: 4 }}
                  isAnimationActive={false} // Disable animation for performance
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-muted-foreground text-sm">
              {error ? "Error loading data" : "Waiting for data..."}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RealTimeDataChart; 