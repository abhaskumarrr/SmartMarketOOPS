'use client';

import React, { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import { 
  createChart, 
  IChartApi, 
  ISeriesApi,
  CandlestickData, 
  Time,
  LineWidth,
  CrosshairMode,
  CandlestickSeries
} from 'lightweight-charts';
import { useTheme } from 'next-themes';

export interface TradingViewWidgetProps {
  symbol?: string;
  data?: CandlestickData[];
  height?: number;
  autosize?: boolean;
  onDataRequest?: (symbol: string) => void;
  className?: string;
}

export interface TradingViewWidgetRef {
  updateData: (data: CandlestickData) => void;
  setData: (data: CandlestickData[]) => void;
  getChart: () => IChartApi | null;
  fitContent: () => void;
  scrollToRealtime: () => void;
}

const TradingViewWidget = forwardRef<TradingViewWidgetRef, TradingViewWidgetProps>(
  ({ 
    symbol = 'BTCUSDT',
    data = [],
    height = 400,
    autosize = true,
    onDataRequest,
    className = ''
  }, ref) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
    const { theme } = useTheme();

    // Chart initialization
    useEffect(() => {
      if (!chartContainerRef.current) return;

      const isDark = theme === 'dark';

      const chartOptions = {
        layout: {
          textColor: isDark ? '#D1D5DB' : '#374151',
          background: { 
            color: isDark ? '#1F2937' : '#FFFFFF' 
          },
          fontSize: 12,
          fontFamily: 'ui-sans-serif, system-ui, sans-serif',
        },
        grid: {
          vertLines: { 
            color: isDark ? '#374151' : '#E5E7EB',
            style: 0,
            visible: true,
          },
          horzLines: { 
            color: isDark ? '#374151' : '#E5E7EB',
            style: 0,
            visible: true,
          },
        },
        crosshair: {
          mode: CrosshairMode.Normal,
          vertLine: {
            color: isDark ? '#6B7280' : '#9CA3AF',
            width: 1 as LineWidth,
            style: 2,
          },
          horzLine: {
            color: isDark ? '#6B7280' : '#9CA3AF',
            width: 1 as LineWidth,
            style: 2,
          },
        },
        rightPriceScale: {
          borderColor: isDark ? '#4B5563' : '#D1D5DB',
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        },
        timeScale: {
          borderColor: isDark ? '#4B5563' : '#D1D5DB',
          timeVisible: true,
          secondsVisible: true,
          fixLeftEdge: false,
          fixRightEdge: false,
        },
        width: autosize ? chartContainerRef.current.clientWidth : undefined,
        height: height,
      };

      // Create chart instance
      chartRef.current = createChart(chartContainerRef.current, chartOptions);

      // Add candlestick series
      seriesRef.current = chartRef.current.addSeries(CandlestickSeries, {
        upColor: '#10B981', // Green for bullish candles
        downColor: '#EF4444', // Red for bearish candles
        borderUpColor: '#10B981',
        borderDownColor: '#EF4444',
        wickUpColor: '#10B981',
        wickDownColor: '#EF4444',
        priceLineVisible: true,
        lastValueVisible: true,
        priceFormat: {
          type: 'price',
          precision: 4,
          minMove: 0.0001,
        },
      });

      // Set initial data if provided
      if (data.length > 0) {
        seriesRef.current.setData(data);
        // Fit content to show all data
        chartRef.current.timeScale().fitContent();
      }

      // Request data for symbol if callback provided
      if (onDataRequest) {
        onDataRequest(symbol);
      }

      // Handle resize if autosize is enabled
      const handleResize = () => {
        if (chartRef.current && autosize && chartContainerRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
          });
        }
      };

      if (autosize) {
        window.addEventListener('resize', handleResize);
      }

      // Cleanup function
      return () => {
        if (autosize) {
          window.removeEventListener('resize', handleResize);
        }
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
          seriesRef.current = null;
        }
      };
    }, [theme]); // Re-initialize when theme changes

    // Update chart theme when theme changes
    useEffect(() => {
      if (!chartRef.current) return;

      const isDark = theme === 'dark';

      chartRef.current.applyOptions({
        layout: {
          textColor: isDark ? '#D1D5DB' : '#374151',
          background: { 
            color: isDark ? '#1F2937' : '#FFFFFF' 
          },
        },
        grid: {
          vertLines: { 
            color: isDark ? '#374151' : '#E5E7EB',
          },
          horzLines: { 
            color: isDark ? '#374151' : '#E5E7EB',
          },
        },
        rightPriceScale: {
          borderColor: isDark ? '#4B5563' : '#D1D5DB',
        },
        timeScale: {
          borderColor: isDark ? '#4B5563' : '#D1D5DB',
        },
      });
    }, [theme]);

    // Expose methods to parent components
    useImperativeHandle(ref, () => ({
      updateData: (dataPoint: CandlestickData) => {
        if (seriesRef.current) {
          seriesRef.current.update(dataPoint);
        }
      },
      setData: (newData: CandlestickData[]) => {
        if (seriesRef.current) {
          seriesRef.current.setData(newData);
          chartRef.current?.timeScale().fitContent();
        }
      },
      getChart: () => chartRef.current,
      fitContent: () => {
        if (chartRef.current) {
          chartRef.current.timeScale().fitContent();
        }
      },
      scrollToRealtime: () => {
        if (chartRef.current) {
          chartRef.current.timeScale().scrollToRealTime();
        }
      },
    }), []);

    return (
      <div className={`relative ${className}`}>
        {/* Chart Container */}
        <div 
          ref={chartContainerRef}
          className="w-full"
          style={{ height: `${height}px` }}
        />
        
        {/* Symbol Label */}
        <div className="absolute top-2 left-2 bg-background/80 backdrop-blur-sm rounded px-2 py-1 border text-sm font-medium">
          {symbol}
        </div>

        {/* Real-time Indicator */}
        <div className="absolute top-2 right-2 flex items-center gap-2 bg-background/80 backdrop-blur-sm rounded px-2 py-1 border text-xs">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          <span className="text-muted-foreground">LIVE</span>
        </div>

        {/* Loading State */}
        {data.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50 backdrop-blur-sm">
            <div className="flex items-center gap-2 text-muted-foreground">
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              <span>Loading chart data...</span>
            </div>
          </div>
        )}
      </div>
    );
  }
);

TradingViewWidget.displayName = 'TradingViewWidget';

export default TradingViewWidget; 