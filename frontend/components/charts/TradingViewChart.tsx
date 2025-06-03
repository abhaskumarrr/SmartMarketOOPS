'use client';

import React, { useState, useEffect, useRef } from 'react';
import { BarChart3, TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import dynamic from 'next/dynamic';

// Dynamically import the chart component to avoid SSR issues
const LightweightChart = dynamic(() => import('./LightweightChart'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-slate-950">
      <div className="text-center">
        <Loader2 className="w-8 h-8 text-blue-500 animate-spin mx-auto mb-2" />
        <p className="text-sm text-slate-400">Loading chart...</p>
      </div>
    </div>
  ),
});

interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface Position {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  entryPrice: number;
  currentPrice?: number;
  stopLoss: number;
  takeProfitLevels: Array<{
    percentage: number;
    price: number;
    executed: boolean;
  }>;
  status: 'open' | 'closed';
}

interface TradingViewChartProps {
  symbol: string;
  data: CandleData[];
  positions?: Position[];
  height?: number;
  onCrosshairMove?: (price: number | null) => void;
}

export function TradingViewChart({
  symbol,
  data,
  positions = [],
  height = 400,
  onCrosshairMove
}: TradingViewChartProps) {
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1m');
  const [isLoading, setIsLoading] = useState(true);

  // Calculate price change from data
  useEffect(() => {
    console.log('TradingViewChart: Data received:', {
      dataLength: data.length,
      symbol,
      sampleData: data[0],
      isLoading
    });

    if (data.length > 1) {
      const latest = data[data.length - 1];
      const previous = data[data.length - 2];
      const change = ((latest.close - previous.close) / previous.close) * 100;

      console.log('TradingViewChart: Price calculation:', {
        latest: latest.close,
        previous: previous.close,
        change
      });

      setCurrentPrice(latest.close);
      setPriceChange(change);
      setIsLoading(false);
    } else if (data.length === 1) {
      console.log('TradingViewChart: Single candle data:', data[0]);
      setCurrentPrice(data[0].close);
      setIsLoading(false);
    } else {
      console.log('TradingViewChart: No data available');
    }
  }, [data, symbol]);

  const isPositive = priceChange >= 0;

  return (
    <div className="relative">
      {/* Chart Header */}
      <div className="flex items-center justify-between p-4 bg-slate-900/50 border-b border-slate-800">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-semibold text-white">{symbol}</h3>
          {currentPrice && (
            <div className="flex items-center space-x-2">
              <span className="text-2xl font-bold text-white">
                ${currentPrice.toFixed(2)}
              </span>
              <span className={`text-sm flex items-center ${isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
                {isPositive ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                {isPositive ? '+' : ''}{priceChange.toFixed(2)}%
              </span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {/* Timeframe buttons */}
          {['1m', '5m', '15m', '1h', '4h', '1d'].map((timeframe) => (
            <button
              key={timeframe}
              onClick={() => setSelectedTimeframe(timeframe)}
              className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
                selectedTimeframe === timeframe
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {timeframe}
            </button>
          ))}
        </div>
      </div>

      {/* Chart Container */}
      <div
        className="w-full bg-slate-950 border border-slate-800 relative"
        style={{ height: `${height}px` }}
      >
        {isLoading || data.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center">
              <BarChart3 className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-slate-400 mb-2">Loading Chart Data</h3>
              <p className="text-sm text-slate-500 mb-4">
                Fetching {symbol} candlestick data...
              </p>
              <div className="text-xs text-slate-600">
                {data.length > 0 ? `${data.length} candles loaded` : 'Waiting for data...'}
              </div>
              <div className="text-xs text-slate-500 mt-2">
                Debug: isLoading={isLoading.toString()}, dataLength={data.length}
              </div>
            </div>
          </div>
        ) : (
          <LightweightChart
            data={data}
            positions={positions}
            height={height}
            onCrosshairMove={onCrosshairMove}
          />
        )}
      </div>

      {/* Chart Footer */}
      <div className="flex items-center justify-between p-2 bg-slate-900/50 border-t border-slate-800 text-xs text-slate-400">
        <div className="flex items-center space-x-4">
          <span>Volume: 1.2M</span>
          <span>24h High: $2,650.00</span>
          <span>24h Low: $2,450.00</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
            <span>Live Data</span>
          </div>
        </div>
      </div>
    </div>
  );
}
