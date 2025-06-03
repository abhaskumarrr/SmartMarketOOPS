/**
 * Real-Time Price Chart Component
 * Task #30: Real-Time Trading Dashboard
 * TradingView-style chart with real-time updates and signal overlays
 */

'use client';

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  ChartOptions,
  TooltipItem,
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { useTradingStore } from '../../lib/stores/tradingStore';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface PricePoint {
  timestamp: number;
  price: number;
  volume: number;
}

interface SignalPoint {
  timestamp: number;
  price: number;
  type: 'buy' | 'sell' | 'strong_buy' | 'strong_sell';
  confidence: number;
  quality: string;
}

interface RealTimePriceChartProps {
  symbol: string;
  height?: number;
  showSignals?: boolean;
  showVolume?: boolean;
  timeframe?: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
}

export const RealTimePriceChart: React.FC<RealTimePriceChartProps> = ({
  symbol,
  height = 400,
  showSignals = true,
  showVolume = false,
  timeframe = '5m'
}) => {
  const chartRef = useRef<ChartJS<'line'>>(null);
  const [priceData, setPriceData] = useState<PricePoint[]>([]);
  const [signalData, setSignalData] = useState<SignalPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Get data from trading store with stable selectors
  const marketData = useTradingStore(useCallback((state) => state.marketData?.[symbol], [symbol]));
  const allTradingSignals = useTradingStore((state) => state.tradingSignals);
  const isConnected = useTradingStore((state) => state.isConnected);

  // Memoize filtered trading signals to prevent infinite loops
  const tradingSignals = useMemo(() => {
    return allTradingSignals?.filter(signal => signal.symbol === symbol) || [];
  }, [allTradingSignals, symbol]);

  // Update price data when market data changes
  useEffect(() => {
    if (marketData) {
      setPriceData(prev => {
        const newPoint: PricePoint = {
          timestamp: marketData.timestamp,
          price: marketData.price,
          volume: marketData.volume
        };

        // Add new point and keep last 200 points for memory efficiency
        const updated = [...prev, newPoint].slice(-200);

        // Remove duplicates based on timestamp
        const unique = updated.filter((point, index, arr) =>
          index === arr.findIndex(p => p.timestamp === point.timestamp)
        );

        return unique.sort((a, b) => a.timestamp - b.timestamp);
      });
      setIsLoading(false);
    }
  }, [marketData]);

  // Update signal data when trading signals change
  useEffect(() => {
    if (showSignals && tradingSignals.length > 0) {
      const signals: SignalPoint[] = tradingSignals
        .filter(signal => signal.signal_type !== 'hold')
        .map(signal => ({
          timestamp: signal.timestamp,
          price: signal.price,
          type: signal.signal_type as 'buy' | 'sell' | 'strong_buy' | 'strong_sell',
          confidence: signal.confidence,
          quality: signal.quality
        }))
        .slice(-50); // Keep last 50 signals for performance

      setSignalData(signals);
    }
  }, [tradingSignals, showSignals]);

  // Chart configuration
  const chartData = useMemo(() => {
    const labels = priceData.map(point => new Date(point.timestamp));
    const prices = priceData.map(point => point.price);

    const datasets = [
      {
        label: `${symbol} Price`,
        data: prices,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4,
      }
    ];

    // Add volume dataset if enabled
    if (showVolume) {
      const volumes = priceData.map(point => point.volume);
      datasets.push({
        label: 'Volume',
        data: volumes,
        borderColor: 'rgba(156, 163, 175, 0.5)',
        backgroundColor: 'rgba(156, 163, 175, 0.2)',
        borderWidth: 1,
        fill: true,
        tension: 0,
        pointRadius: 0,
        yAxisID: 'volume',
      } as any);
    }

    // Add signal overlays
    if (showSignals && signalData.length > 0) {
      const buySignals = signalData.filter(s => s.type === 'buy' || s.type === 'strong_buy');
      const sellSignals = signalData.filter(s => s.type === 'sell' || s.type === 'strong_sell');

      if (buySignals.length > 0) {
        datasets.push({
          label: 'Buy Signals',
          data: buySignals.map(signal => ({
            x: new Date(signal.timestamp),
            y: signal.price
          })),
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgb(34, 197, 94)',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false,
          pointStyle: 'triangle',
        } as any);
      }

      if (sellSignals.length > 0) {
        datasets.push({
          label: 'Sell Signals',
          data: sellSignals.map(signal => ({
            x: new Date(signal.timestamp),
            y: signal.price
          })),
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgb(239, 68, 68)',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false,
          pointStyle: 'triangle',
          rotation: 180,
        } as any);
      }
    }

    return {
      labels,
      datasets
    };
  }, [priceData, signalData, showSignals, showVolume, symbol]);

  const chartOptions: ChartOptions<'line'> = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index',
    },
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          filter: (legendItem) => legendItem.text !== 'Volume' || showVolume
        }
      },
      title: {
        display: true,
        text: `${symbol} - Real-Time Price Chart`,
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        callbacks: {
          title: (context) => {
            return new Date(context[0].parsed.x).toLocaleString();
          },
          label: (context: TooltipItem<'line'>) => {
            const label = context.dataset.label || '';
            if (label.includes('Signal')) {
              const signalPoint = signalData.find(s =>
                Math.abs(new Date(s.timestamp).getTime() - context.parsed.x) < 60000
              );
              if (signalPoint) {
                return [
                  `${label}: $${context.parsed.y?.toFixed(2)}`,
                  `Confidence: ${(signalPoint.confidence * 100).toFixed(1)}%`,
                  `Quality: ${signalPoint.quality}`
                ];
              }
            }
            if (label === 'Volume') {
              return `${label}: ${context.parsed.y?.toLocaleString()}`;
            }
            return `${label}: $${context.parsed.y?.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'time',
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
            day: 'MMM dd'
          }
        },
        title: {
          display: true,
          text: 'Time'
        }
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        title: {
          display: true,
          text: 'Price ($)'
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.2)'
        }
      },
      ...(showVolume && {
        volume: {
          type: 'linear',
          display: true,
          position: 'right',
          title: {
            display: true,
            text: 'Volume'
          },
          grid: {
            drawOnChartArea: false,
          },
          max: Math.max(...priceData.map(p => p.volume)) * 2
        }
      })
    },
    animation: {
      duration: 0 // Disable animations for real-time performance
    },
    elements: {
      point: {
        radius: 0 // Hide points by default for performance
      }
    }
  }), [symbol, showVolume, signalData, priceData]);

  // Connection status indicator
  const connectionStatus = useMemo(() => {
    if (!isConnected) {
      return (
        <div className="absolute top-2 right-2 flex items-center space-x-2 bg-red-100 text-red-800 px-2 py-1 rounded text-sm">
          <div className="w-2 h-2 bg-red-500 rounded-full"></div>
          <span>Disconnected</span>
        </div>
      );
    }
    return (
      <div className="absolute top-2 right-2 flex items-center space-x-2 bg-green-100 text-green-800 px-2 py-1 rounded text-sm">
        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
        <span>Live</span>
      </div>
    );
  }, [isConnected]);

  // Loading state
  if (isLoading) {
    return (
      <div className="relative" style={{ height }}>
        <div className="flex items-center justify-center h-full bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-gray-600">Loading chart data...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative bg-white rounded-lg shadow-sm border" style={{ height }}>
      {connectionStatus}

      {/* Chart Controls */}
      <div className="absolute top-2 left-2 flex space-x-2 z-10">
        <div className="bg-white bg-opacity-90 rounded px-2 py-1 text-sm font-medium">
          {timeframe.toUpperCase()}
        </div>
        {marketData && (
          <div className="bg-white bg-opacity-90 rounded px-2 py-1 text-sm">
            <span className={`font-medium ${marketData.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              ${marketData.price.toFixed(2)} ({marketData.changePercent >= 0 ? '+' : ''}{marketData.changePercent.toFixed(2)}%)
            </span>
          </div>
        )}
      </div>

      {/* Chart */}
      <div className="p-4 h-full">
        <Line ref={chartRef} data={chartData} options={chartOptions} />
      </div>

      {/* Signal Quality Indicator */}
      {showSignals && signalData.length > 0 && (
        <div className="absolute bottom-2 left-2 bg-white bg-opacity-90 rounded px-2 py-1 text-xs">
          <span className="text-gray-600">Latest Signal: </span>
          <span className={`font-medium ${
            signalData[signalData.length - 1]?.type.includes('buy') ? 'text-green-600' : 'text-red-600'
          }`}>
            {signalData[signalData.length - 1]?.type.toUpperCase()}
          </span>
          <span className="text-gray-500 ml-1">
            ({(signalData[signalData.length - 1]?.confidence * 100).toFixed(0)}%)
          </span>
        </div>
      )}
    </div>
  );
};
