'use client';

import React, { useEffect, useRef, useState } from 'react';

// Dynamic import state
let chartLibrary: any = null;

// Types for lightweight-charts
interface Time {
  valueOf(): number;
}

interface CandlestickData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
}

// Load lightweight-charts dynamically
const loadChartLibrary = async () => {
  if (chartLibrary) return chartLibrary;

  try {
    console.log('Attempting to load lightweight-charts...');
    const lib = await import('lightweight-charts');
    console.log('Lightweight-charts loaded:', !!lib);
    console.log('Available methods:', Object.keys(lib));
    console.log('createChart function:', typeof lib.createChart);
    chartLibrary = lib;
    return lib;
  } catch (error) {
    console.error('Failed to load lightweight-charts:', error);
    return null;
  }
};

// Error boundary for chart-specific errors
const withChartErrorHandling = (fn: Function, context: string) => {
  return (...args: any[]) => {
    try {
      return fn(...args);
    } catch (error) {
      console.error(`Chart Error (${context}):`, error);
      // Don't throw, just log and continue
    }
  };
};

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

interface LightweightChartProps {
  data: CandleData[];
  positions?: Position[];
  height: number;
  onCrosshairMove?: (price: number | null) => void;
}

export default function LightweightChart({
  data,
  positions = [],
  height,
  onCrosshairMove
}: LightweightChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const candlestickSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const [isChartReady, setIsChartReady] = useState(false);
  const [isLibraryLoaded, setIsLibraryLoaded] = useState(false);
  const [chartLib, setChartLib] = useState<any>(null);

  console.log('LightweightChart render:', {
    dataLength: data.length,
    isLibraryLoaded,
    isChartReady,
    hasChartLib: !!chartLib
  });

  // Load library on mount
  useEffect(() => {
    const initLibrary = async () => {
      console.log('Loading chart library...');
      const lib = await loadChartLibrary();
      if (lib) {
        console.log('Chart library loaded successfully');
        setChartLib(lib);
        setIsLibraryLoaded(true);
      } else {
        console.error('Failed to load chart library');
      }
    };

    initLibrary();
  }, []);

  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current || !isLibraryLoaded || !chartLib) {
      console.log('Chart init blocked:', {
        hasContainer: !!chartContainerRef.current,
        isLibraryLoaded,
        hasChartLib: !!chartLib
      });
      return;
    }

    console.log('Initializing chart...');

    const initializeChart = withChartErrorHandling(() => {
      // Create chart using loaded library
      console.log('Creating chart with library:', chartLib);
      console.log('Available createChart:', typeof chartLib.createChart);

      const chartContainer = chartContainerRef.current!;
      const chart = chartLib.createChart(chartContainer, {
      width: chartContainer.clientWidth,
      height: height - 4, // Account for border
      layout: {
        background: { color: '#0f172a' }, // slate-950
        textColor: '#94a3b8', // slate-400
      },
      grid: {
        vertLines: { color: '#1e293b' }, // slate-800
        horzLines: { color: '#1e293b' }, // slate-800
      },
      crosshair: {
        mode: chartLib.CrosshairMode?.Normal || 1,
        vertLine: {
          color: '#3b82f6', // blue-500
          width: 1,
          style: chartLib.LineStyle?.Dashed || 2,
        },
        horzLine: {
          color: '#3b82f6', // blue-500
          width: 1,
          style: chartLib.LineStyle?.Dashed || 2,
        },
      },
      rightPriceScale: {
        borderColor: '#334155', // slate-700
        textColor: '#94a3b8', // slate-400
      },
      timeScale: {
        borderColor: '#334155', // slate-700
        textColor: '#94a3b8', // slate-400
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    // Create candlestick series with error handling
    let candlestickSeries;
    let volumeSeries;

    try {
      console.log('Creating candlestick series...');
      console.log('Available chart methods:', Object.keys(chart));
      console.log('Chart library methods:', Object.keys(chartLib));

      // Try newer API first (v5.0+)
      if (chartLib.CandlestickSeries) {
        console.log('Using newer API: chart.addSeries(CandlestickSeries, options)');
        candlestickSeries = chart.addSeries(chartLib.CandlestickSeries, {
          upColor: '#10b981', // emerald-500
          downColor: '#ef4444', // red-500
          borderUpColor: '#10b981',
          borderDownColor: '#ef4444',
          wickUpColor: '#10b981',
          wickDownColor: '#ef4444',
          borderVisible: false,
        });
      } else if (chart.addCandlestickSeries) {
        // Fallback to older API (v4.x)
        console.log('Using older API: chart.addCandlestickSeries(options)');
        candlestickSeries = chart.addCandlestickSeries({
          upColor: '#10b981', // emerald-500
          downColor: '#ef4444', // red-500
          borderUpColor: '#10b981',
          borderDownColor: '#ef4444',
          wickUpColor: '#10b981',
          wickDownColor: '#ef4444',
          borderVisible: false,
        });
      } else {
        throw new Error('No candlestick series method available');
      }
      console.log('Candlestick series created successfully');
    } catch (error) {
      console.error('Error creating candlestick series:', error);
      // Fallback to line series if candlestick fails
      try {
        console.log('Falling back to line series...');
        if (chartLib.LineSeries) {
          candlestickSeries = chart.addSeries(chartLib.LineSeries, {
            color: '#10b981',
            lineWidth: 2,
          });
        } else {
          candlestickSeries = chart.addLineSeries({
            color: '#10b981',
            lineWidth: 2,
          });
        }
        console.log('Line series created as fallback');
      } catch (lineError) {
        console.error('Error creating line series:', lineError);
        return; // Exit if both fail
      }
    }

    try {
      console.log('Creating volume series...');
      // Try newer API first (v5.0+)
      if (chartLib.HistogramSeries) {
        console.log('Using newer API: chart.addSeries(HistogramSeries, options)');
        volumeSeries = chart.addSeries(chartLib.HistogramSeries, {
          color: '#64748b', // slate-500
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
        });
      } else if (chart.addHistogramSeries) {
        // Fallback to older API (v4.x)
        console.log('Using older API: chart.addHistogramSeries(options)');
        volumeSeries = chart.addHistogramSeries({
          color: '#64748b', // slate-500
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
        });
      } else {
        throw new Error('No histogram series method available');
      }
      console.log('Volume series created successfully');
    } catch (error) {
      console.error('Error creating volume series:', error);
      // Create a simple histogram series without volume formatting
      try {
        console.log('Creating simple histogram series...');
        if (chartLib.HistogramSeries) {
          volumeSeries = chart.addSeries(chartLib.HistogramSeries, {
            color: '#64748b',
            priceScaleId: 'volume',
          });
        } else {
          volumeSeries = chart.addHistogramSeries({
            color: '#64748b',
            priceScaleId: 'volume',
          });
        }
        console.log('Simple histogram series created');
      } catch (histError) {
        console.error('Error creating histogram series:', histError);
        // Continue without volume series
        volumeSeries = null;
        console.log('Continuing without volume series');
      }
    }

    // Set up volume price scale with error handling
    if (volumeSeries) {
      try {
        chart.priceScale('volume').applyOptions({
          scaleMargins: {
            top: 0.8,
            bottom: 0,
          },
        });
      } catch (error) {
        console.error('Error setting up volume price scale:', error);
      }
    }

    // Store references
    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;
    volumeSeriesRef.current = volumeSeries;

    // Set up crosshair move handler
    if (onCrosshairMove && candlestickSeries) {
      try {
        chart.subscribeCrosshairMove((param: any) => {
          try {
            if (param.point && param.time && candlestickSeries.coordinateToPrice) {
              const price = candlestickSeries.coordinateToPrice(param.point.y);
              onCrosshairMove(price);
            } else {
              onCrosshairMove(null);
            }
          } catch (error) {
            console.error('Error in crosshair move handler:', error);
            onCrosshairMove(null);
          }
        });
      } catch (error) {
        console.error('Error setting up crosshair move handler:', error);
      }
    }

      console.log('Chart initialized successfully');
      console.log('Chart container dimensions:', {
        width: chartContainerRef.current?.offsetWidth,
        height: chartContainerRef.current?.offsetHeight
      });
      setIsChartReady(true);

      // Handle resize
      const handleResize = withChartErrorHandling(() => {
        if (chartContainerRef.current && chart) {
          chart.applyOptions({
            width: chartContainerRef.current.clientWidth,
          });
        }
      }, 'resize');

      window.addEventListener('resize', handleResize);

      // Cleanup
      return () => {
        window.removeEventListener('resize', handleResize);
        if (chart) {
          withChartErrorHandling(() => chart.remove(), 'cleanup')();
        }
        chartRef.current = null;
        candlestickSeriesRef.current = null;
        volumeSeriesRef.current = null;
        setIsChartReady(false);
      };
    }, 'initialization');

    initializeChart();
  }, [height, onCrosshairMove, isLibraryLoaded, chartLib]);

  // Update chart data
  useEffect(() => {
    if (!isChartReady || !candlestickSeriesRef.current) {
      console.log('Chart update blocked:', {
        isChartReady,
        hasCandlestickSeries: !!candlestickSeriesRef.current
      });
      return;
    }

    console.log('Chart update: Received', data.length, 'candles');
    if (data.length === 0) {
      console.log('No chart data available');
      return;
    }

    try {
      // Convert data to lightweight-charts format
      // IMPORTANT: Lightweight Charts expects time in SECONDS, not milliseconds
      const chartData: CandlestickData[] = data.map((candle, index) => {
        const timeInSeconds = Math.floor(candle.time / 1000); // Convert milliseconds to seconds

        // Log first few candles for debugging
        if (index < 3) {
          console.log(`Converting candle ${index}:`, {
            originalTime: candle.time,
            timeInSeconds,
            dateString: new Date(candle.time).toISOString(),
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close
          });
        }

        return {
          time: timeInSeconds as Time,
          open: Number(candle.open),
          high: Number(candle.high),
          low: Number(candle.low),
          close: Number(candle.close),
        };
      });

      const volumeData = data.map((candle) => ({
        time: Math.floor(candle.time / 1000) as Time,
        value: candle.volume || 0,
        color: candle.close >= candle.open ? '#10b98150' : '#ef444450',
      }));

      console.log('Setting chart data:', chartData.length, 'candles');
      console.log('Sample chart data:', chartData[0]);
      console.log('Time range:', chartData[0]?.time, 'to', chartData[chartData.length - 1]?.time);

      // Set data for candlestick series
      candlestickSeriesRef.current.setData(chartData);
      console.log('Candlestick data set successfully');

      // Set data for volume series
      if (volumeSeriesRef.current) {
        volumeSeriesRef.current.setData(volumeData);
        console.log('Volume data set successfully');
      }

      // Fit content to show all data
      if (chartData.length > 0 && chartRef.current) {
        chartRef.current.timeScale().fitContent();
        console.log('Chart fitted to content');
      }

      console.log('Chart data updated successfully');
    } catch (error) {
      console.error('Error updating chart data:', error);
      console.error('Error details:', {
        dataLength: data.length,
        sampleData: data[0],
        error: error instanceof Error ? error.message : String(error)
      });
    }
  }, [data, isChartReady]);

  // Add position markers
  useEffect(() => {
    if (!isChartReady || !candlestickSeriesRef.current || positions.length === 0) return;

    try {
      // Create price lines for positions
      positions.forEach((position) => {
        if (position.status === 'open') {
          // Entry price line
          candlestickSeriesRef.current?.createPriceLine({
            price: position.entryPrice,
            color: position.side === 'buy' ? '#10b981' : '#ef4444',
            lineWidth: 2,
            lineStyle: chartLib?.LineStyle?.Solid || 0,
            axisLabelVisible: true,
            title: `${position.side.toUpperCase()} ${position.size} @ ${position.entryPrice}`,
          });

          // Stop loss line
          candlestickSeriesRef.current?.createPriceLine({
            price: position.stopLoss,
            color: '#ef4444',
            lineWidth: 1,
            lineStyle: chartLib?.LineStyle?.Dashed || 2,
            axisLabelVisible: true,
            title: `SL: ${position.stopLoss}`,
          });

          // Take profit lines
          position.takeProfitLevels.forEach((tp, index) => {
            if (!tp.executed) {
              candlestickSeriesRef.current?.createPriceLine({
                price: tp.price,
                color: '#10b981',
                lineWidth: 1,
                lineStyle: chartLib?.LineStyle?.Dotted || 1,
                axisLabelVisible: true,
                title: `TP${index + 1}: ${tp.price}`,
              });
            }
          });
        }
      });
    } catch (error) {
      console.error('Error adding position markers:', error);
    }
  }, [positions, isChartReady, chartLib]);

  if (!isLibraryLoaded) {
    return (
      <div className="w-full h-full relative flex items-center justify-center bg-slate-900/50 rounded">
        <div className="text-slate-400 text-sm">Loading chart...</div>
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      <div
        ref={chartContainerRef}
        className="w-full h-full"
        style={{ height: `${height}px` }}
      />

      {/* Chart overlay info */}
      <div className="absolute top-2 left-2 bg-slate-900/80 rounded px-2 py-1 text-xs text-slate-300">
        <div className="flex items-center space-x-4">
          <span>Candles: {data.length}</span>
          {positions.length > 0 && (
            <span className="text-blue-400">
              Positions: {positions.filter(p => p.status === 'open').length}
            </span>
          )}
          {!isChartReady && (
            <span className="text-yellow-400">Initializing...</span>
          )}
        </div>
      </div>
    </div>
  );
}
