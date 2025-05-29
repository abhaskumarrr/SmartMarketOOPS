import React, { useEffect, useRef, useState } from 'react';
import { io, Socket } from 'socket.io-client';

// Define prediction data interface
interface PredictionData {
  time: number | string;
  value: number;
  confidence: number;
}

interface TradeSignal {
  time: number | string;
  type: 'buy' | 'sell' | 'hold';
  confidence?: number;
}

interface TradingViewChartContainerProps {
  symbol: string;
  interval: string;
  darkMode?: boolean;
  predictionsData?: PredictionData[];
  signals?: TradeSignal[];
  showPredictions?: boolean;
  showIndicators?: boolean;
}

// TradingView Widget API declaration
declare global {
  interface Window {
    TradingView: {
      widget: new (config: any) => any;
    };
  }
}

export const TradingViewChartContainer: React.FC<TradingViewChartContainerProps> = ({
  symbol,
  interval = '1h',
  darkMode = true,
  predictionsData = [],
  signals = [],
  showPredictions = true,
  showIndicators = true,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const widgetRef = useRef<any>(null);
  const socketRef = useRef<Socket | null>(null);
  const [isChartReady, setIsChartReady] = useState(false);
  const [latestPredictions, setLatestPredictions] = useState<PredictionData[]>(predictionsData);
  const [latestSignals, setLatestSignals] = useState<TradeSignal[]>(signals);

  // Load TradingView widget script
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => {
      if (chartContainerRef.current && window.TradingView) {
        initializeChart();
      }
    };
    document.head.appendChild(script);

    return () => {
      document.head.removeChild(script);
    };
  }, []);

  // Initialize the chart
  const initializeChart = () => {
    if (!chartContainerRef.current || !window.TradingView) return;

    // Clear previous widget if exists
    if (widgetRef.current) {
      chartContainerRef.current.innerHTML = '';
    }

    const widgetOptions = {
      symbol: symbol,
      interval: interval,
      container: chartContainerRef.current,
      datafeed: {
        onReady: (callback: Function) => {
          setTimeout(() => callback({
            supported_resolutions: ['1', '5', '15', '30', '60', '240', 'D', 'W', 'M'],
            exchanges: [{ value: '', name: 'All Exchanges', desc: '' }],
          }), 0);
        },
        resolveSymbol: (symbolName: string, onSymbolResolvedCallback: Function) => {
          setTimeout(() => {
            onSymbolResolvedCallback({
              name: symbolName,
              full_name: symbolName,
              description: symbolName,
              type: 'crypto',
              session: '24x7',
              timezone: 'Etc/UTC',
              has_intraday: true,
              has_daily: true,
              has_weekly_and_monthly: true,
              minmov: 1,
              pricescale: 100,
              volume_precision: 8,
              data_status: 'streaming',
            });
          }, 0);
        },
        getBars: (symbolInfo: any, resolution: string, from: number, to: number, onHistoryCallback: Function) => {
          // In a real implementation, this would fetch historical data from your API
          // For this example, we'll just return a simple dummy data
          const bars = [];
          let lastClose = 40000 + Math.random() * 2000;
          
          for (let i = from; i <= to; i += 3600) {
            const open = lastClose + Math.random() * 200 - 100;
            const high = open + Math.random() * 100;
            const low = open - Math.random() * 100;
            const close = (open + high + low) / 3 + Math.random() * 50 - 25;
            
            bars.push({
              time: i * 1000,
              open: open,
              high: high,
              low: low,
              close: close,
              volume: Math.random() * 100
            });
            
            lastClose = close;
          }
          
          onHistoryCallback(bars, { noData: bars.length === 0 });
        },
        subscribeBars: (symbolInfo: any, resolution: string, onRealtimeCallback: Function, subscriberUID: string) => {
          // This would be connected to your WebSocket for real-time data
          socketRef.current?.on('market:data', (payload) => {
            if (payload.symbol === symbol && payload.data) {
              // Format the data properly for TradingView
              onRealtimeCallback({
                time: payload.data.time * 1000,
                open: payload.data.open || payload.data.value,
                high: payload.data.high || payload.data.value,
                low: payload.data.low || payload.data.value,
                close: payload.data.close || payload.data.value,
                volume: payload.data.volume || 0
              });
            }
          });
        },
        unsubscribeBars: () => {
          // Cleanup WebSocket listeners when unsubscribing
        },
      },
      library_path: 'https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js',
      locale: 'en',
      theme: darkMode ? 'dark' : 'light',
      disabled_features: [
        'header_symbol_search',
        'use_localstorage_for_settings',
      ],
      enabled_features: [
        'study_templates',
        'save_chart_properties_to_local_storage',
      ],
      charts_storage_url: 'https://saveload.tradingview.com',
      charts_storage_api_version: '1.1',
      client_id: 'tradingview.com',
      user_id: 'public_user',
      fullscreen: false,
      autosize: true,
      studies_overrides: {},
      overrides: {
        'mainSeriesProperties.candleStyle.upColor': '#26a69a',
        'mainSeriesProperties.candleStyle.downColor': '#ef5350',
        'mainSeriesProperties.candleStyle.wickUpColor': '#26a69a',
        'mainSeriesProperties.candleStyle.wickDownColor': '#ef5350',
      },
      time_frames: [
        { text: '1D', resolution: '5' },
        { text: '1W', resolution: '15' },
        { text: '1M', resolution: '60' },
        { text: '3M', resolution: 'D' },
        { text: '1Y', resolution: 'W' },
      ],
      debug: false,
      loading_screen: { backgroundColor: darkMode ? '#181A20' : '#ffffff' },
    };

    widgetRef.current = new window.TradingView.widget(widgetOptions);
    
    widgetRef.current.onChartReady(() => {
      setIsChartReady(true);
      applyPredictionOverlay();
      applySignalMarkers();
    });
  };

  // Connect to WebSocket for real-time data
  useEffect(() => {
    // Connect to backend WebSocket
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:3001';
    const socket = io(wsUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      timeout: 10000,
    });
    socketRef.current = socket;

    socket.on('connect', () => {
      socket.emit('subscribe:market', symbol);
    });

    socket.on('market:predictions', (payload) => {
      if (payload.symbol === symbol && Array.isArray(payload.data)) {
        setLatestPredictions(payload.data);
      }
    });

    socket.on('market:signals', (payload) => {
      if (payload.symbol === symbol && Array.isArray(payload.data)) {
        setLatestSignals(payload.data);
      }
    });

    socket.on('disconnect', () => {
      // Optionally handle disconnect
    });

    socket.on('error', (err) => {
      // Handle error
      console.error('WebSocket error:', err);
    });

    return () => {
      socket.emit('unsubscribe:market', symbol);
      socket.disconnect();
    };
  }, [symbol]);

  // Apply prediction overlay when chart is ready or predictions change
  useEffect(() => {
    if (isChartReady && showPredictions) {
      applyPredictionOverlay();
    }
  }, [isChartReady, latestPredictions, showPredictions]);

  // Apply signal markers when chart is ready or signals change
  useEffect(() => {
    if (isChartReady) {
      applySignalMarkers();
    }
  }, [isChartReady, latestSignals]);

  // Function to apply prediction overlay
  const applyPredictionOverlay = () => {
    if (!widgetRef.current || !isChartReady || !showPredictions) return;

    try {
      const chart = widgetRef.current.chart();
      
      // Remove existing prediction line if any
      chart.removeEntity('prediction_line');
      
      if (latestPredictions.length === 0) return;
      
      // Create prediction line data points
      const lineData = latestPredictions.map(pred => ({
        time: typeof pred.time === 'string' ? new Date(pred.time).getTime() / 1000 : pred.time,
        value: pred.value,
        confidence: pred.confidence
      }));
      
      // Add prediction line with custom styling
      chart.createStudy(
        'Linear Regression', 
        false, 
        false, 
        {
          inputs: { source: 'close' },
          precision: 2,
          style: 2, // Line style
          linewidth: 2,
          color: '#f6c175',
          bgcolor: 'rgba(246, 193, 117, 0.2)',
          transparency: 40,
          linestyle: 0,
          showPrevClose: false,
          showLabels: true,
          baseIndex: 0
        },
        { id: 'prediction_line' }
      );
      
      // Add confidence area around the prediction line
      // This would normally require custom Pine Script in real TradingView
      // For our widget, we can simulate with multiple lines or bands
    } catch (error) {
      console.error('Failed to apply prediction overlay:', error);
    }
  };

  // Function to apply signal markers
  const applySignalMarkers = () => {
    if (!widgetRef.current || !isChartReady) return;

    try {
      const chart = widgetRef.current.chart();
      
      // Remove existing signals
      chart.removeAllShapes();
      
      // Add new signal markers
      latestSignals.forEach((signal, index) => {
        const time = typeof signal.time === 'string' ? new Date(signal.time).getTime() / 1000 : signal.time;
        
        const shape = {
          time: time,
          price: 0, // This should be set to the actual price at that time
          shape: signal.type === 'buy' ? 'arrow_up' : signal.type === 'sell' ? 'arrow_down' : 'circle',
          color: signal.type === 'buy' ? '#26a69a' : signal.type === 'sell' ? '#ef5350' : '#888888',
          text: `${signal.type.toUpperCase()} ${signal.confidence ? `(${Math.round(signal.confidence * 100)}%)` : ''}`,
          tooltip: `${signal.type.toUpperCase()} Signal${signal.confidence ? ` - Confidence: ${Math.round(signal.confidence * 100)}%` : ''}`,
          size: 2,
          lock: true,
        };
        
        chart.createShape(shape, { id: `signal_${index}` });
      });
    } catch (error) {
      console.error('Failed to apply signal markers:', error);
    }
  };

  return (
    <div className="trading-view-chart-container">
      <div 
        ref={chartContainerRef} 
        style={{ 
          width: '100%', 
          height: 600,
          borderRadius: '8px',
          overflow: 'hidden',
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
        }} 
      />
      {!isChartReady && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: darkMode ? 'rgba(24, 26, 32, 0.7)' : 'rgba(255, 255, 255, 0.7)',
          color: darkMode ? '#fff' : '#000',
          fontSize: '16px',
          zIndex: 10
        }}>
          Loading chart...
        </div>
      )}
    </div>
  );
}; 