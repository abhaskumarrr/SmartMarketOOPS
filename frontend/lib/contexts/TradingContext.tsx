import React, { createContext, useState, useContext, useEffect } from 'react';
import useWebSocket from '../hooks/useWebSocket';

// Define types for market data
export interface MarketData {
  symbol: string;
  price: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  change: number;
  changePercent: number;
  time: number;
}

export interface TradingSignal {
  id: string;
  symbol: string;
  action: 'Buy' | 'Sell' | 'Hold';
  confidence: number;
  timestamp: string;
  source: string;
  details?: string;
}

export interface Position {
  id: string;
  symbol: string;
  type: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  liquidationPrice?: number;
  leverage?: number;
  margin?: number;
  timestamp: string;
}

export interface OrderData {
  id: string;
  symbol: string;
  type: 'limit' | 'market' | 'stop' | 'stop_limit';
  side: 'buy' | 'sell';
  size: number;
  price?: number;
  status: 'open' | 'filled' | 'canceled' | 'rejected';
  filledSize?: number;
  avgFillPrice?: number;
  timestamp: string;
}

interface TradingContextType {
  // Selected trading pair
  selectedSymbol: string;
  setSelectedSymbol: (symbol: string) => void;
  
  // Market data
  marketData: Record<string, MarketData>;
  isMarketDataLoading: boolean;
  
  // Trading pairs
  availablePairs: string[];
  
  // Candlestick data
  candlestickData: any[];
  timeframe: string;
  setTimeframe: (timeframe: string) => void;
  
  // Trading signals
  tradingSignals: TradingSignal[];
  
  // Portfolio data
  positions: Position[];
  orders: OrderData[];
  
  // WebSocket status
  isWebSocketConnected: boolean;
  
  // UI state
  isDarkMode: boolean;
  toggleDarkMode: () => void;
}

// Create context with default values
const TradingContext = createContext<TradingContextType>({
  selectedSymbol: 'BTCUSDT',
  setSelectedSymbol: () => {},
  marketData: {},
  isMarketDataLoading: true,
  availablePairs: [],
  candlestickData: [],
  timeframe: '1h',
  setTimeframe: () => {},
  tradingSignals: [],
  positions: [],
  orders: [],
  isWebSocketConnected: false,
  isDarkMode: true,
  toggleDarkMode: () => {},
});

// Mock data generation helper
const generateMockMarketData = (): Record<string, MarketData> => {
  const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT'];
  const data: Record<string, MarketData> = {};

  symbols.forEach(symbol => {
    const basePrice = symbol.includes('BTC') ? 40000 : 
                      symbol.includes('ETH') ? 2800 : 
                      symbol.includes('BNB') ? 350 : 
                      symbol.includes('SOL') ? 90 :
                      symbol.includes('ADA') ? 1.2 : 0.1;
    
    const variation = (Math.random() * 0.03 - 0.015) * basePrice;
    const price = basePrice + variation;
    const change = variation;
    const changePercent = (variation / basePrice) * 100;
    
    data[symbol] = {
      symbol,
      price,
      open: price - (Math.random() * 0.01 * price),
      high: price + (Math.random() * 0.01 * price),
      low: price - (Math.random() * 0.02 * price),
      close: price,
      volume: Math.random() * 1000000,
      change,
      changePercent,
      time: Date.now(),
    };
  });

  return data;
};

// Provider component
export const TradingProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  // State for the trading context
  const [selectedSymbol, setSelectedSymbol] = useState<string>('BTCUSDT');
  const [marketData, setMarketData] = useState<Record<string, MarketData>>(generateMockMarketData());
  const [isMarketDataLoading, setIsMarketDataLoading] = useState<boolean>(true);
  const [availablePairs, setAvailablePairs] = useState<string[]>(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT']);
  const [candlestickData, setCandlestickData] = useState<any[]>([]);
  const [timeframe, setTimeframe] = useState<string>('1h');
  const [tradingSignals, setTradingSignals] = useState<TradingSignal[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<OrderData[]>([]);
  const [isDarkMode, setIsDarkMode] = useState<boolean>(true);

  // Connect to WebSocket for real-time data
  const { data: wsData, isConnected } = useWebSocket<any>(
    'market', 
    'market:data', 
    selectedSymbol
  );

  // Handle WebSocket data updates
  useEffect(() => {
    if (wsData && wsData.type === 'marketData') {
      setMarketData(prevData => ({
        ...prevData,
        [wsData.symbol]: {
          ...wsData.data,
          symbol: wsData.symbol,
          time: Date.now(),
        }
      }));
    } else if (wsData && wsData.type === 'candlestick') {
      setCandlestickData(wsData.data);
    } else if (wsData && wsData.type === 'signal') {
      setTradingSignals(prev => [wsData.data, ...prev]);
    }
  }, [wsData]);

  // Generate mock data on initial load
  useEffect(() => {
    // Simulate loading delay
    const timer = setTimeout(() => {
      setIsMarketDataLoading(false);
      
      // Generate mock positions
      const mockPositions: Position[] = [
        {
          id: '1',
          symbol: 'BTCUSDT',
          type: 'long',
          size: 0.05,
          entryPrice: 39500,
          currentPrice: marketData['BTCUSDT']?.price || 40000,
          pnl: 0.05 * (marketData['BTCUSDT']?.price - 39500) || 25,
          pnlPercent: ((marketData['BTCUSDT']?.price / 39500) - 1) * 100 || 1.25,
          leverage: 10,
          margin: 0.05 * 39500 / 10,
          timestamp: new Date().toISOString(),
        },
        {
          id: '2',
          symbol: 'ETHUSDT',
          type: 'short',
          size: 0.5,
          entryPrice: 2900,
          currentPrice: marketData['ETHUSDT']?.price || 2850,
          pnl: 0.5 * (2900 - (marketData['ETHUSDT']?.price || 2850)),
          pnlPercent: (1 - (marketData['ETHUSDT']?.price || 2850) / 2900) * 100,
          leverage: 5,
          margin: 0.5 * 2900 / 5,
          timestamp: new Date().toISOString(),
        }
      ];
      setPositions(mockPositions);
      
      // Generate mock orders
      const mockOrders: OrderData[] = [
        {
          id: '1',
          symbol: 'BTCUSDT',
          type: 'limit',
          side: 'buy',
          size: 0.1,
          price: 38500,
          status: 'open',
          timestamp: new Date().toISOString(),
        },
        {
          id: '2',
          symbol: 'ETHUSDT',
          type: 'stop',
          side: 'sell',
          size: 1.0,
          price: 2700,
          status: 'open',
          timestamp: new Date().toISOString(),
        }
      ];
      setOrders(mockOrders);
      
      // Generate mock trading signals
      const mockSignals: TradingSignal[] = [
        {
          id: '1',
          symbol: 'BTCUSDT',
          action: 'Buy',
          confidence: 87,
          timestamp: new Date().toISOString(),
          source: 'ML Model v1.2',
          details: 'Strong bullish divergence detected on RSI',
        },
        {
          id: '2',
          symbol: 'ETHUSDT',
          action: 'Hold',
          confidence: 62,
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          source: 'ML Model v1.2',
          details: 'Consolidation pattern forming',
        },
        {
          id: '3',
          symbol: 'SOLUSDT',
          action: 'Sell',
          confidence: 75,
          timestamp: new Date(Date.now() - 7200000).toISOString(),
          source: 'ML Model v1.2',
          details: 'Bearish trend confirmed by volume analysis',
        }
      ];
      setTradingSignals(mockSignals);
      
      // Generate mock candlestick data
      const mockCandlesticks = [];
      const now = Math.floor(Date.now() / 1000);
      const hourInSeconds = 3600;
      let lastClose = 40000;
      
      for (let i = 0; i < 100; i++) {
        const time = now - (i * hourInSeconds);
        const open = lastClose;
        const high = open * (1 + Math.random() * 0.02);
        const low = open * (1 - Math.random() * 0.02);
        const close = low + Math.random() * (high - low);
        const volume = Math.random() * 1000;
        
        mockCandlesticks.unshift({
          time,
          open,
          high,
          low,
          close,
          volume
        });
        
        lastClose = close;
      }
      
      setCandlestickData(mockCandlesticks);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);

  // Function to toggle dark mode
  const toggleDarkMode = () => {
    setIsDarkMode(prev => !prev);
  };

  // Return the provider with the context value
  return (
    <TradingContext.Provider
      value={{
        selectedSymbol,
        setSelectedSymbol,
        marketData,
        isMarketDataLoading,
        availablePairs,
        candlestickData,
        timeframe,
        setTimeframe,
        tradingSignals,
        positions,
        orders,
        isWebSocketConnected: isConnected,
        isDarkMode,
        toggleDarkMode,
      }}
    >
      {children}
    </TradingContext.Provider>
  );
};

// Custom hook to use the trading context
export const useTradingContext = () => useContext(TradingContext);

export default TradingContext; 