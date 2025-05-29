/**
 * TradingView Chart Configuration
 * Contains settings and utilities for configuring the TradingView chart
 */

// Default indicator settings
export const defaultIndicators = [
  {
    id: 'macd',
    name: 'MACD',
    params: {
      fastLength: 12,
      slowLength: 26,
      signalLength: 9,
    },
  },
  {
    id: 'rsi',
    name: 'RSI',
    params: {
      length: 14,
      overbought: 70,
      oversold: 30,
    },
  },
  {
    id: 'ma',
    name: 'Moving Average',
    params: {
      length: 50,
      type: 'sma',
    },
  },
  {
    id: 'ma2',
    name: 'Moving Average',
    params: {
      length: 200,
      type: 'sma',
    },
  },
  {
    id: 'bbands',
    name: 'Bollinger Bands',
    params: {
      length: 20,
      stdDev: 2,
    },
  },
];

// Chart theme styles
export const chartThemes = {
  dark: {
    backgroundColor: '#181A20',
    textColor: '#D9D9D9',
    gridColor: '#222',
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderColor: '#363A45',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
  },
  light: {
    backgroundColor: '#FFFFFF',
    textColor: '#131722',
    gridColor: '#F0F3FA',
    upColor: '#089981',
    downColor: '#F23645',
    borderColor: '#D1D4DC',
    wickUpColor: '#089981',
    wickDownColor: '#F23645',
  },
};

// Prediction overlay styles
export const predictionStyles = {
  lineColor: '#f6c175',
  areaColor: 'rgba(246, 193, 117, 0.2)',
  lineWidth: 2,
  lineStyle: 'solid', // solid, dashed, dotted
  confidence: {
    high: 'rgba(246, 193, 117, 0.6)',
    medium: 'rgba(246, 193, 117, 0.4)',
    low: 'rgba(246, 193, 117, 0.2)',
  },
};

// Signal marker styles
export const signalStyles = {
  buy: {
    shape: 'arrow_up',
    color: '#26a69a',
    textColor: '#FFFFFF',
    size: 2,
  },
  sell: {
    shape: 'arrow_down',
    color: '#ef5350',
    textColor: '#FFFFFF',
    size: 2,
  },
  hold: {
    shape: 'circle',
    color: '#888888',
    textColor: '#FFFFFF',
    size: 1,
  },
};

// Time intervals
export const timeIntervals = [
  { text: '1m', value: '1' },
  { text: '5m', value: '5' },
  { text: '15m', value: '15' },
  { text: '30m', value: '30' },
  { text: '1h', value: '60' },
  { text: '4h', value: '240' },
  { text: '1d', value: 'D' },
  { text: '1w', value: 'W' },
  { text: '1M', value: 'M' },
];

// Utility function to format prediction data
export const formatPredictionData = (data: any[]) => {
  if (!Array.isArray(data)) return [];
  
  return data.map(item => ({
    time: typeof item.time === 'string' ? new Date(item.time).getTime() / 1000 : item.time,
    value: parseFloat(item.value),
    confidence: item.confidence ? parseFloat(item.confidence) : 0.5,
  }));
};

// Utility function to format signal data
export const formatSignalData = (data: any[]) => {
  if (!Array.isArray(data)) return [];
  
  return data.map(item => ({
    time: typeof item.time === 'string' ? new Date(item.time).getTime() / 1000 : item.time,
    type: item.type || 'hold',
    confidence: item.confidence ? parseFloat(item.confidence) : undefined,
  }));
};

// Helper to get confidence level category
export const getConfidenceLevel = (confidence: number): 'high' | 'medium' | 'low' => {
  if (confidence >= 0.7) return 'high';
  if (confidence >= 0.4) return 'medium';
  return 'low';
};

// Supported technical indicators
export const technicalIndicators = [
  { id: 'macd', name: 'MACD' },
  { id: 'rsi', name: 'RSI' },
  { id: 'sma', name: 'Simple Moving Average' },
  { id: 'ema', name: 'Exponential Moving Average' },
  { id: 'bbands', name: 'Bollinger Bands' },
  { id: 'ichimoku', name: 'Ichimoku Cloud' },
  { id: 'stoch', name: 'Stochastic' },
  { id: 'adx', name: 'ADX' },
  { id: 'atr', name: 'Average True Range' },
  { id: 'obv', name: 'On-Balance Volume' },
]; 