/**
 * Market Data Types and Interfaces
 */

export interface OHLCV {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketDataPoint extends OHLCV {
  symbol: string;
  exchange: string;
  timeframe: string;
}

export interface TechnicalIndicators {
  sma_20?: number;
  sma_50?: number;
  ema_12?: number;
  ema_26?: number;
  rsi?: number;
  macd?: number;
  macd_signal?: number;
  macd_histogram?: number;
  bollinger_upper?: number;
  bollinger_middle?: number;
  bollinger_lower?: number;
  volume_sma?: number;
}

export interface EnhancedMarketData extends MarketDataPoint {
  indicators: TechnicalIndicators;
}

export interface MarketDataRequest {
  symbol: string;
  timeframe: string;
  startDate: Date;
  endDate: Date;
  exchange?: string;
}

export interface MarketDataResponse {
  symbol: string;
  timeframe: string;
  data: MarketDataPoint[];
  count: number;
  startDate: Date;
  endDate: Date;
  source?: string;
}

// Trading Signal Types
export interface TradingSignal {
  id: string;
  timestamp: number;
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  price: number;
  quantity: number;
  confidence: number;
  strategy: string;
  reason: string;
  stopLoss?: number;
  takeProfit?: number;
  riskReward?: number;
}

// Portfolio and Trade Types
export interface Position {
  symbol: string;
  side: 'LONG' | 'SHORT';
  size: number;
  entryPrice: number;
  entryTime: number;
  currentPrice: number;
  unrealizedPnl: number;
  leverage: number;
  stopLoss?: number;
  takeProfitLevels?: TakeProfitLevel[];
  originalSize?: number; // Track original size for partial exits
  partialExits?: PartialExit[];
}

export interface TakeProfitLevel {
  percentage: number; // Percentage of position to close
  priceTarget: number; // Price target for this level
  riskRewardRatio: number; // Risk-reward ratio for this level
  trailingDistance?: number; // Trailing distance in price points
  executed?: boolean; // Whether this level has been executed
  executedAt?: number; // Timestamp when executed
}

export interface PartialExit {
  percentage: number;
  price: number;
  timestamp: number;
  pnl: number;
  reason: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  entryTime: number;
  exitTime: number;
  pnl: number;
  pnlPercent: number;
  commission: number;
  strategy: string;
  reason: string;
  duration: number; // in milliseconds
}

export interface PortfolioSnapshot {
  timestamp: number;
  totalValue: number;
  cash: number;
  positions: Position[];
  totalPnl: number;
  totalPnlPercent: number;
  drawdown: number;
  maxDrawdown: number;
  leverage: number;
}

// Backtesting Configuration
export interface BacktestConfig {
  symbol: string;
  timeframe: string;
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  leverage: number;
  riskPerTrade: number; // percentage
  commission: number; // percentage
  slippage: number; // percentage
  strategy: string;
  parameters: Record<string, any>;
}

// Performance Metrics
export interface PerformanceMetrics {
  totalReturn: number;
  totalReturnPercent: number;
  annualizedReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  winRate: number;
  profitFactor: number;
  averageWin: number;
  averageLoss: number;
  averageWinPercent: number;
  averageLossPercent: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  largestWin: number;
  largestLoss: number;
  averageTradeDuration: number; // in hours
  volatility: number;
  calmarRatio: number;
  recoveryFactor: number;
  payoffRatio: number;
  expectancy: number;
}

export interface BacktestResult {
  config: BacktestConfig;
  performance: PerformanceMetrics;
  trades: Trade[];
  portfolioHistory: PortfolioSnapshot[];
  finalPortfolio: PortfolioSnapshot;
  startTime: number;
  endTime: number;
  duration: number;
  dataPoints: number;
}

// Market Data Provider Interface
export interface MarketDataProvider {
  name: string;
  fetchHistoricalData(request: MarketDataRequest): Promise<MarketDataResponse>;
  isAvailable(): boolean;
}

// Technical Analysis Functions
export interface TechnicalAnalysis {
  calculateSMA(prices: number[], period: number): number[];
  calculateEMA(prices: number[], period: number): number[];
  calculateRSI(prices: number[], period: number): number[];
  calculateMACD(prices: number[], fastPeriod: number, slowPeriod: number, signalPeriod: number): {
    macd: number[];
    signal: number[];
    histogram: number[];
  };
  calculateBollingerBands(prices: number[], period: number, stdDev: number): {
    upper: number[];
    middle: number[];
    lower: number[];
  };
}

// Strategy Interface
export interface TradingStrategy {
  name: string;
  parameters: Record<string, any>;
  generateSignal(data: EnhancedMarketData[], currentIndex: number): TradingSignal | null;
  initialize(config: BacktestConfig): void;
  getDescription(): string;
}

// Event Types for Redis Streams
export interface MarketDataEvent {
  type: 'MARKET_DATA_RECEIVED';
  data: MarketDataPoint;
}

export interface TradingSignalEvent {
  type: 'TRADING_SIGNAL_GENERATED';
  data: TradingSignal;
}

export interface TradeExecutedEvent {
  type: 'TRADE_EXECUTED';
  data: Trade;
}

export interface PortfolioUpdateEvent {
  type: 'PORTFOLIO_UPDATED';
  data: PortfolioSnapshot;
}

// Utility Types
export type TimeFrame = '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w';
export type Exchange = 'binance' | 'coinbase' | 'kraken' | 'mock';

// Constants
export const TIMEFRAMES: Record<TimeFrame, number> = {
  '1m': 60 * 1000,
  '5m': 5 * 60 * 1000,
  '15m': 15 * 60 * 1000,
  '1h': 60 * 60 * 1000,
  '4h': 4 * 60 * 60 * 1000,
  '1d': 24 * 60 * 60 * 1000,
  '1w': 7 * 24 * 60 * 60 * 1000,
};

export const SYMBOLS = {
  BTCUSD: 'BTCUSD',
  ETHUSD: 'ETHUSD',
  ADAUSD: 'ADAUSD',
} as const;

export type Symbol = typeof SYMBOLS[keyof typeof SYMBOLS];
