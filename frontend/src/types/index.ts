// API Response Types

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

// Delta Exchange API Types

export interface DeltaExchangeMarket {
  id: number;
  symbol: string;
  description: string;
  quoteAsset: string;
  baseAsset: string;
  status: string;
  contractType?: string;
  isActive: boolean;
  markPrice: number;
  indexPrice?: number;
  lastPrice: number;
  volume24h: number;
  openInterest?: number;
  fundingRate?: number;
  nextFundingTime?: string;
  minOrderSize: number;
  maxLeverage: number;
  change24h: number;
}

export interface DeltaExchangePosition {
  id: number;
  symbol: string;
  size: number;
  entryPrice: number;
  markPrice: number;
  liquidationPrice?: number;
  margin: number;
  leverage: number;
  unrealizedPnl: number;
  unrealizedPnlPercentage: number;
  side: 'buy' | 'sell';
  timestamp: string;
  type: 'cross' | 'isolated';
}

export interface DeltaExchangeOrder {
  id: number;
  symbol: string;
  type: DeltaOrderType;
  side: OrderSide;
  price: number;
  size: number;
  status: OrderStatus;
  filledSize: number;
  averageFillPrice?: number;
  created_at: string;
  updated_at: string;
  leverage?: number;
  reduceOnly?: boolean;
  postOnly?: boolean;
}

export interface DeltaExchangeBalance {
  asset: string;
  available: number;
  reserved: number;
  total: number;
}

export type DeltaOrderType = 'market_order' | 'limit_order' | 'stop_market_order' | 'stop_limit_order';
export type OrderSide = 'buy' | 'sell';
export type OrderStatus = 'open' | 'filled' | 'cancelled' | 'rejected' | 'partially_filled';

export interface DeltaOrderRequest {
  symbol: string;
  side: OrderSide;
  orderType: DeltaOrderType;
  size: number;
  limitPrice?: number;
  stopPrice?: number;
  leverage?: number;
  reduceOnly?: boolean;
  postOnly?: boolean;
}

// UI and State Types

export interface MarketData {
  symbol: string;
  price: number;
  change24h: number;
  volume: number;
  high24h?: number;
  low24h?: number;
  lastUpdated: Date;
}

export interface PortfolioData {
  totalBalance: number;
  availableBalance: number;
  totalPnL: number;
  totalPnLPercentage: number;
  dayPnL: number;
  dayPnLPercentage: number;
  positions: DeltaExchangePosition[];
  lastUpdated: Date;
}

export interface TradeSignal {
  id: string;
  symbol: string;
  type: 'buy' | 'sell';
  price: number;
  confidence: number;
  timestamp: string;
  reason: string;
  source: 'ml' | 'technical' | 'manual';
  timeFrame: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
}

// UI Components Types

export type Theme = 'light' | 'dark' | 'system';

export interface ChartOptions {
  interval: '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | '1w';
  showVolume: boolean;
  showGrid: boolean;
  indicators: ChartIndicator[];
  theme: 'light' | 'dark';
}

export interface ChartIndicator {
  type: 'ma' | 'ema' | 'bollinger' | 'rsi' | 'macd' | 'volume';
  params: Record<string, number>;
  color: string;
  visible: boolean;
}

export interface OHLCData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
} 