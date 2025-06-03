/**
 * Bot Management Types
 * TypeScript interfaces for bot management system
 */

export interface Bot {
  id: string;
  userId: string;
  name: string;
  symbol: string;
  strategy: string;
  timeframe: string;
  parameters: Record<string, any>;
  isActive: boolean;
  status?: string;
  createdAt: string;
  updatedAt: string;
  riskSettings?: RiskSettings[];
  positions?: Position[];
  orders?: Order[];
}

export interface BotCreateRequest {
  name: string;
  symbol: string;
  strategy: string;
  timeframe: string;
  parameters?: Record<string, any>;
}

export interface BotUpdateRequest {
  name?: string;
  symbol?: string;
  strategy?: string;
  timeframe?: string;
  parameters?: Record<string, any>;
}

export interface BotStatus {
  id: string;
  name: string;
  symbol: string;
  strategy: string;
  timeframe: string;
  isActive: boolean;
  isRunning?: boolean;
  status: {
    id: string;
    name: string;
    symbol: string;
    strategy: string;
    timeframe: string;
    isActive: boolean;
    status: string;
  };
  lastUpdate: string;
  health: 'good' | 'warning' | 'error' | 'unknown';
  metrics: Record<string, any>;
  activePositions: number;
  errors: BotError[];
  logs: BotLog[];
}

export interface BotError {
  timestamp: string;
  message: string;
  code?: string;
}

export interface BotLog {
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  message: string;
}

export interface RiskSettings {
  id: string;
  name: string;
  description?: string;
  userId: string;
  botId?: string;
  isActive: boolean;
  
  // Position sizing
  positionSizingMethod: string;
  riskPercentage: number;
  maxPositionSize: number;
  kellyFraction?: number;
  winRate?: number;
  customSizingParams?: Record<string, any>;
  
  // Stop loss configuration
  stopLossType: string;
  stopLossValue: number;
  trailingCallback?: number;
  trailingStep?: number;
  timeLimit?: number;
  stopLossLevels?: Record<string, any>;
  
  // Take profit configuration
  takeProfitType: string;
  takeProfitValue: number;
  trailingActivation?: number;
  takeProfitLevels?: Record<string, any>;
  
  // Risk limits
  maxRiskPerTrade: number;
  maxRiskPerSymbol: number;
  maxRiskPerDirection: number;
  maxTotalRisk: number;
  maxDrawdown: number;
  maxPositions: number;
  maxDailyLoss: number;
  cooldownPeriod: number;
  
  // Volatility settings
  volatilityLookback: number;
  
  // Circuit breaker
  circuitBreakerEnabled: boolean;
  maxDailyLossBreaker: number;
  maxDrawdownBreaker: number;
  volatilityMultiplier: number;
  consecutiveLossesBreaker: number;
  tradingPause: number;
  marketWideEnabled: boolean;
  enableManualOverride: boolean;
  
  createdAt: string;
  updatedAt: string;
}

export interface Position {
  id: string;
  userId: string;
  botId?: string;
  symbol: string;
  side: string;
  entryPrice: number;
  currentPrice?: number;
  amount: number;
  leverage: number;
  takeProfitPrice?: number;
  stopLossPrice?: number;
  status: string;
  pnl?: number;
  openedAt: string;
  closedAt?: string;
  metadata?: Record<string, any>;
}

export interface Order {
  id: string;
  status: string;
  symbol: string;
  type: string;
  side: string;
  quantity: number;
  price?: number;
  stopPrice?: number;
  avgFillPrice?: number;
  filledQuantity: number;
  remainingQuantity: number;
}

export interface BotMetrics {
  totalTrades: number;
  winRate: number;
  totalPnL: number;
  totalPnLPercentage: number;
  maxDrawdown: number;
  sharpeRatio: number;
  averageTradeTime: number;
  lastTradeTime?: string;
}

export interface BotHealthData {
  health: 'good' | 'warning' | 'error' | 'unknown';
  metrics: Record<string, any>;
  errors?: BotError[];
  logs?: BotLog[];
}

// Strategy types
export const STRATEGY_TYPES = [
  'ML_PREDICTION',
  'TECHNICAL_ANALYSIS',
  'ARBITRAGE',
  'GRID_TRADING',
  'DCA',
  'CUSTOM'
] as const;

export type StrategyType = typeof STRATEGY_TYPES[number];

// Timeframe types
export const TIMEFRAMES = [
  '1m',
  '5m',
  '15m',
  '30m',
  '1h',
  '4h',
  '1d'
] as const;

export type Timeframe = typeof TIMEFRAMES[number];

// Trading symbols
export const TRADING_SYMBOLS = [
  'BTCUSD',
  'ETHUSD',
  'ADAUSD',
  'SOLUSD',
  'DOTUSD',
  'LINKUSD'
] as const;

export type TradingSymbol = typeof TRADING_SYMBOLS[number];

// Bot status types
export const BOT_STATUS_TYPES = [
  'STOPPED',
  'STARTING',
  'RUNNING',
  'PAUSED',
  'STOPPING',
  'ERROR'
] as const;

export type BotStatusType = typeof BOT_STATUS_TYPES[number];
