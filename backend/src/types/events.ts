/**
 * Event Schema and Types for Event-Driven Architecture
 * Defines all event types and schemas for Redis Streams
 */

// ============================================================================
// BASE EVENT INTERFACES
// ============================================================================

export interface BaseEvent {
  id?: string;
  timestamp: number;
  version: string;
  source: string;
  correlationId?: string;
  causationId?: string;
  userId?: string;
  metadata?: Record<string, any>;
}

export interface EventMetadata {
  retryCount?: number;
  processingStartTime?: number;
  processingEndTime?: number;
  processingDuration?: number;
  errorMessage?: string;
  errorCode?: string;
  deadLetterReason?: string;
}

// ============================================================================
// MARKET DATA EVENTS
// ============================================================================

export interface MarketDataEvent extends BaseEvent {
  type: 'MARKET_DATA_RECEIVED' | 'MARKET_DATA_PROCESSED' | 'MARKET_DATA_ERROR';
  data: {
    symbol: string;
    exchange: string;
    price: number;
    volume: number;
    timestamp: number;
    bid?: number;
    ask?: number;
    high24h?: number;
    low24h?: number;
    change?: number;
    changePercent?: number;
    trades?: number;
    raw?: any;
  };
}

export interface OHLCVEvent extends BaseEvent {
  type: 'OHLCV_CANDLE_RECEIVED' | 'OHLCV_CANDLE_PROCESSED';
  data: {
    symbol: string;
    exchange: string;
    timeframe: string;
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    trades?: number;
  };
}

// ============================================================================
// TRADING SIGNAL EVENTS
// ============================================================================

export interface TradingSignalEvent extends BaseEvent {
  type: 'SIGNAL_GENERATED' | 'SIGNAL_VALIDATED' | 'SIGNAL_EXECUTED' | 'SIGNAL_EXPIRED' | 'SIGNAL_ERROR';
  data: {
    signalId: string;
    symbol: string;
    signalType: 'ENTRY' | 'EXIT' | 'INCREASE' | 'DECREASE' | 'HOLD';
    direction: 'LONG' | 'SHORT' | 'NEUTRAL';
    strength: 'VERY_WEAK' | 'WEAK' | 'MODERATE' | 'STRONG' | 'VERY_STRONG';
    timeframe: string;
    price: number;
    targetPrice?: number;
    stopLoss?: number;
    confidenceScore: number;
    expectedReturn: number;
    expectedRisk: number;
    riskRewardRatio: number;
    modelSource: string;
    modelVersion?: string;
    expiresAt?: number;
    validatedAt?: number;
    executedAt?: number;
    predictionValues?: number[];
    features?: Record<string, number>;
  };
}

export interface MLPredictionEvent extends BaseEvent {
  type: 'PREDICTION_GENERATED' | 'PREDICTION_PROCESSED' | 'PREDICTION_ERROR';
  data: {
    predictionId: string;
    modelId: string;
    modelName: string;
    modelVersion: string;
    symbol: string;
    timeframe: string;
    predictionType: 'PRICE' | 'DIRECTION' | 'PROBABILITY';
    values: number[];
    confidenceScores: number[];
    features: Record<string, number>;
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    processingTime: number;
  };
}

// ============================================================================
// ORDER EVENTS
// ============================================================================

export interface OrderEvent extends BaseEvent {
  type: 'ORDER_CREATED' | 'ORDER_SUBMITTED' | 'ORDER_FILLED' | 'ORDER_PARTIALLY_FILLED' | 'ORDER_CANCELLED' | 'ORDER_REJECTED' | 'ORDER_ERROR';
  data: {
    orderId: string;
    clientOrderId?: string;
    exchangeOrderId?: string;
    botId?: string;
    signalId?: string;
    symbol: string;
    side: 'BUY' | 'SELL';
    type: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
    status: 'PENDING' | 'SUBMITTED' | 'FILLED' | 'PARTIALLY_FILLED' | 'CANCELLED' | 'REJECTED';
    exchange: string;
    quantity: number;
    price?: number;
    stopPrice?: number;
    filledQuantity: number;
    remainingQuantity: number;
    avgFillPrice?: number;
    fee?: number;
    feeCurrency?: string;
    submittedAt?: number;
    filledAt?: number;
    cancelledAt?: number;
    rejectedAt?: number;
    errorCode?: string;
    errorMessage?: string;
    latency?: number;
  };
}

export interface PositionEvent extends BaseEvent {
  type: 'POSITION_OPENED' | 'POSITION_UPDATED' | 'POSITION_CLOSED' | 'POSITION_LIQUIDATED' | 'POSITION_ERROR';
  data: {
    positionId: string;
    botId?: string;
    symbol: string;
    side: 'LONG' | 'SHORT';
    entryPrice: number;
    currentPrice?: number;
    quantity: number;
    leverage: number;
    takeProfitPrice?: number;
    stopLossPrice?: number;
    status: 'OPEN' | 'CLOSED' | 'LIQUIDATED';
    pnl?: number;
    pnlPercentage?: number;
    openedAt: number;
    updatedAt?: number;
    closedAt?: number;
    liquidatedAt?: number;
    margin?: number;
    marginRatio?: number;
  };
}

// ============================================================================
// RISK MANAGEMENT EVENTS
// ============================================================================

export interface RiskEvent extends BaseEvent {
  type: 'RISK_CHECK_PASSED' | 'RISK_CHECK_FAILED' | 'RISK_LIMIT_EXCEEDED' | 'RISK_ALERT' | 'CIRCUIT_BREAKER_TRIGGERED';
  data: {
    riskCheckId: string;
    botId?: string;
    symbol?: string;
    riskType: 'POSITION_SIZE' | 'DAILY_LOSS' | 'DRAWDOWN' | 'VOLATILITY' | 'CORRELATION' | 'EXPOSURE';
    currentValue: number;
    threshold: number;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    action: 'ALLOW' | 'BLOCK' | 'REDUCE' | 'CLOSE' | 'PAUSE';
    reason: string;
    recommendations?: string[];
    affectedOrders?: string[];
    affectedPositions?: string[];
  };
}

export interface PortfolioEvent extends BaseEvent {
  type: 'PORTFOLIO_UPDATED' | 'PORTFOLIO_REBALANCED' | 'PORTFOLIO_ALERT';
  data: {
    portfolioId: string;
    botId?: string;
    totalValue: number;
    cashBalance: number;
    unrealizedPnL: number;
    realizedPnL: number;
    dailyPnL: number;
    totalPositions: number;
    openPositions: number;
    drawdown: number;
    maxDrawdown: number;
    sharpeRatio?: number;
    winRate?: number;
    profitFactor?: number;
    allocations: Record<string, number>;
    exposures: Record<string, number>;
  };
}

// ============================================================================
// SYSTEM EVENTS
// ============================================================================

export interface SystemEvent extends BaseEvent {
  type: 'SYSTEM_STARTED' | 'SYSTEM_STOPPED' | 'SYSTEM_ERROR' | 'SYSTEM_HEALTH_CHECK' | 'SYSTEM_ALERT';
  data: {
    component: string;
    status: 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY' | 'DOWN';
    message: string;
    metrics?: Record<string, number>;
    errorDetails?: any;
    uptime?: number;
    memoryUsage?: number;
    cpuUsage?: number;
    connections?: number;
  };
}

export interface PerformanceEvent extends BaseEvent {
  type: 'PERFORMANCE_METRIC' | 'LATENCY_MEASURED' | 'THROUGHPUT_MEASURED';
  data: {
    component: string;
    metric: string;
    value: number;
    unit: string;
    tags?: Record<string, string>;
    threshold?: number;
    status?: 'NORMAL' | 'WARNING' | 'CRITICAL';
  };
}

// ============================================================================
// BOT EVENTS
// ============================================================================

export interface BotEvent extends BaseEvent {
  type: 'BOT_STARTED' | 'BOT_STOPPED' | 'BOT_PAUSED' | 'BOT_RESUMED' | 'BOT_ERROR' | 'BOT_CONFIG_UPDATED';
  data: {
    botId: string;
    botName: string;
    status: 'RUNNING' | 'STOPPED' | 'PAUSED' | 'ERROR';
    symbol: string;
    strategy: string;
    timeframe: string;
    parameters?: Record<string, any>;
    errorMessage?: string;
    performance?: {
      totalTrades: number;
      winRate: number;
      totalPnL: number;
      maxDrawdown: number;
    };
  };
}

// ============================================================================
// UNION TYPES AND UTILITIES
// ============================================================================

export type TradingEvent = 
  | MarketDataEvent
  | OHLCVEvent
  | TradingSignalEvent
  | MLPredictionEvent
  | OrderEvent
  | PositionEvent
  | RiskEvent
  | PortfolioEvent
  | SystemEvent
  | PerformanceEvent
  | BotEvent;

export type EventType = TradingEvent['type'];

// Event stream names
export const STREAM_NAMES = {
  MARKET_DATA: 'market-data-stream',
  TRADING_SIGNALS: 'trading-signals-stream',
  ML_PREDICTIONS: 'ml-predictions-stream',
  ORDERS: 'orders-stream',
  POSITIONS: 'positions-stream',
  RISK_MANAGEMENT: 'risk-management-stream',
  PORTFOLIO: 'portfolio-stream',
  SYSTEM: 'system-stream',
  PERFORMANCE: 'performance-stream',
  BOTS: 'bots-stream',
} as const;

export type StreamName = typeof STREAM_NAMES[keyof typeof STREAM_NAMES];

// Consumer group names
export const CONSUMER_GROUPS = {
  SIGNAL_PROCESSOR: 'signal-processor-group',
  ORDER_EXECUTOR: 'order-executor-group',
  RISK_MANAGER: 'risk-manager-group',
  PORTFOLIO_MANAGER: 'portfolio-manager-group',
  ANALYTICS: 'analytics-group',
  MONITORING: 'monitoring-group',
  NOTIFICATION: 'notification-group',
} as const;

export type ConsumerGroup = typeof CONSUMER_GROUPS[keyof typeof CONSUMER_GROUPS];

// Event priorities
export enum EventPriority {
  LOW = 1,
  NORMAL = 2,
  HIGH = 3,
  CRITICAL = 4,
}

// Event processing status
export enum ProcessingStatus {
  PENDING = 'PENDING',
  PROCESSING = 'PROCESSING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  RETRYING = 'RETRYING',
  DEAD_LETTER = 'DEAD_LETTER',
}

// Event validation schema
export interface EventValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

// Event processing result
export interface EventProcessingResult {
  eventId: string;
  status: ProcessingStatus;
  processingTime: number;
  error?: string;
  result?: any;
  nextEvents?: TradingEvent[];
}

// Event filter
export interface EventFilter {
  types?: EventType[];
  sources?: string[];
  symbols?: string[];
  userIds?: string[];
  botIds?: string[];
  startTime?: number;
  endTime?: number;
  correlationId?: string;
}

// Event aggregation
export interface EventAggregation {
  count: number;
  types: Record<EventType, number>;
  sources: Record<string, number>;
  symbols: Record<string, number>;
  timeRange: {
    start: number;
    end: number;
  };
}

// Utility functions
export function createEventId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function createCorrelationId(): string {
  return `corr-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function isMarketDataEvent(event: TradingEvent): event is MarketDataEvent {
  return event.type.startsWith('MARKET_DATA_');
}

export function isTradingSignalEvent(event: TradingEvent): event is TradingSignalEvent {
  return event.type.startsWith('SIGNAL_');
}

export function isOrderEvent(event: TradingEvent): event is OrderEvent {
  return event.type.startsWith('ORDER_');
}

export function isRiskEvent(event: TradingEvent): event is RiskEvent {
  return event.type.startsWith('RISK_');
}

export function isSystemEvent(event: TradingEvent): event is SystemEvent {
  return event.type.startsWith('SYSTEM_');
}

export function isBotEvent(event: TradingEvent): event is BotEvent {
  return event.type.startsWith('BOT_');
}
