/**
 * Order Execution Types
 * Interfaces for the order execution service
 */

import { Timestamp, UUID } from './common';
// The DeltaExchange import isn't needed for now, removing it
// import { DeltaExchange } from './deltaExchange';

/**
 * Order execution status
 */
export enum OrderExecutionStatus {
  PENDING = 'PENDING',         // Order is pending submission
  SUBMITTED = 'SUBMITTED',     // Order has been submitted to the exchange
  PARTIALLY_FILLED = 'PARTIALLY_FILLED', // Order is partially filled
  FILLED = 'FILLED',           // Order is completely filled
  CANCELLED = 'CANCELLED',     // Order was cancelled
  REJECTED = 'REJECTED',       // Order was rejected by the exchange
  EXPIRED = 'EXPIRED'          // Order has expired
}

/**
 * Order type
 */
export enum OrderType {
  MARKET = 'MARKET',     // Market order
  LIMIT = 'LIMIT',       // Limit order
  STOP = 'STOP',         // Stop order
  STOP_LIMIT = 'STOP_LIMIT', // Stop-limit order
  TRAILING_STOP = 'TRAILING_STOP', // Trailing stop order
  OCO = 'OCO'            // One-cancels-the-other order
}

/**
 * Order side
 */
export enum OrderSide {
  BUY = 'BUY',           // Buy order
  SELL = 'SELL'          // Sell order
}

/**
 * Time in force
 */
export enum TimeInForce {
  GTC = 'GTC',           // Good 'til cancelled
  IOC = 'IOC',           // Immediate or cancel
  FOK = 'FOK'            // Fill or kill
}

/**
 * Execution source
 */
export enum ExecutionSource {
  MANUAL = 'MANUAL',     // Manual order from user
  STRATEGY = 'STRATEGY', // Order from a trading strategy
  BOT = 'BOT',           // Order from a trading bot
  SIGNAL = 'SIGNAL',     // Order from a trading signal
  SYSTEM = 'SYSTEM'      // Order from system operations
}

/**
 * Order execution request
 */
export interface OrderExecutionRequest {
  symbol: string;
  type: OrderType;
  side: OrderSide;
  quantity: number;
  price?: number;        // Required for limit orders
  stopPrice?: number;    // Required for stop orders
  timeInForce?: TimeInForce;
  clientOrderId?: string;
  positionId?: string;   // Related position ID
  strategyId?: string;   // Related strategy ID
  botId?: string;        // Related bot ID
  signalId?: string;     // Related signal ID
  source: ExecutionSource;
  userId: string;
  exchangeId: string;    // Which exchange to use
  reduceOnly?: boolean;  // Only reduce position size
  postOnly?: boolean;    // Only maker orders
  leverage?: number;     // Leverage to use
  marginType?: 'ISOLATED' | 'CROSS';
  takeProfitPrice?: number;
  stopLossPrice?: number;
  trailingOffset?: number; // For trailing stop orders
  notes?: string;       // Additional notes
  metadata?: Record<string, any>; // Additional metadata
}

/**
 * Order execution result
 */
export interface OrderExecutionResult {
  id: UUID;
  requestId?: UUID;      // Original request ID
  status: OrderExecutionStatus;
  symbol: string;
  type: OrderType;
  side: OrderSide;
  quantity: number;
  price?: number;
  stopPrice?: number;
  avgFillPrice?: number;
  filledQuantity: number;
  remainingQuantity: number;
  fee?: number;
  feeCurrency?: string;
  clientOrderId?: string;
  exchangeOrderId?: string;
  positionId?: string;
  strategyId?: string;
  botId?: string;
  signalId?: string;
  source: ExecutionSource;
  userId: string;
  exchangeId: string;
  exchangeTimestamp?: Timestamp;
  submittedAt: Timestamp;
  updatedAt: Timestamp;
  completedAt?: Timestamp;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  raw?: any; // Raw exchange response
}

/**
 * Order execution options
 */
export interface OrderExecutionOptions {
  retry?: {
    maxAttempts: number;
    interval: number;  // ms
  };
  timeout?: number;    // ms
  validateOnly?: boolean; // Only validate, don't execute
  dryRun?: boolean;    // Don't actually execute
  smartRouting?: boolean; // Use smart order routing
  slicer?: {           // Split large orders
    enabled: boolean;
    maxSlices: number;
    interval: number;  // ms
  };
  notifications?: {    // Notification preferences
    onSubmission?: boolean;
    onFill?: boolean;
    onCancel?: boolean;
    onError?: boolean;
  };
}

/**
 * Order execution service interface
 */
export interface IOrderExecutionService {
  executeOrder(request: OrderExecutionRequest, options?: OrderExecutionOptions): Promise<OrderExecutionResult>;
  
  cancelOrder(orderId: string, exchangeId: string, userId: string): Promise<boolean>;
  
  getOrder(orderId: string, exchangeId: string, userId: string): Promise<OrderExecutionResult>;
  
  getOrdersByUser(userId: string, status?: OrderExecutionStatus[]): Promise<OrderExecutionResult[]>;
  
  getOrdersByPosition(positionId: string): Promise<OrderExecutionResult[]>;
  
  getOrdersByStrategy(strategyId: string): Promise<OrderExecutionResult[]>;
  
  getOrdersByBot(botId: string): Promise<OrderExecutionResult[]>;
  
  getOrdersBySignal(signalId: string): Promise<OrderExecutionResult[]>;
}

/**
 * Exchange connector interface
 */
export interface IExchangeConnector {
  executeOrder(request: OrderExecutionRequest): Promise<OrderExecutionResult>;
  
  cancelOrder(orderId: string, symbol: string): Promise<boolean>;
  
  getOrder(orderId: string): Promise<OrderExecutionResult>;
  
  getOrderBook(symbol: string, limit?: number): Promise<{
    bids: [number, number][];  // [price, quantity]
    asks: [number, number][];  // [price, quantity]
  }>;
  
  getTicker(symbol: string): Promise<{
    lastPrice: number;
    bidPrice: number;
    askPrice: number;
    volume: number;
    timestamp: Timestamp;
  }>;
}

/**
 * Smart order router interface
 */
export interface ISmartOrderRouter {
  route(request: OrderExecutionRequest): Promise<string>; // Returns exchangeId
  
  getBestExchange(symbol: string, side: OrderSide, quantity: number): Promise<{
    exchangeId: string;
    price: number;
    fee: number;
  }>;
  
  splitOrder(request: OrderExecutionRequest): Promise<OrderExecutionRequest[]>;
}

/**
 * Order slicer interface
 */
export interface IOrderSlicer {
  slice(request: OrderExecutionRequest): OrderExecutionRequest[];
}

/**
 * Market impact estimator interface
 */
export interface IMarketImpactEstimator {
  estimateImpact(symbol: string, quantity: number, side: OrderSide): Promise<{
    priceImpact: number;
    recommendedSlices: number;
    maxSingleOrderSize: number;
  }>;
} 