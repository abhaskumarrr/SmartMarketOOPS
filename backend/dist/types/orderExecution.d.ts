/**
 * Order Execution Types
 * Interfaces for the order execution service
 */
import { Timestamp, UUID } from './common';
/**
 * Order execution status
 */
export declare enum OrderExecutionStatus {
    PENDING = "PENDING",// Order is pending submission
    SUBMITTED = "SUBMITTED",// Order has been submitted to the exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED",// Order is partially filled
    FILLED = "FILLED",// Order is completely filled
    CANCELLED = "CANCELLED",// Order was cancelled
    REJECTED = "REJECTED",// Order was rejected by the exchange
    EXPIRED = "EXPIRED"
}
/**
 * Order type
 */
export declare enum OrderType {
    MARKET = "MARKET",// Market order
    LIMIT = "LIMIT",// Limit order
    STOP = "STOP",// Stop order
    STOP_LIMIT = "STOP_LIMIT",// Stop-limit order
    TRAILING_STOP = "TRAILING_STOP",// Trailing stop order
    OCO = "OCO"
}
/**
 * Order side
 */
export declare enum OrderSide {
    BUY = "BUY",// Buy order
    SELL = "SELL"
}
/**
 * Time in force
 */
export declare enum TimeInForce {
    GTC = "GTC",// Good 'til cancelled
    IOC = "IOC",// Immediate or cancel
    FOK = "FOK"
}
/**
 * Execution source
 */
export declare enum ExecutionSource {
    MANUAL = "MANUAL",// Manual order from user
    STRATEGY = "STRATEGY",// Order from a trading strategy
    BOT = "BOT",// Order from a trading bot
    SIGNAL = "SIGNAL",// Order from a trading signal
    SYSTEM = "SYSTEM"
}
/**
 * Order execution request
 */
export interface OrderExecutionRequest {
    symbol: string;
    type: OrderType;
    side: OrderSide;
    quantity: number;
    price?: number;
    stopPrice?: number;
    timeInForce?: TimeInForce;
    clientOrderId?: string;
    positionId?: string;
    strategyId?: string;
    botId?: string;
    signalId?: string;
    source: ExecutionSource;
    userId: string;
    exchangeId: string;
    reduceOnly?: boolean;
    postOnly?: boolean;
    leverage?: number;
    marginType?: 'ISOLATED' | 'CROSS';
    takeProfitPrice?: number;
    stopLossPrice?: number;
    trailingOffset?: number;
    notes?: string;
    metadata?: Record<string, any>;
}
/**
 * Order execution result
 */
export interface OrderExecutionResult {
    id: UUID;
    requestId?: UUID;
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
    raw?: any;
}
/**
 * Order execution options
 */
export interface OrderExecutionOptions {
    retry?: {
        maxAttempts: number;
        interval: number;
    };
    timeout?: number;
    validateOnly?: boolean;
    dryRun?: boolean;
    smartRouting?: boolean;
    slicer?: {
        enabled: boolean;
        maxSlices: number;
        interval: number;
    };
    notifications?: {
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
        bids: [number, number][];
        asks: [number, number][];
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
    route(request: OrderExecutionRequest): Promise<string>;
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
