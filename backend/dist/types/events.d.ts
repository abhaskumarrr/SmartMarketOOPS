/**
 * Event Schema and Types for Event-Driven Architecture
 * Defines all event types and schemas for Redis Streams
 */
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
export type TradingEvent = MarketDataEvent | OHLCVEvent | TradingSignalEvent | MLPredictionEvent | OrderEvent | PositionEvent | RiskEvent | PortfolioEvent | SystemEvent | PerformanceEvent | BotEvent;
export type EventType = TradingEvent['type'];
export declare const STREAM_NAMES: {
    readonly MARKET_DATA: "market-data-stream";
    readonly TRADING_SIGNALS: "trading-signals-stream";
    readonly ML_PREDICTIONS: "ml-predictions-stream";
    readonly ORDERS: "orders-stream";
    readonly POSITIONS: "positions-stream";
    readonly RISK_MANAGEMENT: "risk-management-stream";
    readonly PORTFOLIO: "portfolio-stream";
    readonly SYSTEM: "system-stream";
    readonly PERFORMANCE: "performance-stream";
    readonly BOTS: "bots-stream";
};
export type StreamName = typeof STREAM_NAMES[keyof typeof STREAM_NAMES];
export declare const CONSUMER_GROUPS: {
    readonly SIGNAL_PROCESSOR: "signal-processor-group";
    readonly ORDER_EXECUTOR: "order-executor-group";
    readonly RISK_MANAGER: "risk-manager-group";
    readonly PORTFOLIO_MANAGER: "portfolio-manager-group";
    readonly ANALYTICS: "analytics-group";
    readonly MONITORING: "monitoring-group";
    readonly NOTIFICATION: "notification-group";
};
export type ConsumerGroup = typeof CONSUMER_GROUPS[keyof typeof CONSUMER_GROUPS];
export declare enum EventPriority {
    LOW = 1,
    NORMAL = 2,
    HIGH = 3,
    CRITICAL = 4
}
export declare enum ProcessingStatus {
    PENDING = "PENDING",
    PROCESSING = "PROCESSING",
    COMPLETED = "COMPLETED",
    FAILED = "FAILED",
    RETRYING = "RETRYING",
    DEAD_LETTER = "DEAD_LETTER"
}
export interface EventValidationResult {
    valid: boolean;
    errors: string[];
    warnings: string[];
}
export interface EventProcessingResult {
    eventId: string;
    status: ProcessingStatus;
    processingTime: number;
    error?: string;
    result?: any;
    nextEvents?: TradingEvent[];
}
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
export declare function createEventId(): string;
export declare function createCorrelationId(): string;
export declare function isMarketDataEvent(event: TradingEvent): event is MarketDataEvent;
export declare function isTradingSignalEvent(event: TradingEvent): event is TradingSignalEvent;
export declare function isOrderEvent(event: TradingEvent): event is OrderEvent;
export declare function isRiskEvent(event: TradingEvent): event is RiskEvent;
export declare function isSystemEvent(event: TradingEvent): event is SystemEvent;
export declare function isBotEvent(event: TradingEvent): event is BotEvent;
