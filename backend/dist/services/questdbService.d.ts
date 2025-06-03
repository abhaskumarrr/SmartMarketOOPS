/**
 * QuestDB Service
 * High-performance time-series database service for SmartMarketOOPS
 * Provides optimized operations for financial time-series data
 */
export interface TimeSeriesDataPoint {
    timestamp: Date | number;
    symbol?: string;
    value?: number;
    tags?: Record<string, string | number>;
    fields?: Record<string, number | string | boolean>;
}
export interface QueryOptions {
    limit?: number;
    offset?: number;
    orderBy?: string;
    orderDirection?: 'ASC' | 'DESC';
    where?: string;
    groupBy?: string;
    having?: string;
}
export interface MetricData extends TimeSeriesDataPoint {
    name: string;
    value: number;
    tags?: Record<string, string | number>;
}
export interface TradingSignalData extends TimeSeriesDataPoint {
    id: string;
    symbol: string;
    type: string;
    direction: string;
    strength: string;
    timeframe: string;
    price: number;
    targetPrice?: number;
    stopLoss?: number;
    confidenceScore: number;
    expectedReturn: number;
    expectedRisk: number;
    riskRewardRatio: number;
    source: string;
    metadata?: Record<string, any>;
}
export interface MLPredictionData extends TimeSeriesDataPoint {
    id: string;
    modelId: string;
    symbol: string;
    timeframe: string;
    predictionType: string;
    values: number[];
    confidenceScores: number[];
    metadata?: Record<string, any>;
}
export interface PerformanceMetricData extends TimeSeriesDataPoint {
    system: string;
    component: string;
    metric: string;
    value: number;
    unit: string;
    tags?: Record<string, string | number>;
}
export declare class QuestDBService {
    private static instance;
    private client;
    private constructor();
    static getInstance(): QuestDBService;
    initialize(): Promise<void>;
    shutdown(): Promise<void>;
    private ensureClient;
    insertMetric(data: MetricData): Promise<void>;
    insertMetrics(metrics: MetricData[]): Promise<void>;
    insertTradingSignal(data: TradingSignalData): Promise<void>;
    insertMLPrediction(data: MLPredictionData): Promise<void>;
    insertPerformanceMetric(data: PerformanceMetricData): Promise<void>;
    insertMarketData(data: {
        timestamp: Date;
        symbol: string;
        exchange: string;
        timeframe: string;
        open: number;
        high: number;
        low: number;
        close: number;
        volume: number;
    }): Promise<void>;
    insertMarketDataBatch(dataPoints: Array<{
        timestamp: Date;
        symbol: string;
        exchange: string;
        timeframe: string;
        open: number;
        high: number;
        low: number;
        close: number;
        volume: number;
    }>): Promise<void>;
    insertTrade(data: {
        timestamp: Date;
        id: string;
        symbol: string;
        side: string;
        entryPrice: number;
        exitPrice: number;
        quantity: number;
        entryTime: Date;
        exitTime: Date;
        pnl: number;
        pnlPercent: number;
        commission: number;
        strategy: string;
        reason: string;
        duration: number;
    }): Promise<void>;
    insertPortfolioSnapshot(data: {
        timestamp: Date;
        totalValue: number;
        cash: number;
        totalPnl: number;
        totalPnlPercent: number;
        drawdown: number;
        maxDrawdown: number;
        leverage: number;
        positionCount: number;
    }): Promise<void>;
    batchInsert(tableName: string, data: TimeSeriesDataPoint[], formatter: (item: TimeSeriesDataPoint) => string): Promise<void>;
    executeQuery(query: string): Promise<any[]>;
    getMetricsByTimeRange(metricName: string, startTime: Date, endTime: Date, options?: QueryOptions): Promise<any[]>;
    getTradingSignalsBySymbol(symbol: string, startTime: Date, endTime: Date, options?: QueryOptions): Promise<any[]>;
    getLatestMetrics(metricNames: string[], limit?: number): Promise<any[]>;
    getMetricAggregation(metricName: string, aggregation: 'AVG' | 'SUM' | 'MIN' | 'MAX' | 'COUNT', interval: string, startTime: Date, endTime: Date): Promise<any[]>;
    healthCheck(): Promise<boolean>;
    getTableStats(tableName: string): Promise<any>;
    flush(): Promise<void>;
    isReady(): boolean;
}
export declare const questdbService: QuestDBService;
