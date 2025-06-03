/**
 * Mock QuestDB Service for Testing
 * Provides a mock implementation for testing without actual QuestDB
 */
export interface MetricData {
    timestamp: Date | number;
    name: string;
    value: number;
    tags?: Record<string, string | number>;
}
export interface TradingSignalData {
    timestamp: Date | number;
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
export interface MLPredictionData {
    timestamp: Date | number;
    id: string;
    modelId: string;
    symbol: string;
    timeframe: string;
    predictionType: string;
    values: number[];
    confidenceScores: number[];
    metadata?: Record<string, any>;
}
export interface PerformanceMetricData {
    timestamp: Date | number;
    system: string;
    component: string;
    metric: string;
    unit: string;
    value: number;
    tags?: Record<string, string | number>;
}
export declare class MockQuestDBService {
    private static instance;
    private isInitialized;
    private data;
    private constructor();
    static getInstance(): MockQuestDBService;
    initialize(): Promise<void>;
    shutdown(): Promise<void>;
    insertMetric(data: MetricData): Promise<void>;
    insertMetrics(metrics: MetricData[]): Promise<void>;
    insertTradingSignal(data: TradingSignalData): Promise<void>;
    insertMLPrediction(data: MLPredictionData): Promise<void>;
    insertPerformanceMetric(data: PerformanceMetricData): Promise<void>;
    executeQuery(query: string): Promise<any[]>;
    healthCheck(): Promise<boolean>;
    getTableStats(tableName: string): Promise<any>;
    flush(): Promise<void>;
    isReady(): boolean;
    getStoredData(tableName: string): any[];
    getTotalRecords(): number;
    clearData(): void;
    getStats(): {
        isInitialized: boolean;
        totalRecords: number;
        tableStats: Record<string, number>;
    };
}
export declare const mockQuestdbService: MockQuestDBService;
