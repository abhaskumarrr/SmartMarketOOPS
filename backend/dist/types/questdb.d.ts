/**
 * QuestDB TypeScript Interfaces
 * Type definitions for QuestDB time-series data structures
 */
export interface QuestDBTimestamp {
    timestamp: Date | string | number;
}
export interface QuestDBRecord extends QuestDBTimestamp {
    [key: string]: any;
}
export interface QuestDBQueryResult<T = any> {
    query: string;
    columns: Array<{
        name: string;
        type: string;
    }>;
    dataset: T[];
    count: number;
    timings: {
        compiler: number;
        execute: number;
        count: number;
    };
}
export interface QuestDBMetric extends QuestDBTimestamp {
    name: string;
    value: number;
    tags?: Record<string, string | number>;
}
export interface QuestDBMetricQuery {
    name?: string;
    names?: string[];
    startTime?: Date | string;
    endTime?: Date | string;
    tags?: Record<string, string | number>;
    limit?: number;
    orderBy?: 'timestamp' | 'value' | 'name';
    orderDirection?: 'ASC' | 'DESC';
}
export interface QuestDBMetricAggregation {
    timestamp: Date | string;
    avg_value?: number;
    sum_value?: number;
    min_value?: number;
    max_value?: number;
    count_value?: number;
    first_value?: number;
    last_value?: number;
}
export interface QuestDBTradingSignal extends QuestDBTimestamp {
    id: string;
    symbol: string;
    type: 'ENTRY' | 'EXIT' | 'INCREASE' | 'DECREASE' | 'HOLD';
    direction: 'LONG' | 'SHORT' | 'NEUTRAL';
    strength: 'VERY_WEAK' | 'WEAK' | 'MODERATE' | 'STRONG' | 'VERY_STRONG';
    timeframe: string;
    source: string;
    price: number;
    target_price?: number;
    stop_loss?: number;
    confidence_score: number;
    expected_return: number;
    expected_risk: number;
    risk_reward_ratio: number;
}
export interface QuestDBTradingSignalQuery {
    id?: string;
    symbol?: string;
    symbols?: string[];
    type?: string;
    direction?: string;
    strength?: string;
    timeframe?: string;
    source?: string;
    startTime?: Date | string;
    endTime?: Date | string;
    minConfidence?: number;
    limit?: number;
    orderBy?: 'timestamp' | 'confidence_score' | 'expected_return';
    orderDirection?: 'ASC' | 'DESC';
}
export interface QuestDBMLPrediction extends QuestDBTimestamp {
    id: string;
    model_id: string;
    symbol: string;
    timeframe: string;
    prediction_type: 'PRICE' | 'DIRECTION' | 'PROBABILITY';
    values: string;
    confidence_scores: string;
}
export interface QuestDBMLPredictionParsed extends Omit<QuestDBMLPrediction, 'values' | 'confidence_scores'> {
    values: number[];
    confidence_scores: number[];
}
export interface QuestDBMLPredictionQuery {
    id?: string;
    model_id?: string;
    symbol?: string;
    symbols?: string[];
    timeframe?: string;
    prediction_type?: string;
    startTime?: Date | string;
    endTime?: Date | string;
    limit?: number;
    orderBy?: 'timestamp' | 'model_id' | 'symbol';
    orderDirection?: 'ASC' | 'DESC';
}
export interface QuestDBPerformanceMetric extends QuestDBTimestamp {
    system: 'API' | 'ML' | 'TRADING' | 'DATABASE' | 'FRONTEND' | 'WEBSOCKET';
    component: string;
    metric: string;
    unit: string;
    value: number;
    tags?: Record<string, string | number>;
}
export interface QuestDBPerformanceMetricQuery {
    system?: string;
    systems?: string[];
    component?: string;
    components?: string[];
    metric?: string;
    metrics?: string[];
    startTime?: Date | string;
    endTime?: Date | string;
    tags?: Record<string, string | number>;
    limit?: number;
    orderBy?: 'timestamp' | 'value' | 'system' | 'component';
    orderDirection?: 'ASC' | 'DESC';
}
export interface QuestDBMarketData extends QuestDBTimestamp {
    symbol: string;
    exchange: string;
    timeframe: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    trades?: number;
}
export interface QuestDBMarketDataQuery {
    symbol?: string;
    symbols?: string[];
    exchange?: string;
    exchanges?: string[];
    timeframe?: string;
    timeframes?: string[];
    startTime?: Date | string;
    endTime?: Date | string;
    limit?: number;
    orderBy?: 'timestamp' | 'volume' | 'close';
    orderDirection?: 'ASC' | 'DESC';
}
export interface QuestDBOrderExecution extends QuestDBTimestamp {
    order_id: string;
    user_id: string;
    bot_id?: string;
    symbol: string;
    side: 'BUY' | 'SELL';
    type: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';
    status: 'PENDING' | 'FILLED' | 'PARTIALLY_FILLED' | 'CANCELLED' | 'REJECTED';
    exchange: string;
    quantity: number;
    price?: number;
    filled_quantity: number;
    avg_fill_price?: number;
    fee?: number;
    latency_ms: number;
}
export interface QuestDBOrderExecutionQuery {
    order_id?: string;
    user_id?: string;
    bot_id?: string;
    symbol?: string;
    symbols?: string[];
    side?: string;
    type?: string;
    status?: string;
    exchange?: string;
    startTime?: Date | string;
    endTime?: Date | string;
    limit?: number;
    orderBy?: 'timestamp' | 'latency_ms' | 'quantity';
    orderDirection?: 'ASC' | 'DESC';
}
export interface QuestDBPortfolioSnapshot extends QuestDBTimestamp {
    user_id: string;
    bot_id?: string;
    total_value: number;
    cash_balance: number;
    unrealized_pnl: number;
    realized_pnl: number;
    total_positions: number;
    daily_pnl: number;
    drawdown: number;
}
export interface QuestDBPortfolioSnapshotQuery {
    user_id?: string;
    bot_id?: string;
    startTime?: Date | string;
    endTime?: Date | string;
    limit?: number;
    orderBy?: 'timestamp' | 'total_value' | 'daily_pnl';
    orderDirection?: 'ASC' | 'DESC';
}
export interface QuestDBRiskMetric extends QuestDBTimestamp {
    user_id: string;
    bot_id?: string;
    symbol?: string;
    metric_type: string;
    value: number;
    threshold: number;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}
export interface QuestDBRiskMetricQuery {
    user_id?: string;
    bot_id?: string;
    symbol?: string;
    symbols?: string[];
    metric_type?: string;
    metric_types?: string[];
    severity?: string;
    severities?: string[];
    startTime?: Date | string;
    endTime?: Date | string;
    limit?: number;
    orderBy?: 'timestamp' | 'value' | 'severity';
    orderDirection?: 'ASC' | 'DESC';
}
export interface QuestDBSampleByOptions {
    interval: string;
    aggregation?: 'AVG' | 'SUM' | 'MIN' | 'MAX' | 'COUNT' | 'FIRST' | 'LAST';
    fillMode?: 'NULL' | 'PREV' | 'LINEAR' | 'NONE';
}
export interface QuestDBTimeSeriesPoint {
    timestamp: Date | string;
    value: number;
}
export interface QuestDBTimeSeriesData {
    series: QuestDBTimeSeriesPoint[];
    metadata: {
        symbol?: string;
        metric?: string;
        aggregation?: string;
        interval?: string;
        count: number;
        startTime: Date | string;
        endTime: Date | string;
    };
}
export interface QuestDBBatchInsertOptions {
    tableName: string;
    data: QuestDBRecord[];
    batchSize?: number;
    autoFlush?: boolean;
    validateData?: boolean;
}
export interface QuestDBBatchInsertResult {
    success: boolean;
    recordsInserted: number;
    errors: Array<{
        index: number;
        error: string;
        record: QuestDBRecord;
    }>;
    duration: number;
    throughput: number;
}
export interface QuestDBMigrationConfig {
    sourceTable: string;
    targetTable: string;
    batchSize: number;
    parallelWorkers: number;
    validateData: boolean;
    dryRun: boolean;
    startDate?: Date;
    endDate?: Date;
}
export interface QuestDBMigrationProgress {
    totalRecords: number;
    processedRecords: number;
    successfulRecords: number;
    failedRecords: number;
    progress: number;
    estimatedTimeRemaining: number;
    throughput: number;
    errors: Array<{
        record: any;
        error: string;
        timestamp: Date;
    }>;
}
export interface QuestDBMigrationResult {
    success: boolean;
    totalRecords: number;
    migratedRecords: number;
    failedRecords: number;
    duration: number;
    averageThroughput: number;
    errors: Array<{
        record: any;
        error: string;
    }>;
}
export type QuestDBTableName = 'metrics' | 'trading_signals' | 'ml_predictions' | 'performance_metrics' | 'market_data' | 'order_executions' | 'portfolio_snapshots' | 'risk_metrics';
export type QuestDBDataType = 'BOOLEAN' | 'BYTE' | 'SHORT' | 'INT' | 'LONG' | 'FLOAT' | 'DOUBLE' | 'STRING' | 'SYMBOL' | 'TIMESTAMP' | 'DATE' | 'BINARY';
export interface QuestDBColumnInfo {
    name: string;
    type: QuestDBDataType;
    indexed: boolean;
    designated: boolean;
}
export interface QuestDBTableInfo {
    name: string;
    columns: QuestDBColumnInfo[];
    partitionBy: string;
    maxUncommittedRows: number;
    commitLag: number;
}
export interface QuestDBError {
    message: string;
    position?: number;
    query?: string;
    code?: string;
}
export interface QuestDBConnectionError extends QuestDBError {
    host: string;
    port: number;
    retryAttempt: number;
}
export interface QuestDBQueryError extends QuestDBError {
    query: string;
    executionTime: number;
}
export interface QuestDBIngestionError extends QuestDBError {
    tableName: string;
    recordCount: number;
    failedRecords: any[];
}
