/**
 * Database Optimization Service
 * Handles connection pooling, query optimization, and performance monitoring
 */
interface QueryMetrics {
    query: string;
    duration: number;
    timestamp: number;
    success: boolean;
    error?: string;
}
interface DatabaseStats {
    totalQueries: number;
    avgQueryTime: number;
    slowQueries: number;
    failedQueries: number;
    connectionPoolSize: number;
    activeConnections: number;
}
declare class DatabaseOptimizationService {
    private prisma;
    private queryMetrics;
    private slowQueryThreshold;
    private maxMetricsHistory;
    private stats;
    constructor();
    private setupQueryLogging;
    private addQueryMetric;
    executeQuery<T>(queryFn: () => Promise<T>, cacheKey?: string, cacheTtl?: number): Promise<T>;
    batchInsert<T>(model: any, data: T[], batchSize?: number): Promise<void>;
    paginatedQuery<T>(queryFn: (skip: number, take: number) => Promise<T[]>, countFn: () => Promise<number>, page?: number, limit?: number, cacheKey?: string): Promise<{
        data: T[];
        total: number;
        page: number;
        limit: number;
        totalPages: number;
        hasNext: boolean;
        hasPrev: boolean;
    }>;
    getConnectionInfo(): Promise<any>;
    getSlowQueries(limit?: number): QueryMetrics[];
    getQueryStats(): DatabaseStats;
    healthCheck(): Promise<{
        status: 'healthy' | 'degraded' | 'unhealthy';
        details: any;
    }>;
    analyzeIndexUsage(): Promise<any>;
    getTableStats(): Promise<any>;
    cleanup(): Promise<void>;
    disconnect(): Promise<void>;
}
export declare const createDatabaseOptimizationService: () => DatabaseOptimizationService;
export declare const getDatabaseOptimizationService: () => DatabaseOptimizationService | null;
export { DatabaseOptimizationService, QueryMetrics, DatabaseStats };
