/**
 * Performance Monitoring Script
 * Monitors system performance and generates reports
 */
interface PerformanceReport {
    timestamp: string;
    system: {
        uptime: number;
        memory: NodeJS.MemoryUsage;
        cpu: any;
    };
    api: {
        responseTime: number;
        throughput: number;
        errorRate: number;
    };
    database: {
        avgQueryTime: number;
        slowQueries: number;
        connectionPool: any;
    };
    cache: {
        hitRate: number;
        memoryUsage: number;
        operations: any;
    };
}
declare class PerformanceMonitor {
    private baseUrl;
    private monitoringInterval;
    private isRunning;
    constructor(baseUrl?: string, monitoringInterval?: number);
    start(): Promise<void>;
    stop(): void;
    generateReport(): Promise<PerformanceReport>;
    private getCpuUsage;
    analyzeReport(report: PerformanceReport): Promise<void>;
    private calculateHealthScore;
    private sleep;
    runOnce(): Promise<void>;
    loadTest(duration?: number, concurrency?: number): Promise<void>;
}
export { PerformanceMonitor };
