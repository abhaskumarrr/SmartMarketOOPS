/**
 * Performance Monitoring System
 * Comprehensive metrics collection and monitoring for trading system performance
 * Built with Prometheus integration, real-time alerts, and advanced analytics
 */
import { TradingDecision } from './EnhancedTradingDecisionEngine';
export interface PerformanceMetrics {
    totalTrades: number;
    successfulTrades: number;
    failedTrades: number;
    winRate: number;
    totalPnL: number;
    averageTradeReturn: number;
    systemUptime: number;
    averageLatency: number;
    errorRate: number;
    throughput: number;
    modelAccuracy: number;
    modelPrecision: number;
    modelRecall: number;
    modelF1Score: number;
    currentDrawdown: number;
    maxDrawdown: number;
    sharpeRatio: number;
    volatility: number;
}
export interface AlertRule {
    id: string;
    name: string;
    metric: string;
    condition: 'greater_than' | 'less_than' | 'equals' | 'not_equals';
    threshold: number;
    duration: number;
    severity: 'critical' | 'warning' | 'info';
    enabled: boolean;
    description: string;
}
export interface Alert {
    id: string;
    ruleId: string;
    name: string;
    severity: 'critical' | 'warning' | 'info';
    message: string;
    value: number;
    threshold: number;
    triggeredAt: number;
    resolvedAt?: number;
    status: 'firing' | 'resolved';
}
export interface MonitoringConfig {
    metricsPort: number;
    enableDefaultMetrics: boolean;
    scrapeInterval: number;
    alertCheckInterval: number;
    retentionPeriod: number;
    enableGrafanaIntegration: boolean;
}
export declare class PerformanceMonitoringSystem {
    private decisionEngine;
    private positionManager;
    private riskManager;
    private bridge;
    private dataIntegration;
    private app;
    private server;
    private tradingMetrics;
    private alertRules;
    private activeAlerts;
    private alertHistory;
    private performanceHistory;
    private startTime;
    private config;
    constructor(customConfig?: Partial<MonitoringConfig>);
    /**
     * Initialize the Performance Monitoring System
     */
    initialize(): Promise<void>;
    /**
     * Start the monitoring system
     */
    start(): Promise<void>;
    /**
     * Record trading decision metrics
     */
    recordTradingDecision(decision: TradingDecision, latency: number): void;
    /**
     * Record trade execution metrics
     */
    recordTradeExecution(symbol: string, action: string, status: 'success' | 'failed', pnl?: number, latency?: number): void;
    /**
     * Record API request metrics
     */
    recordApiRequest(method: string, endpoint: string, status: number, latency: number): void;
    /**
     * Record system error
     */
    recordError(component: string, errorType: string): void;
    /**
     * Update risk metrics
     */
    updateRiskMetrics(): void;
    /**
     * Update data quality metrics
     */
    updateDataQualityMetrics(symbol: string, timeframe: string, quality: number, latency: number): void;
    /**
     * Get current performance metrics
     */
    getPerformanceMetrics(): PerformanceMetrics;
    /**
     * Setup Prometheus metrics collection
     */
    private setupPrometheusMetrics;
    /**
     * Setup metrics server endpoints
     */
    private setupMetricsServer;
    /**
     * Initialize default alert rules
     */
    private initializeDefaultAlertRules;
    /**
     * Start monitoring loops
     */
    private startMonitoringLoops;
    /**
     * Update performance history
     */
    private updatePerformanceHistory;
    /**
     * Check alert rules and trigger alerts
     */
    private checkAlerts;
    /**
     * Evaluate individual alert rule
     */
    private evaluateAlertRule;
    /**
     * Get current value for a metric
     */
    private getMetricValue;
    /**
     * Evaluate alert condition
     */
    private evaluateCondition;
    /**
     * Calculate system error rate
     */
    private calculateErrorRate;
    /**
     * Calculate system throughput
     */
    private calculateThroughput;
    /**
     * Cleanup old data
     */
    private cleanupOldData;
    /**
     * Add custom alert rule
     */
    addAlertRule(rule: Omit<AlertRule, 'id'>): string;
    /**
     * Remove alert rule
     */
    removeAlertRule(ruleId: string): boolean;
    /**
     * Get active alerts
     */
    getActiveAlerts(): Alert[];
    /**
     * Get alert history
     */
    getAlertHistory(limit?: number): Alert[];
    /**
     * Get performance history
     */
    getPerformanceHistory(limit?: number): PerformanceMetrics[];
    /**
     * Stop the monitoring system
     */
    stop(): Promise<void>;
    /**
     * Cleanup resources
     */
    cleanup(): Promise<void>;
}
