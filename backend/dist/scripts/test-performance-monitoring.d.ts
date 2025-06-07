#!/usr/bin/env node
/**
 * Performance Monitoring System Test
 * Comprehensive testing of Prometheus metrics, alerts, and monitoring functionality
 */
declare class PerformanceMonitoringTest {
    private monitoringSystem;
    private baseUrl;
    private testSymbols;
    constructor();
    /**
     * Run comprehensive performance monitoring test
     */
    runTest(): Promise<void>;
    /**
     * Test monitoring system initialization
     */
    private testMonitoringInitialization;
    /**
     * Test Prometheus metrics collection
     */
    private testPrometheusMetrics;
    /**
     * Test trading metrics recording
     */
    private testTradingMetricsRecording;
    /**
     * Test alert system functionality
     */
    private testAlertSystem;
    /**
     * Test performance metrics calculation
     */
    private testPerformanceMetricsCalculation;
    /**
     * Test API endpoints
     */
    private testApiEndpoints;
    /**
     * Test data quality monitoring
     */
    private testDataQualityMonitoring;
    /**
     * Test system health monitoring
     */
    private testSystemHealthMonitoring;
    /**
     * Validate performance metrics structure
     */
    private validatePerformanceMetrics;
    /**
     * Sleep utility
     */
    private sleep;
}
export { PerformanceMonitoringTest };
