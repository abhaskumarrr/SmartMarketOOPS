/**
 * Performance Testing Types
 * Interfaces for performance testing and optimization
 */
/**
 * Performance Test Configuration
 */
export interface PerformanceTestConfig {
    id?: string;
    name: string;
    description?: string;
    testType: PerformanceTestType;
    duration: number;
    concurrency: number;
    rampUp?: number;
    targetEndpoint?: string;
    modelId?: string;
    strategyId?: string;
    symbol?: string;
    timeframe?: string;
    options?: Record<string, any>;
    createdAt?: string;
    updatedAt?: string;
}
/**
 * Performance Test Type
 */
export declare enum PerformanceTestType {
    API_LATENCY = "API_LATENCY",
    ML_PREDICTION_THROUGHPUT = "ML_PREDICTION_THROUGHPUT",
    SIGNAL_GENERATION = "SIGNAL_GENERATION",
    STRATEGY_EXECUTION = "STRATEGY_EXECUTION",
    END_TO_END = "END_TO_END",
    LOAD_TEST = "LOAD_TEST",
    STRESS_TEST = "STRESS_TEST"
}
/**
 * Performance Test Result
 */
export interface PerformanceTestResult {
    id: string;
    testId: string;
    testConfig: PerformanceTestConfig;
    status: TestStatus;
    startTime: string;
    endTime?: string;
    duration?: number;
    metrics: PerformanceMetrics;
    errors?: PerformanceError[];
    createdAt: string;
    updatedAt: string;
}
/**
 * Test Status
 */
export declare enum TestStatus {
    RUNNING = "RUNNING",
    COMPLETED = "COMPLETED",
    FAILED = "FAILED",
    CANCELLED = "CANCELLED"
}
/**
 * Performance Metrics
 */
export interface PerformanceMetrics {
    throughput: number;
    latencyP50: number;
    latencyP90: number;
    latencyP95: number;
    latencyP99: number;
    latencyAvg: number;
    latencyMin: number;
    latencyMax: number;
    errorRate: number;
    successRate: number;
    cpuUsageAvg?: number;
    cpuUsageMax?: number;
    memoryUsageAvg?: number;
    memoryUsageMax?: number;
    predictionLatencyAvg?: number;
    predictionThroughput?: number;
    modelLoadTime?: number;
    signalGenerationLatencyAvg?: number;
    signalsThroughput?: number;
    strategyExecutionLatencyAvg?: number;
    strategyThroughput?: number;
    endToEndLatencyAvg?: number;
    endToEndThroughput?: number;
    customMetrics?: Record<string, number>;
    timeSeriesData?: PerformanceTimeSeries[];
}
/**
 * Performance Time Series
 */
export interface PerformanceTimeSeries {
    timestamp: string;
    metric: string;
    value: number;
}
/**
 * Performance Error
 */
export interface PerformanceError {
    timestamp: string;
    message: string;
    code?: string;
    count: number;
}
/**
 * Performance Optimization Recommendation
 */
export interface OptimizationRecommendation {
    id: string;
    testId: string;
    category: OptimizationCategory;
    impact: OptimizationImpact;
    description: string;
    implementation?: string;
    estimatedImprovement?: string;
    createdAt: string;
}
/**
 * Optimization Category
 */
export declare enum OptimizationCategory {
    CACHING = "CACHING",
    DATABASE = "DATABASE",
    ML_MODEL = "ML_MODEL",
    API_ENDPOINT = "API_ENDPOINT",
    CONCURRENCY = "CONCURRENCY",
    MEMORY_USAGE = "MEMORY_USAGE",
    CODE_OPTIMIZATION = "CODE_OPTIMIZATION",
    CONFIGURATION = "CONFIGURATION"
}
/**
 * Optimization Impact
 */
export declare enum OptimizationImpact {
    LOW = "LOW",
    MEDIUM = "MEDIUM",
    HIGH = "HIGH",
    CRITICAL = "CRITICAL"
}
/**
 * A/B Test Configuration
 */
export interface ABTestConfig {
    id?: string;
    name: string;
    description?: string;
    variantA: string;
    variantB: string;
    type: ABTestType;
    metric: string;
    targetImprovement: number;
    status: ABTestStatus;
    startDate?: string;
    endDate?: string;
    createdAt?: string;
    updatedAt?: string;
}
/**
 * A/B Test Type
 */
export declare enum ABTestType {
    ML_MODEL = "ML_MODEL",
    STRATEGY = "STRATEGY",
    SIGNAL_GENERATION = "SIGNAL_GENERATION",
    API_CONFIGURATION = "API_CONFIGURATION"
}
/**
 * A/B Test Status
 */
export declare enum ABTestStatus {
    DRAFT = "DRAFT",
    RUNNING = "RUNNING",
    COMPLETED = "COMPLETED",
    CANCELLED = "CANCELLED"
}
/**
 * A/B Test Result
 */
export interface ABTestResult {
    id: string;
    testId: string;
    status: ABTestStatus;
    startDate: string;
    endDate?: string;
    variantAMetrics: Record<string, number>;
    variantBMetrics: Record<string, number>;
    winner?: 'A' | 'B' | 'INCONCLUSIVE';
    improvement?: number;
    confidenceLevel?: number;
    notes?: string;
    createdAt: string;
    updatedAt: string;
}
/**
 * System Load Test
 */
export interface SystemLoadTest {
    id?: string;
    name: string;
    description?: string;
    duration: number;
    stages: LoadTestStage[];
    targetEndpoints: string[];
    options?: Record<string, any>;
    createdAt?: string;
    updatedAt?: string;
}
/**
 * Load Test Stage
 */
export interface LoadTestStage {
    duration: number;
    target: number;
}
/**
 * Performance Dashboard Config
 */
export interface PerformanceDashboardConfig {
    id?: string;
    name: string;
    description?: string;
    metrics: string[];
    timeRange: string;
    refreshInterval: number;
    layout?: any;
    createdAt?: string;
    updatedAt?: string;
}
