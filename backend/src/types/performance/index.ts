/**
 * Performance Testing Types
 * Interfaces for performance testing and optimization
 */

import { Timestamp } from '../common';

/**
 * Performance Test Configuration
 */
export interface PerformanceTestConfig {
  id?: string;
  name: string;
  description?: string;
  testType: PerformanceTestType;
  duration: number; // Duration in seconds
  concurrency: number; // Number of concurrent virtual users
  rampUp?: number; // Ramp-up period in seconds
  targetEndpoint?: string; // For API tests
  modelId?: string; // For ML model tests
  strategyId?: string; // For strategy tests
  symbol?: string;
  timeframe?: string;
  options?: Record<string, any>;
  createdAt?: string;
  updatedAt?: string;
}

/**
 * Performance Test Type
 */
export enum PerformanceTestType {
  API_LATENCY = 'API_LATENCY',
  ML_PREDICTION_THROUGHPUT = 'ML_PREDICTION_THROUGHPUT',
  SIGNAL_GENERATION = 'SIGNAL_GENERATION',
  STRATEGY_EXECUTION = 'STRATEGY_EXECUTION',
  END_TO_END = 'END_TO_END',
  LOAD_TEST = 'LOAD_TEST',
  STRESS_TEST = 'STRESS_TEST'
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
  duration?: number; // Actual duration in milliseconds
  metrics: PerformanceMetrics;
  errors?: PerformanceError[];
  createdAt: string;
  updatedAt: string;
}

/**
 * Test Status
 */
export enum TestStatus {
  RUNNING = 'RUNNING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED'
}

/**
 * Performance Metrics
 */
export interface PerformanceMetrics {
  // Common metrics
  throughput: number; // Requests/operations per second
  latencyP50: number; // Median latency in ms
  latencyP90: number; // 90th percentile latency in ms
  latencyP95: number; // 95th percentile latency in ms
  latencyP99: number; // 99th percentile latency in ms
  latencyAvg: number; // Average latency in ms
  latencyMin: number; // Minimum latency in ms
  latencyMax: number; // Maximum latency in ms
  errorRate: number; // Error rate percentage
  successRate: number; // Success rate percentage
  
  // CPU/Memory metrics
  cpuUsageAvg?: number; // Average CPU usage percentage
  cpuUsageMax?: number; // Maximum CPU usage percentage
  memoryUsageAvg?: number; // Average memory usage in MB
  memoryUsageMax?: number; // Maximum memory usage in MB
  
  // ML-specific metrics
  predictionLatencyAvg?: number; // Average prediction time in ms
  predictionThroughput?: number; // Predictions per second
  modelLoadTime?: number; // Time to load model in ms
  
  // Signal generation metrics
  signalGenerationLatencyAvg?: number; // Average signal generation time in ms
  signalsThroughput?: number; // Signals per second
  
  // Strategy execution metrics
  strategyExecutionLatencyAvg?: number; // Average strategy execution time in ms
  strategyThroughput?: number; // Strategy executions per second
  
  // End-to-end metrics
  endToEndLatencyAvg?: number; // Average end-to-end time in ms
  endToEndThroughput?: number; // End-to-end operations per second
  
  // Custom metrics
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
export enum OptimizationCategory {
  CACHING = 'CACHING',
  DATABASE = 'DATABASE',
  ML_MODEL = 'ML_MODEL',
  API_ENDPOINT = 'API_ENDPOINT',
  CONCURRENCY = 'CONCURRENCY',
  MEMORY_USAGE = 'MEMORY_USAGE',
  CODE_OPTIMIZATION = 'CODE_OPTIMIZATION',
  CONFIGURATION = 'CONFIGURATION'
}

/**
 * Optimization Impact
 */
export enum OptimizationImpact {
  LOW = 'LOW',
  MEDIUM = 'MEDIUM',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

/**
 * A/B Test Configuration
 */
export interface ABTestConfig {
  id?: string;
  name: string;
  description?: string;
  variantA: string; // Reference to a configuration, model, or strategy
  variantB: string; // Reference to a configuration, model, or strategy
  type: ABTestType;
  metric: string; // The metric to compare
  targetImprovement: number; // Percentage improvement target
  status: ABTestStatus;
  startDate?: string;
  endDate?: string;
  createdAt?: string;
  updatedAt?: string;
}

/**
 * A/B Test Type
 */
export enum ABTestType {
  ML_MODEL = 'ML_MODEL',
  STRATEGY = 'STRATEGY',
  SIGNAL_GENERATION = 'SIGNAL_GENERATION',
  API_CONFIGURATION = 'API_CONFIGURATION'
}

/**
 * A/B Test Status
 */
export enum ABTestStatus {
  DRAFT = 'DRAFT',
  RUNNING = 'RUNNING',
  COMPLETED = 'COMPLETED',
  CANCELLED = 'CANCELLED'
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
  improvement?: number; // Percentage improvement
  confidenceLevel?: number; // Statistical confidence level
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
  duration: number; // Duration in seconds
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
  duration: number; // Duration in seconds
  target: number; // Target users/requests
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
  refreshInterval: number; // In seconds
  layout?: any;
  createdAt?: string;
  updatedAt?: string;
} 