/**
 * Performance Test Service
 * Handles performance testing operations
 */
import { PerformanceTestConfig, PerformanceTestResult, PerformanceTestType } from '../../types/performance';
/**
 * Performance Test Service
 * Manages performance testing
 */
declare class PerformanceTestService {
    /**
     * Create a new performance test
     * @param config - Performance test configuration
     * @returns Created test configuration
     */
    createTest(config: PerformanceTestConfig): Promise<PerformanceTestConfig>;
    /**
     * Get performance test by ID
     * @param id - Test ID
     * @returns Test configuration
     */
    getTest(id: string): Promise<PerformanceTestConfig>;
    /**
     * Get all performance tests
     * @param testType - Optional filter by test type
     * @returns Array of test configurations
     */
    getAllTests(testType?: PerformanceTestType): Promise<PerformanceTestConfig[]>;
    /**
     * Start a performance test
     * @param id - Test ID
     * @returns Test result
     */
    startTest(id: string): Promise<PerformanceTestResult>;
    /**
     * Execute a performance test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    private executeTest;
    /**
     * Execute API latency test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    private executeApiLatencyTest;
    /**
     * Execute ML prediction test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    private executeMlPredictionTest;
    /**
     * Execute signal generation test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    private executeSignalGenerationTest;
    /**
     * Execute strategy execution test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    private executeStrategyExecutionTest;
    /**
     * Execute end-to-end test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    private executeEndToEndTest;
    /**
     * Execute load test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    private executeLoadTest;
    /**
     * Execute stress test
     * @param test - Test configuration
     * @param resultId - Test result ID
     * @private
     */
    private executeStressTest;
    /**
     * Update test result
     * @param resultId - Test result ID
     * @param update - Update data
     * @private
     */
    private updateTestResult;
    /**
     * Generate optimization recommendations
     * @param resultId - Test result ID
     * @private
     */
    private generateRecommendations;
    /**
     * Get test result by ID
     * @param id - Result ID
     * @returns Test result
     */
    getTestResult(id: string): Promise<PerformanceTestResult>;
    /**
     * Get all test results for a test
     * @param testId - Test ID
     * @returns Array of test results
     */
    getTestResults(testId: string): Promise<PerformanceTestResult[]>;
}
declare const perfTestService: PerformanceTestService;
export default perfTestService;
