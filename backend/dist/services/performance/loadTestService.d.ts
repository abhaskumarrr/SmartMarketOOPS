/**
 * Load Test Service
 * Handles load testing using k6
 */
import { SystemLoadTest, PerformanceTestResult } from '../../types/performance';
/**
 * Load Test Service
 * Manages load testing using k6
 */
declare class LoadTestService {
    constructor();
    /**
     * Run a load test using k6
     * @param config - Load test configuration
     * @param onResultCallback - Callback for test results
     * @returns Test result ID
     */
    runLoadTest(config: SystemLoadTest, onResultCallback: (result: PerformanceTestResult) => Promise<void>): Promise<string>;
    /**
     * Create k6 script file
     * @param config - Load test configuration
     * @returns Path to the script file
     * @private
     */
    private createK6Script;
    /**
     * Run k6 test
     * @param scriptPath - Path to the k6 script
     * @param config - Load test configuration
     * @param resultId - Test result ID
     * @param onResultCallback - Callback for test results
     * @private
     */
    private runK6;
    /**
     * Parse k6 output to extract metrics
     * @param output - k6 output
     * @returns Performance metrics
     * @private
     */
    private parseK6Output;
    /**
     * Get maximum concurrency from stages
     * @param stages - Load test stages
     * @returns Maximum concurrency
     * @private
     */
    private getMaxConcurrency;
}
declare const loadTestService: LoadTestService;
export default loadTestService;
