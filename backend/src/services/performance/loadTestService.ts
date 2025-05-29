/**
 * Load Test Service
 * Handles load testing using k6
 */

import { v4 as uuidv4 } from 'uuid';
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { createLogger, LogData } from '../../utils/logger';
import {
  SystemLoadTest,
  LoadTestStage,
  PerformanceTestResult,
  TestStatus,
  PerformanceMetrics
} from '../../types/performance';

// Create logger
const logger = createLogger('LoadTestService');

// Temporary directory for k6 scripts
const TEMP_DIR = path.join(__dirname, '../../../temp');

/**
 * Load Test Service
 * Manages load testing using k6
 */
class LoadTestService {
  constructor() {
    // Ensure temp directory exists
    if (!fs.existsSync(TEMP_DIR)) {
      fs.mkdirSync(TEMP_DIR, { recursive: true });
    }
  }
  
  /**
   * Run a load test using k6
   * @param config - Load test configuration
   * @param onResultCallback - Callback for test results
   * @returns Test result ID
   */
  async runLoadTest(
    config: SystemLoadTest,
    onResultCallback: (result: PerformanceTestResult) => Promise<void>
  ): Promise<string> {
    try {
      logger.info(`Starting load test: ${config.name}`);
      
      // Generate unique result ID
      const resultId = uuidv4();
      
      // Create k6 script
      const scriptPath = await this.createK6Script(config);
      
      // Initialize result
      const initialResult: PerformanceTestResult = {
        id: resultId,
        testId: config.id || uuidv4(),
        testConfig: {
          id: config.id,
          name: config.name,
          description: config.description,
          testType: 'LOAD_TEST',
          duration: config.duration,
          concurrency: this.getMaxConcurrency(config.stages),
          options: config.options
        },
        status: TestStatus.RUNNING,
        startTime: new Date().toISOString(),
        metrics: {
          throughput: 0,
          latencyP50: 0,
          latencyP90: 0,
          latencyP95: 0,
          latencyP99: 0,
          latencyAvg: 0,
          latencyMin: 0,
          latencyMax: 0,
          errorRate: 0,
          successRate: 100
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      // Send initial result
      await onResultCallback(initialResult);
      
      // Run k6 in the background
      this.runK6(scriptPath, config, resultId, onResultCallback);
      
      return resultId;
    } catch (error) {
      const logData: LogData = {
        config,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error starting load test: ${config.name}`, logData);
      throw error;
    }
  }
  
  /**
   * Create k6 script file
   * @param config - Load test configuration
   * @returns Path to the script file
   * @private
   */
  private async createK6Script(config: SystemLoadTest): Promise<string> {
    try {
      const scriptId = uuidv4();
      const scriptPath = path.join(TEMP_DIR, `k6-test-${scriptId}.js`);
      
      // Prepare stages for k6
      const stagesJson = JSON.stringify(config.stages.map(stage => ({
        duration: `${stage.duration}s`,
        target: stage.target
      })));
      
      // Create k6 script content
      const scriptContent = `
import http from 'k6/http';
import { check, sleep } from 'k6';

// Test configuration
export const options = {
  stages: ${stagesJson},
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests should be below 500ms
    'http_req_failed': ['rate<0.01'], // Error rate should be below 1%
  },
};

// Test endpoints
const ENDPOINTS = ${JSON.stringify(config.targetEndpoints)};

export default function() {
  // Randomly select an endpoint
  const endpoint = ENDPOINTS[Math.floor(Math.random() * ENDPOINTS.length)];
  
  // Make request
  const response = http.get(endpoint);
  
  // Check response
  check(response, {
    'is status 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });
  
  // Wait between requests
  sleep(1);
}
      `;
      
      // Write script to file
      await fs.promises.writeFile(scriptPath, scriptContent);
      
      logger.debug(`Created k6 script: ${scriptPath}`);
      
      return scriptPath;
    } catch (error) {
      const logData: LogData = {
        config,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error creating k6 script for: ${config.name}`, logData);
      throw error;
    }
  }
  
  /**
   * Run k6 test
   * @param scriptPath - Path to the k6 script
   * @param config - Load test configuration
   * @param resultId - Test result ID
   * @param onResultCallback - Callback for test results
   * @private
   */
  private runK6(
    scriptPath: string,
    config: SystemLoadTest,
    resultId: string,
    onResultCallback: (result: PerformanceTestResult) => Promise<void>
  ): void {
    try {
      // Prepare k6 command
      const k6Process = spawn('k6', ['run', '--summary-export=stdout', scriptPath]);
      
      let output = '';
      let errorOutput = '';
      
      // Collect stdout
      k6Process.stdout.on('data', (data) => {
        output += data.toString();
        logger.debug(`k6 output: ${data.toString()}`);
      });
      
      // Collect stderr
      k6Process.stderr.on('data', (data) => {
        errorOutput += data.toString();
        logger.debug(`k6 error: ${data.toString()}`);
      });
      
      // Handle completion
      k6Process.on('close', async (code) => {
        logger.info(`k6 process exited with code ${code}`);
        
        try {
          // Clean up script file
          fs.unlinkSync(scriptPath);
          
          // Parse metrics from output
          const metrics = this.parseK6Output(output);
          
          // Determine test status
          const status = code === 0 ? TestStatus.COMPLETED : TestStatus.FAILED;
          
          // Prepare final result
          const finalResult: PerformanceTestResult = {
            id: resultId,
            testId: config.id || '',
            testConfig: {
              id: config.id,
              name: config.name,
              description: config.description,
              testType: 'LOAD_TEST',
              duration: config.duration,
              concurrency: this.getMaxConcurrency(config.stages),
              options: config.options
            },
            status,
            startTime: new Date(Date.now() - (config.duration * 1000)).toISOString(),
            endTime: new Date().toISOString(),
            duration: config.duration * 1000,
            metrics,
            errors: errorOutput ? [{
              timestamp: new Date().toISOString(),
              message: errorOutput,
              count: 1
            }] : undefined,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          };
          
          // Send final result
          await onResultCallback(finalResult);
        } catch (error) {
          logger.error(`Error processing k6 results: ${error instanceof Error ? error.message : String(error)}`);
          
          // Send error result
          await onResultCallback({
            id: resultId,
            testId: config.id || '',
            testConfig: {
              id: config.id,
              name: config.name,
              description: config.description,
              testType: 'LOAD_TEST',
              duration: config.duration,
              concurrency: this.getMaxConcurrency(config.stages),
              options: config.options
            },
            status: TestStatus.FAILED,
            startTime: new Date(Date.now() - (config.duration * 1000)).toISOString(),
            endTime: new Date().toISOString(),
            duration: config.duration * 1000,
            metrics: {
              throughput: 0,
              latencyP50: 0,
              latencyP90: 0,
              latencyP95: 0,
              latencyP99: 0,
              latencyAvg: 0,
              latencyMin: 0,
              latencyMax: 0,
              errorRate: 100,
              successRate: 0
            },
            errors: [{
              timestamp: new Date().toISOString(),
              message: error instanceof Error ? error.message : String(error),
              count: 1
            }],
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
          });
        }
      });
    } catch (error) {
      const logData: LogData = {
        scriptPath,
        resultId,
        error: error instanceof Error ? error.message : String(error)
      };
      
      logger.error(`Error running k6 for test: ${resultId}`, logData);
      throw error;
    }
  }
  
  /**
   * Parse k6 output to extract metrics
   * @param output - k6 output
   * @returns Performance metrics
   * @private
   */
  private parseK6Output(output: string): PerformanceMetrics {
    try {
      // Default metrics
      const metrics: PerformanceMetrics = {
        throughput: 0,
        latencyP50: 0,
        latencyP90: 0,
        latencyP95: 0,
        latencyP99: 0,
        latencyAvg: 0,
        latencyMin: 0,
        latencyMax: 0,
        errorRate: 0,
        successRate: 100,
        cpuUsageAvg: 0,
        memoryUsageAvg: 0
      };
      
      // Extract request rate
      const requestRateMatch = output.match(/http_reqs\\.+([0-9.]+)/);
      if (requestRateMatch && requestRateMatch[1]) {
        metrics.throughput = parseFloat(requestRateMatch[1]);
      }
      
      // Extract latencies
      const avgLatencyMatch = output.match(/http_req_duration\\s+avg=([0-9.]+)/);
      if (avgLatencyMatch && avgLatencyMatch[1]) {
        metrics.latencyAvg = parseFloat(avgLatencyMatch[1]);
      }
      
      const minLatencyMatch = output.match(/http_req_duration\\s+min=([0-9.]+)/);
      if (minLatencyMatch && minLatencyMatch[1]) {
        metrics.latencyMin = parseFloat(minLatencyMatch[1]);
      }
      
      const maxLatencyMatch = output.match(/http_req_duration\\s+max=([0-9.]+)/);
      if (maxLatencyMatch && maxLatencyMatch[1]) {
        metrics.latencyMax = parseFloat(maxLatencyMatch[1]);
      }
      
      const p50LatencyMatch = output.match(/http_req_duration\\s+p\\(50\\)=([0-9.]+)/);
      if (p50LatencyMatch && p50LatencyMatch[1]) {
        metrics.latencyP50 = parseFloat(p50LatencyMatch[1]);
      }
      
      const p90LatencyMatch = output.match(/http_req_duration\\s+p\\(90\\)=([0-9.]+)/);
      if (p90LatencyMatch && p90LatencyMatch[1]) {
        metrics.latencyP90 = parseFloat(p90LatencyMatch[1]);
      }
      
      const p95LatencyMatch = output.match(/http_req_duration\\s+p\\(95\\)=([0-9.]+)/);
      if (p95LatencyMatch && p95LatencyMatch[1]) {
        metrics.latencyP95 = parseFloat(p95LatencyMatch[1]);
      }
      
      const p99LatencyMatch = output.match(/http_req_duration\\s+p\\(99\\)=([0-9.]+)/);
      if (p99LatencyMatch && p99LatencyMatch[1]) {
        metrics.latencyP99 = parseFloat(p99LatencyMatch[1]);
      }
      
      // Extract error rate
      const errorRateMatch = output.match(/http_req_failed\\s+value=([0-9.]+)/);
      if (errorRateMatch && errorRateMatch[1]) {
        metrics.errorRate = parseFloat(errorRateMatch[1]) * 100;
        metrics.successRate = 100 - metrics.errorRate;
      }
      
      return metrics;
    } catch (error) {
      logger.error(`Error parsing k6 output: ${error instanceof Error ? error.message : String(error)}`);
      
      // Return default metrics on error
      return {
        throughput: 0,
        latencyP50: 0,
        latencyP90: 0,
        latencyP95: 0,
        latencyP99: 0,
        latencyAvg: 0,
        latencyMin: 0,
        latencyMax: 0,
        errorRate: 0,
        successRate: 100
      };
    }
  }
  
  /**
   * Get maximum concurrency from stages
   * @param stages - Load test stages
   * @returns Maximum concurrency
   * @private
   */
  private getMaxConcurrency(stages: LoadTestStage[]): number {
    if (!stages || stages.length === 0) {
      return 0;
    }
    
    return Math.max(...stages.map(stage => stage.target));
  }
}

// Create singleton instance
const loadTestService = new LoadTestService();

export default loadTestService; 